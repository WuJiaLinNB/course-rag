// Task 13 基准测试：同一批固定种子合成数据灌三种索引（brute / ivf / hnsw），
// 同一批查询测召回率@10、单线程 P50/P99 查询延迟、QPS、建索引耗时与内存占用。
// stdout 输出 markdown 表格，可直接粘贴进 docs/benchmark.md。
//
// 用法：course-rag-bench [n] [dim] [queries]
//   n       库内向量数（默认 100000；项目约束 IVF 训练需 ≥10000）
//   dim     向量维度（默认 1024）
//   queries 查询条数（默认 200）
//
// 可移植性：只使用标准设施（mt19937 / chrono / printf）；内存实测用
// GetProcessMemoryInfo，以 #ifdef _WIN32 守卫——Linux CI 只编译不运行，
// 非 Windows 打印 N/A。无任何 MSVC 特有语法。

#include <core/brute_index.hpp>
#include <core/hnsw_index.hpp>
#include <core/ivf_index.hpp>
#include <core/index.hpp>
#include <core/types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX   // windows.h 的 min/max 宏会打断 std::min/std::max，必须屏蔽
#include <windows.h>
#include <psapi.h>
#endif

using namespace core;

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t kSeed = 1234;   // 固定种子：数据与查询完全可复现

double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
double elapsed_us(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// 输出助手：格式串经 ... 转发到 vfprintf（与 log.hpp 的 LOG_IMPL 同思路）。
// 不直接用 std::printf 的原因：MSVC 14.29 的 printf 格式分析器对"含中文的
// 窄字面量 + 多个 % 说明符"存在误报（实测 C4819 代码页(0) / C4477 参数错配，
// 运行时输出已逐项验证正确）；非字面量格式串不参与该分析，警告归零
void outf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
}

// ---------- 合成数据生成器（固定种子，逐位可复现） ----------
// 结构：kClusters 个簇的混合高斯——簇中心 = g × 随机方向（每维 N(0,1) 后整体
// 缩放到幅度 g），点 = 所选簇中心 + sigma × 每维 N(0,1) 噪声。
// 向量保持原始幅度，归一化交给索引入库（VectorStore::add 内部做）；余弦排序
// 对正缩放不变，查询不预归一化也不影响结果（brute/ivf/hnsw 的 search 内部
// 都会对查询向量归一化后再算分）。
// 高维几何：同簇点余弦 ≈ g²/(g²+sigma²·dim)，跨簇 ≈ 0（波动 ≈ 同簇/√dim），
// 簇间界限清晰 → 召回差异主要来自索引结构本身，而非数据病态。
struct Dataset {
    std::vector<std::vector<float>> data;     // n 条库内向量
    std::vector<std::vector<float>> queries;  // nq 条查询
};

Dataset make_dataset(size_t n, size_t dim, size_t nq, uint32_t seed) {
    constexpr size_t kClusters = 64;    // 簇数（真实语料按主题聚簇的近似）
    constexpr double kCenterG = 4.0;    // 中心幅度 g
    constexpr double kSigma = 0.12;     // 簇内噪声 sigma

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> pick(0, kClusters - 1);

    // 1) 簇中心：随机方向 × 幅度 g（先采完所有中心，随机流次序固定）
    std::vector<std::vector<float>> centers(kClusters, std::vector<float>(dim));
    for (auto& c : centers) {
        double norm2 = 0.0;
        for (auto& x : c) { x = gauss(rng); norm2 += (double)x * x; }
        const float scale = (float)(kCenterG / std::sqrt(norm2));
        for (auto& x : c) x *= scale;
    }

    // 2) 点 = 中心 + sigma × 高斯噪声。先 n 条库内向量、后 nq 条查询，
    //    全部取自同一随机流（与 test_consistency.cpp 同款写法）
    auto sample = [&](const std::vector<float>& c) {
        std::vector<float> v(dim);
        for (size_t d = 0; d < dim; ++d) v[d] = c[d] + kSigma * gauss(rng);
        return v;
    };
    Dataset ds;
    ds.data.reserve(n);
    for (size_t i = 0; i < n; ++i) ds.data.push_back(sample(centers[pick(rng)]));
    ds.queries.reserve(nq);
    for (size_t i = 0; i < nq; ++i) ds.queries.push_back(sample(centers[pick(rng)]));
    return ds;
}

// ---------- 内存实测（Windows 专属） ----------
// 进程工作集（物理内存占用，字节）。非 Windows 返回 false → 调用处打印 N/A。
bool query_working_set(size_t& out) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return false;
    out = pmc.WorkingSetSize;
    return true;
#else
    (void)out;
    return false;
#endif
}

// ---------- 基准结果 ----------
struct Row {
    std::string name;         // 索引名 + 关键参数
    double recall = 0.0;      // 召回率@10（以 brute 结果为 ground truth）
    double p50_us = 0.0;
    double p99_us = 0.0;
    double qps = 0.0;
    double build_s = 0.0;     // 建索引耗时；<0 表示复用上文索引（打印 —）
    double mem_mb = 0.0;      // 建索引前后进程工作集增量（实测）
    bool mem_ok = false;      // 非 Windows = false → 打印 N/A
};

struct QueryStats {
    double p50_us = 0.0, p99_us = 0.0, qps = 0.0;
    std::vector<std::vector<uint32_t>> top10;   // 每条查询返回的 top-10 id
};

// 最近邻分位数（lat 已升序排列）：nearest-rank 定义
double pct(const std::vector<double>& lat_sorted, double p) {
    const size_t rank = (size_t)std::ceil(p * (double)lat_sorted.size());
    return lat_sorted[std::min(lat_sorted.size() - 1, rank - 1)];
}

// 对单个已建好的索引执行查询基准：逐查询计时（steady_clock，单线程）→
// P50/P99/QPS；top-10 结果留档供召回计算
QueryStats run_queries(const Index& idx,
                       const std::vector<std::vector<float>>& queries) {
    // 预热一次：首查询的缺页 / 分配器冷启动不计入统计
    if (!queries.empty()) idx.search(queries[0], 10, {});

    QueryStats st;
    st.top10.resize(queries.size());
    std::vector<double> lat;
    lat.reserve(queries.size());

    const auto t0 = Clock::now();
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto q0 = Clock::now();
        auto r = idx.search(queries[i], 10, {});
        lat.push_back(elapsed_us(q0));
        for (const auto& it : r) st.top10[i].push_back(it.id);
    }
    const double total_s = elapsed_s(t0);

    std::sort(lat.begin(), lat.end());
    st.p50_us = pct(lat, 0.50);
    st.p99_us = pct(lat, 0.99);
    st.qps = (double)queries.size() / total_s;
    return st;
}

// 召回率@10：|待测 top-10 ∩ ground truth top-10| / (查询数 × 10)
double recall_at10(const QueryStats& r, const QueryStats& gt) {
    const size_t q = std::min(r.top10.size(), gt.top10.size());
    size_t hits = 0;
    for (size_t i = 0; i < q; ++i)
        for (uint32_t id : r.top10[i])
            if (std::find(gt.top10[i].begin(), gt.top10[i].end(), id) != gt.top10[i].end())
                ++hits;
    return q ? (double)hits / (double)(q * 10) : 0.0;
}

std::string fmt(double v, const char* spec) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), spec, v);
    return buf;
}

void print_table(const std::vector<Row>& rows, size_t n, size_t dim) {
    // 理论值：向量本体 n × dim × 4 bytes（各索引都存一份归一化后的向量）
    const double theo_mb = (double)n * (double)dim * 4.0 / 1048576.0;
    outf("\n| 索引 | 召回率@10 | P50 延迟 (μs) | P99 延迟 (μs) | QPS (单线程) | 建索引 (s) | 内存理论 (MB) | 内存实测 (MB) |\n");
    outf("|---|---|---|---|---|---|---|---|\n");
    for (const Row& r : rows) {
        const bool reuse = r.build_s < 0.0;   // 复用上文索引（仅调检索参数）：建索引与内存列打 —
        const std::string build = reuse ? "—" : fmt(r.build_s, "%.1f");
        const std::string mem = reuse ? "—" : (r.mem_ok ? fmt(r.mem_mb, "%.1f") : "N/A");
        outf("| %s | %s | %s | %s | %s | %s | %.1f | %s |\n",
             r.name.c_str(), fmt(r.recall, "%.3f").c_str(),
             fmt(r.p50_us, "%.1f").c_str(), fmt(r.p99_us, "%.1f").c_str(),
             fmt(r.qps, "%.1f").c_str(), build.c_str(), theo_mb, mem.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    // ---- 参数解析：argv 覆盖默认值，非法即报错退出，最后打印实际值 ----
    size_t n = 100000, dim = 1024, nq = 200;
    auto parse = [&](const char* s, size_t& out, const char* what) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(s, &end, 10);
        if (end == s || *end != '\0' || v == 0) {
            std::fprintf(stderr, "参数 %s 非法: %s（必须是正整数）\n", what, s);
            std::exit(2);
        }
        out = (size_t)v;
    };
    if (argc > 1) parse(argv[1], n, "n");
    if (argc > 2) parse(argv[2], dim, "dim");
    if (argc > 3) parse(argv[3], nq, "queries");

    outf("# course-rag 三索引基准（Task 13）\n\n");
    outf("- 数据来源：固定种子合成向量（非真实语料）\n");
    outf("- 生成器：64 簇混合高斯，中心幅度 g=4.0，簇内噪声 sigma=0.12，种子=%u\n", kSeed);
    outf("- 实际参数：n=%zu, dim=%zu, queries=%zu\n", n, dim, nq);
    outf("- IVF：min_train=max(n,10000)（批量构建语义：全部灌入后训练一次），k_centroids=32，nprobe=8\n");
    if (n < 10000)
        outf("- 注意：n=%zu < 10000，IVF 不训练（项目约束），search 自动降级为暴力扫描\n", n);
    outf("- HNSW：M=16, efC=200（与 engine.cpp 同款）；efS 测 16/64/128/256 四档（复用同一索引）\n");
    outf("- 相似度：余弦（入库归一化，点积即余弦）；召回@10 以 brute 结果为 ground truth\n");
    outf("- 计时：steady_clock 单线程逐查询计时（含 1 次预热）；QPS = 查询数 / 总耗时\n");
    outf("- 内存：GetProcessMemoryInfo 工作集增量（建索引前后差值，含分配器保留）\n\n");

    // ---- 生成数据 ----
    const auto t_gen = Clock::now();
    Dataset ds = make_dataset(n, dim, nq, kSeed);
    outf("数据生成完成：%zu 条库内 + %zu 条查询，耗时 %.1f s\n\n",
                ds.data.size(), ds.queries.size(), elapsed_s(t_gen));

    std::vector<Row> rows;
    QueryStats gt;   // brute 的 top-10 即 ground truth

    // ---- 1) brute：先跑，结果直接作为 ground truth；brute 对自身召回 = 1.0（自洽校验） ----
    {
        size_t ws0 = 0;
        const bool ok0 = query_working_set(ws0);
        BruteIndex idx(dim);
        const auto t0 = Clock::now();
        const ChunkMeta meta{};
        for (const auto& v : ds.data) idx.add(v, meta);
        const double build_s = elapsed_s(t0);
        size_t ws1 = 0;
        const bool ok1 = ok0 && query_working_set(ws1);

        QueryStats st = run_queries(idx, ds.queries);
        gt = st;
        Row r;
        r.name = "brute (ground truth)";
        r.recall = recall_at10(st, st);   // 自洽校验：应为 1.000
        r.p50_us = st.p50_us; r.p99_us = st.p99_us; r.qps = st.qps;
        r.build_s = build_s;
        r.mem_mb = ok1 ? (double)(ws1 - ws0) / 1048576.0 : 0.0;
        r.mem_ok = ok1;
        rows.push_back(std::move(r));
    }
    // brute 在此析构：约 400MB 的扁平存储是大块分配，归还系统 → IVF 的基线干净

    // ---- 2) ivf：min_train=max(n,10000) —— n 达标时只在最后一次 add 训练一次
    //      （避免流式重训开销混入建索引耗时）；k=32、nprobe=8 均为默认值 ----
    {
        size_t ws0 = 0;
        const bool ok0 = query_working_set(ws0);
        IvfIndex idx(dim, /*min_train=*/std::max<size_t>(n, 10000),
                     /*k_centroids=*/32, /*nprobe=*/8);
        const auto t0 = Clock::now();
        const ChunkMeta meta{};
        for (const auto& v : ds.data) idx.add(v, meta);   // 最后一次 add 触发 maybe_train
        const double build_s = elapsed_s(t0);
        size_t ws1 = 0;
        const bool ok1 = ok0 && query_working_set(ws1);
        outf("[check] ivf size=%zu trained=%d\n", idx.size(), idx.ready() ? 1 : 0);

        QueryStats st = run_queries(idx, ds.queries);
        Row r;
        r.name = "ivf (k=32, nprobe=8)";
        r.recall = recall_at10(st, gt);
        r.p50_us = st.p50_us; r.p99_us = st.p99_us; r.qps = st.qps;
        r.build_s = build_s;
        r.mem_mb = ok1 ? (double)(ws1 - ws0) / 1048576.0 : 0.0;
        r.mem_ok = ok1;
        rows.push_back(std::move(r));
    }

    // ---- 3) hnsw：M=16, efC=200（engine.cpp 同款）。efS 是精度/速度旋钮：
    //      全量基准首跑发现默认 efS=16 召回仅 0.568（1024 维 10 万条），
    //      故测 16/64/128/256 四档完整曲线（复用同一索引，仅调检索参数），
    //      供 engine 默认 efS 定档：选满足召回≥0.95 的最低档。
    {
        size_t ws0 = 0;
        const bool ok0 = query_working_set(ws0);
        HnswIndex idx(dim, /*M=*/16, /*ef_construction=*/200);
        const auto t0 = Clock::now();
        const ChunkMeta meta{};
        for (const auto& v : ds.data) idx.add(v, meta);
        const double build_s = elapsed_s(t0);
        size_t ws1 = 0;
        const bool ok1 = ok0 && query_working_set(ws1);
        outf("[check] hnsw size=%zu\n", idx.size());

        const size_t ef_list[] = {16, 64, 128, 256};
        for (size_t i = 0; i < sizeof(ef_list) / sizeof(ef_list[0]); ++i) {
            const size_t ef = ef_list[i];
            if (i > 0) idx.set_ef_search(ef);   // 第一档即构造默认 efS=16，后续档复用同一索引仅调检索参数
            QueryStats st = run_queries(idx, ds.queries);
            Row r;
            r.name = "hnsw (M=16, efC=200, efS=" + std::to_string(ef) + ")";
            r.recall = recall_at10(st, gt);
            r.p50_us = st.p50_us; r.p99_us = st.p99_us; r.qps = st.qps;
            if (i == 0) {
                r.build_s = build_s;
                r.mem_mb = ok1 ? (double)(ws1 - ws0) / 1048576.0 : 0.0;
                r.mem_ok = ok1;
            } else {
                r.build_s = -1.0;   // 复用同一索引：无重建成本，建索引/内存列打 —
                r.mem_ok = false;
            }
            rows.push_back(std::move(r));
        }
    }

    print_table(rows, n, dim);
    return 0;
}
