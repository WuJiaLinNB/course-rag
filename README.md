# course-rag

C++17 实现的课程资料 RAG（检索增强生成）系统：本地向量检索 + LLM 问答，面向秋招面试的"检索瓶颈在 LLM 之前"场景。

## 项目背景

课程资料问答的瓶颈不在 LLM 生成，而在**检索质量与延迟**：语料是几百章 Markdown 切成的片段，要在其中精确召回与问题相关的段落，再把结果喂给 LLM。本仓库自底向上实现三种向量索引并做工程化收口，暴露检索为 HTTP 服务，供前端问答与调试。

## 架构（四层）

```
┌─────────────────────────────────────────────────────────┐
│ 展示层：单文件网页 index.html（Task 15，鉴权 + 问答 + 调试）│
├─────────────────────────────────────────────────────────┤
│ 服务层：course-rag-server（HTTP：/search /ask /documents）│
│         鉴权中间件 + 限流 + 优雅停机 + 原子落盘            │
├─────────────────────────────────────────────────────────┤
│ 检索层：Engine 门面 → HnswIndex（默认）/ IvfIndex /       │
│         BruteIndex（ground truth + 降级）                 │
├─────────────────────────────────────────────────────────┤
│ 数据层：VectorStore（SoA 扁平数组 + 归一化 + 原子持久化）  │
│         data/vectors.bin（16 字节头）+ data/chunks.json    │
└─────────────────────────────────────────────────────────┘
离线：tools/embedder.py 把 Markdown 切成片段 → 调 embedding
      API → 生成 vectors.bin + chunks.json（core 不做在线 embedding）
```

## 快速开始

```bash
# 1. 离线预处理（Python）：Markdown 语料 → vectors.bin + chunks.json
python tools/embedder.py data/course.md

# 2. 构建（Windows + VS2019 / Ninja）
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release

# 3. 启动服务（默认 127.0.0.1:8080；环境变量见 server/src/main.cpp 顶部）
PORT=8080 EMBED_API_BASE=http://... LLM_API_BASE=http://... ./course-rag-server
```

## Benchmark（固定种子合成向量，n=100000, dim=1024, queries=200）

> 数据来源：合成向量（非真实语料）——在可控分布上对比三种索引的结构性差异；完整方法与复现方式见 [docs/benchmark.md](docs/benchmark.md)。

| 索引 | 召回率@10 | P50 延迟 (μs) | P99 延迟 (μs) | QPS (单线程) | 建索引 (s) | 内存理论 (MB) | 内存实测 (MB) |
|---|---|---|---|---|---|---|---|
| Brute | 1.000 | 117442 | 155124 | 8.3 | 0.5 | 390.6 | 405.7 |
| IVF (k=32, nprobe=8) | 1.000 | 32830 | 45232 | 29.8 | 185.0 | 390.6 | 409.8 |
| HNSW (efS=16) | 0.568 | 1077 | 3064 | 820.3 | 456.8 | 390.6 | 418.0 |
| HNSW (efS=64) | 0.908 | 1768 | 2883 | 551.3 | — | 390.6 | — |
| **HNSW (efS=128，生产默认)** | **0.978** | **2433** | **3789** | **397.3** | — | 390.6 | — |
| HNSW (efS=256) | 0.997 | 2935 | 4217 | 328.4 | — | 390.6 | — |

efS=16 召回不足（0.568）→ 生产默认定档 efS=128（0.978，P50 2.4ms，比 brute 快 48×）。完整方法与解读见 [docs/benchmark.md](docs/benchmark.md)。

## 设计取舍

完整推导见 [docs/design/2026-09-03-course-rag-design.md](docs/design/2026-09-03-course-rag-design.md)，要点：

- **SoA 扁平数组**：`vector<float>` 单数组 + 行偏移，避免 `vector<vector<float>>` 的指针跳转与缓存不连续
- **HNSW 默认 + 暴力兜底**：召回率 ≥95% 关卡把关；数据不足/重建窗口自动降级 brute，Engine 永远保留 brute 作 ground truth
- **重建热切换**：索引持久化只落原始向量（不序列化图），重启后台重建，期间继续服务旧索引，`std::shared_mutex` 原子指针交换
- **原子持久化**：临时文件 + 原子替换，16 字节文件头校验 + 跨文件条数一致性校验，拒绝带病载入
- **fail-fast**：embedding/LLM 调用固定超时（5s/30s）不重试，失败明确降级（502 / llm_ok=false）
- **线程模型**：`std::shared_mutex` 读写锁，读多写少；优雅停机（Ctrl+C/断连）先停接收再 flush

## AI 参与边界声明

本仓库代码与文档在 AI 辅助下编写；`docs/superpowers/plans/` 为任务拆解计划，实现经规格符合性 + 代码质量双阶段审查（subagent-driven development）。仓库副本为权威来源，`docs/superpowers/` 下计划/设计文档仅作参考。性能加速类断言（如 "10x faster"）标注 *pending benchmark validation*，以 [docs/benchmark.md](docs/benchmark.md) 实测为准。
