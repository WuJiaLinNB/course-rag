#include <core/vector_store.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

namespace core {

namespace {
constexpr char kMagic[4] = {'C','R','V','1'};
constexpr uint32_t kVersion = 1;

// 原子替换：全部写完后一步替换，绝不留半个正式文件
void atomic_replace(const std::string& tmp, const std::string& path) {
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("atomic replace failed: " + path);
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("atomic replace failed: " + path);
#endif
}
} // namespace

VectorStore::VectorStore(size_t dim) : dim_(dim) {
    if (dim == 0) throw std::invalid_argument("dim must be > 0");
}

void VectorStore::add(std::vector<float> v) {
    if (v.size() != dim_)
        throw std::invalid_argument("dimension mismatch");
    float n2 = 0.0f;
    for (float x : v) n2 += x * x;
    if (n2 == 0.0f)
        throw std::invalid_argument("zero vector rejected");
    float inv = 1.0f / std::sqrt(n2);
    flat_.reserve(flat_.size() + dim_);   // 先确保容量：中途抛 bad_alloc 也不会留下半条向量
    for (float x : v) flat_.push_back(x * inv);
}

size_t VectorStore::size() const { return flat_.size() / dim_; }
size_t VectorStore::dim() const { return dim_; }
std::vector<float> VectorStore::get(size_t i) const {
    if (i >= size())
        throw std::out_of_range("VectorStore::get: index " + std::to_string(i)
                                + " >= size " + std::to_string(size()));
    return std::vector<float>(flat_.begin() + i * dim_,
                              flat_.begin() + (i + 1) * dim_);
}
const float* VectorStore::raw() const { return flat_.data(); }
void VectorStore::reserve(size_t n) { flat_.reserve(n * dim_); }

void VectorStore::save(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("cannot write: " + tmp);
        const uint32_t dim32 = (uint32_t)dim_;
        const uint32_t count = (uint32_t)size();
        f.write(kMagic, 4);
        f.write((const char*)&kVersion, 4);
        f.write((const char*)&dim32, 4);
        f.write((const char*)&count, 4);
        f.write((const char*)flat_.data(), (std::streamsize)(flat_.size() * sizeof(float)));
        f.flush();
        if (!f) throw std::runtime_error("write failed: " + tmp);
    }                                   // 离开作用域 = 文件确实关闭落盘，之后才替换
    atomic_replace(tmp, path);
}

void VectorStore::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    char magic[4]; uint32_t version = 0, dim32 = 0, count = 0;
    f.read(magic, 4);
    f.read((char*)&version, 4);
    f.read((char*)&dim32, 4);
    f.read((char*)&count, 4);
    if (!f) throw std::runtime_error("vectors.bin header incomplete: " + path);
    if (std::memcmp(magic, kMagic, 4) != 0)
        throw std::runtime_error("bad magic: not a CRV1 vectors file");
    if (version != kVersion)
        throw std::runtime_error("unsupported vectors.bin version");
    if (dim32 != dim_)
        throw std::runtime_error("dim mismatch: file=" + std::to_string(dim32)
                                 + " config=" + std::to_string(dim_));
    // 字节数必须与文件头 count 自洽（截断/损坏文件在此被拒）
    f.seekg(0, std::ios::end);
    const auto bytes = (size_t)f.tellg();
    if (bytes != 16 + (size_t)count * dim_ * sizeof(float))
        throw std::runtime_error("file size inconsistent with header count");
    // 先读入局部缓冲，全部成功后才换入成员（强异常保证：失败不留中间态）
    std::vector<float> tmp;
    tmp.resize((size_t)count * dim_);
    f.seekg(16, std::ios::beg);
    f.read((char*)tmp.data(), (std::streamsize)(tmp.size() * sizeof(float)));
    if (!f) throw std::runtime_error("short read: " + path);
    flat_.swap(tmp);
}

} // namespace core
