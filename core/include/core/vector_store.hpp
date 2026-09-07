#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace core {

// SoA 扁平向量存储：所有向量连续存于一个 vector<float>
// 入库即归一化（项目铁律）；raw() 供索引层做高性能遍历
class VectorStore {
public:
    explicit VectorStore(size_t dim);

    void add(std::vector<float> v);        // 归一化后追加；维度不符/零向量抛异常
    size_t size() const;
    size_t dim() const;
    std::vector<float> get(size_t i) const;
    const float* raw() const;              // 底层大数组（SoA），第 i 个起点 = i*dim_
    void reserve(size_t n);

    // 落盘/载入 vectors.bin（仅向量；元数据 chunks.json 归 server 层管）
    // 文件头 16 字节：magic "CRV1"(4) + version(uint32) + dim(uint32) + count(uint32)
    void save(const std::string& path) const;   // 原子写：先写 .tmp 再原子替换
    void load(const std::string& path);         // 校验失败抛异常，拒绝带病载入

private:
    size_t dim_;
    std::vector<float> flat_;
};

} // namespace core
