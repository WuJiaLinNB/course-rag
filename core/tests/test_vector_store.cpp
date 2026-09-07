#include <gtest/gtest.h>
#include <core/vector_store.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

using namespace core;

TEST(VectorStore, AddAndGet) {
    VectorStore vs(3);
    vs.add({1,0,0});
    vs.add({0,1,0});
    EXPECT_EQ(vs.size(), 2u);
    EXPECT_FLOAT_EQ(vs.get(0)[0], 1.0f);
}

TEST(VectorStore, NormalizesOnAdd) {
    VectorStore vs(2);
    vs.add({3,4});                     // 长度 5 → 归一化 (0.6, 0.8)
    auto v = vs.get(0);
    EXPECT_NEAR(v[0], 0.6f, 1e-6);
    EXPECT_NEAR(v[1], 0.8f, 1e-6);
}

TEST(VectorStore, FlatMemory) {
    // SoA 铁律：第 i 个向量起点 = i * dim（入库即归一化，断言用归一化后的值）
    VectorStore vs(4);
    vs.add({1,2,3,4});
    vs.add({5,6,7,8});
    const float* raw = vs.raw();
    EXPECT_NEAR(raw[0*4 + 2], 3.0f / std::sqrt(30.0f), 1e-6);   // 第 0 个向量的分量 2（原值 3）
    EXPECT_NEAR(raw[1*4 + 0], 5.0f / std::sqrt(174.0f), 1e-6);  // 第 1 个向量起点 = 1*dim（原值 5）
}

TEST(VectorStore, RejectWrongDim) {
    VectorStore vs(3);
    EXPECT_THROW(vs.add({1,2}), std::invalid_argument);
}

TEST(VectorStore, RejectZeroVector) {
    VectorStore vs(3);
    EXPECT_THROW(vs.add({0,0,0}), std::invalid_argument);
}

TEST(VectorStore, SaveLoadRoundtrip) {
    const std::string path = "test_vectors.bin";
    {
        VectorStore vs(2);
        vs.add({3,4});                     // 归一化 (0.6, 0.8)
        vs.add({1,0});
        vs.save(path);
    }
    VectorStore vs2(2);
    vs2.load(path);                        // 通过 magic/版本/维度/字节数全部校验
    EXPECT_EQ(vs2.size(), 2u);
    EXPECT_NEAR(vs2.get(0)[0], 0.6f, 1e-6);
    std::remove(path.c_str());
}

TEST(VectorStore, SaveOverwriteLeavesNoTmp) {
    // 原子写：二次保存 = 原子覆盖，且同目录不残留 .tmp
    const std::string path = "test_overwrite.bin";
    VectorStore vs(1);
    vs.add({2});                           // 归一化 (1)
    vs.save(path);
    vs.save(path);
    std::ifstream probe(path + ".tmp");
    EXPECT_FALSE(probe.good());            // .tmp 已被 rename/MoveFileEx 移走
    VectorStore vs2(1); vs2.load(path);
    EXPECT_EQ(vs2.size(), 1u);
    std::remove(path.c_str());
}

TEST(VectorStore, LoadRejectsBadMagic) {
    { std::ofstream f("bad_magic.bin", std::ios::binary); f << "XXXX" << 'x'; }
    VectorStore vs(2);
    EXPECT_THROW(vs.load("bad_magic.bin"), std::runtime_error);   // 防错配文件
    std::remove("bad_magic.bin");
}

TEST(VectorStore, LoadRejectsDimMismatch) {
    { std::ofstream f("bad_dim.bin", std::ios::binary);
      f.write("CRV1", 4);
      uint32_t v = 1, dim = 8, count = 0;  // 文件 dim=8 ≠ 存储 dim=2
      f.write((char*)&v,4); f.write((char*)&dim,4); f.write((char*)&count,4); }
    VectorStore vs(2);
    EXPECT_THROW(vs.load("bad_dim.bin"), std::runtime_error);
    std::remove("bad_dim.bin");
}

TEST(VectorStore, LoadRejectsTruncated) {
    // 字节数与文件头 count 不自洽 = 文件损坏 → 拒绝载入
    { std::ofstream f("trunc.bin", std::ios::binary);
      f.write("CRV1", 4);
      uint32_t v = 1, dim = 2, count = 10; // 声称 10 条，实际 0 字节数据
      f.write((char*)&v,4); f.write((char*)&dim,4); f.write((char*)&count,4); }
    VectorStore vs(2);
    EXPECT_THROW(vs.load("trunc.bin"), std::runtime_error);
    std::remove("trunc.bin");
}
