#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace core
{
    // 1. 一条检索结果：内部id（对应向量文件的第几行）+ 相似度分数
    struct SearchResult
    {
        uint32_t id;
        float similarity;
    };



}