#pragma once
#include <string>
#include <cstdint>

namespace core
{
    // 1. 一条检索结果：内部id（对应向量文件的第几行）+ 相似度分数
    struct SearchItem
    {
        uint32_t id;
        float similarity;
    };
    // 2. 元数据过滤条件：三个字段，空字符串表示“不限”
    struct MetaFilter
    {
        std::string course;
        std::string semester;
        std::string type_;
    };
    // 3.每条资料的元数据：课程、学期、类型、标题
    struct ChunkMeta
    {
        std::string course;
        std::string semester;
        std::string type_;
        std::string title;
    };
// 4. 一个内联函数 match(m, f)：判断元数据 m 是否满足过滤条件 f
//    规则：f 的每个非空字段都必须和 m 的对应字段相等；f 为空 = 全部通过
    inline bool match(const ChunkMeta& m,const MetaFilter& f)
    { 
        if(!f.course.empty() && m.course != f.course)
        {
            return false;
        }
        if(!f.semester.empty() && m.semester != f.semester)
        {
            return false;
        }
        if(!f.type_.empty() && m.type_ != f.type_)
        {
            return false;
        }
        return true;
    }
}
