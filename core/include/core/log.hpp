#pragma once
#include <cstdio>

// v1 可观测性底线：启动/退出、重建开始/完成/热切换、降级发生、401 计数、致命错误
// 第一个参数必须是字面量格式串；动态字符串一律 LOG_INFO("%s", str)，禁止 LOG_INFO(str)
#define LOG_IMPL(level, ...) do { \
    std::fprintf(stderr, "[%s] ", level); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fprintf(stderr, "\n"); \
} while (0)

#define LOG_INFO(...)  LOG_IMPL("INFO ", __VA_ARGS__)
#define LOG_WARN(...)  LOG_IMPL("WARN ", __VA_ARGS__)
#define LOG_ERROR(...) LOG_IMPL("ERROR", __VA_ARGS__)
