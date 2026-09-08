# Task 13 基准测试：三种索引（brute / ivf / hnsw）统一数据、统一查询的对比。
# CI 的 Linux job 只编译不运行（内存实测是 Windows 专属，非 Windows 输出 N/A）
add_executable(course-rag-bench bench_main.cpp)
target_link_libraries(course-rag-bench PRIVATE core)
if(WIN32)
    # GetProcessMemoryInfo 位于 psapi（MSVC: psapi.lib；MinGW: libpsapi.a）
    target_link_libraries(course-rag-bench PRIVATE psapi)
endif()
