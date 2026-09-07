cmake_minimum_required(VERSION 3.20)
project(course-rag CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# MSVC 默认执行字符集跟随系统 codepage（本机 cp936）：中文窄字面量会变成 GBK
# 运行时字节 → 发给 LLM 的 prompt JSON 与降级文案全成乱码。/utf-8 同时固定源码
# 与执行字符集（含 /validate-charset，误存 GBK 的文件直接编译报错，fail fast）
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>")
enable_testing()
add_subdirectory(core)
add_subdirectory(rag)
add_subdirectory(server)
