cmake_minimum_required(VERSION 3.20)
project(course-rag CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# MSVC 默认执行字符集跟随系统 codepage（本机 cp936）：中文窄字面量会变成 GBK
# 运行时字节 → 发给 LLM 的 prompt JSON 与降级文案全成乱码。/utf-8 同时固定源码
# 与执行字符集（含 /validate-charset，误存 GBK 的文件直接编译报错，fail fast）
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/utf-8>")
enable_testing()

# ---- 出站 TLS（调 https 的 embedding/LLM API，server 在线查询与 rag call_llm）----
# Windows 用 vendored OpenSSL 3 动态库（third_party/openssl，vcpkg x64-windows 产物，
# 来源与版本见该目录 LICENSE/提交记录）；动态库 Debug/Release 通用，DLL 由 rag/server
# 的 post-build 拷到各自输出目录。vendor 进仓库保持"仓库拷贝为权威来源"（不再依赖
# 本机 vcpkg 状态），构建工作流零变化。
# 非 Windows（CI ubuntu）不启用：CI 场景（core/rag/bench 测试）无 https 外呼，
# 不为测试路径新增 libssl 依赖。
# 入站 TLS 仍由公网隧道层终结（设计文档第 14 节）：server 本体继续只说 HTTP、
# 只监听 127.0.0.1，与 Task 16 的部署设计一致。
set(COURSE_RAG_TLS_DEFINES "")
set(COURSE_RAG_TLS_INCLUDE "")
set(COURSE_RAG_TLS_LIBS "")
set(COURSE_RAG_TLS_DLLS "")
if(WIN32 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl/include/openssl/ssl.h)
    set(COURSE_RAG_TLS_DEFINES CPPHTTPLIB_OPENSSL_SUPPORT)
    set(COURSE_RAG_TLS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl/include)
    set(COURSE_RAG_TLS_LIBS
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl/lib/libssl.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl/lib/libcrypto.lib)
    file(GLOB COURSE_RAG_TLS_DLLS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openssl/bin/*.dll)
    message(STATUS "course-rag: outbound TLS enabled (vendored OpenSSL in third_party/openssl)")
endif()

# TLS 变量必须先于 add_subdirectory 赋值：CMake 处理子目录是立即的，
# 若块在后，server/rag 读到的 COURSE_RAG_TLS_* 全是空串（TLS 静默失效）。
add_subdirectory(core)
add_subdirectory(rag)
add_subdirectory(server)
add_subdirectory(bench)
