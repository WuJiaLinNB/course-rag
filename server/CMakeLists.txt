# HTTP 服务（Task 12）：httplib.h / json.hpp 单头在 third_party（与 rag 共用同目录）；
# Windows 下 httplib 需要 ws2_32。
# 出站 TLS（Task 16 前置）：根 CMakeLists 探测到 vendored OpenSSL 时定义
# CPPHTTPLIB_OPENSSL_SUPPORT，httplib 客户端可调 https 的 embedding/LLM API；
# 入站仍是纯 HTTP（TLS 由公网隧道层终结，见根 CMakeLists 注释）。
# main.cpp 是 Windows-only：SetConsoleCtrlHandler / MoveFileExA 无 POSIX 等价实现，
# 部署目标固定为 Windows 本机 + frp 隧道，CI（ubuntu）只验证 core/rag/bench，不构建 server。
if(WIN32)
    add_executable(course-rag-server src/main.cpp)
    target_include_directories(course-rag-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(course-rag-server PRIVATE core rag ws2_32)
    if(COURSE_RAG_TLS_DEFINES)
        target_compile_definitions(course-rag-server PRIVATE ${COURSE_RAG_TLS_DEFINES})
        target_include_directories(course-rag-server PRIVATE ${COURSE_RAG_TLS_INCLUDE})
        target_link_libraries(course-rag-server PRIVATE ${COURSE_RAG_TLS_LIBS})
        if(COURSE_RAG_TLS_DLLS)
            add_custom_command(TARGET course-rag-server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${COURSE_RAG_TLS_DLLS}
                        $<TARGET_FILE_DIR:course-rag-server>)
        endif()
    endif()
endif()
