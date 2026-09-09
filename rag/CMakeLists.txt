add_library(rag STATIC src/pipeline.cpp)
target_include_directories(rag PUBLIC include)
# httplib.h / json.hpp 落在 server/third_party（Task 12 复用同目录）；third_party 是 rag 的私有依赖，不外泄
target_include_directories(rag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../server/third_party)
target_link_libraries(rag PUBLIC core)

# 出站 TLS（根 CMakeLists 探测，Windows + vendored OpenSSL 时非空）
if(COURSE_RAG_TLS_DEFINES)
    target_compile_definitions(rag PRIVATE ${COURSE_RAG_TLS_DEFINES})
    target_include_directories(rag PRIVATE ${COURSE_RAG_TLS_INCLUDE})
    target_link_libraries(rag PRIVATE ${COURSE_RAG_TLS_LIBS})
    # rag_tests 也链 rag（import lib 经 LINK_ONLY 传递），DLL 就位才能跑测试
    if(COURSE_RAG_TLS_DLLS)
        add_custom_command(TARGET rag POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${COURSE_RAG_TLS_DLLS}
                    $<TARGET_FILE_DIR:rag>)
    endif()
endif()

add_executable(rag_tests tests/test_pipeline.cpp)
target_link_libraries(rag_tests PRIVATE rag gtest_main)
include(GoogleTest)
gtest_discover_tests(rag_tests)
