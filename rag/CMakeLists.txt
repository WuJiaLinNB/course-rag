add_library(rag STATIC src/pipeline.cpp)
target_include_directories(rag PUBLIC include)
# httplib.h / json.hpp 落在 server/third_party（Task 12 复用同目录）；third_party 是 rag 的私有依赖，不外泄
target_include_directories(rag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../server/third_party)
target_link_libraries(rag PUBLIC core)

add_executable(rag_tests tests/test_pipeline.cpp)
target_link_libraries(rag_tests PRIVATE rag gtest_main)
include(GoogleTest)
gtest_discover_tests(rag_tests)
