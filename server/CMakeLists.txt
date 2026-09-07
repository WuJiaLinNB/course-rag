# HTTP 服务（Task 12）：httplib.h / json.hpp 单头在 third_party（与 rag 共用同目录）；
# Windows 下 httplib 需要 ws2_32
add_executable(course-rag-server src/main.cpp)
target_include_directories(course-rag-server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(course-rag-server PRIVATE core rag ws2_32)
