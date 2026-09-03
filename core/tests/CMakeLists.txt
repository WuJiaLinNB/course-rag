add_executable(core_tests test_smoke.cpp)
target_link_libraries(core_tests PRIVATE core gtest_main)
include(GoogleTest)
gtest_discover_tests(core_tests)
