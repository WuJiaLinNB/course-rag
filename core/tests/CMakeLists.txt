add_executable(core_tests
  test_smoke.cpp
  test_distance.cpp)
target_link_libraries(core_tests PRIVATE core gtest_main)
include(GoogleTest)
gtest_discover_tests(core_tests)
