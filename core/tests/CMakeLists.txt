add_executable(core_tests
  test_smoke.cpp
  test_distance.cpp
  test_vector_store.cpp
  test_brute.cpp
  test_kmeans.cpp
  test_ivf.cpp
  test_hnsw.cpp
  test_concurrency.cpp
  test_consistency.cpp)
target_link_libraries(core_tests PRIVATE core gtest_main)
include(GoogleTest)
gtest_discover_tests(core_tests)
