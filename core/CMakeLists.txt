add_library(core STATIC src/distance.cpp src/vector_store.cpp src/brute_index.cpp src/kmeans.cpp src/ivf_index.cpp)
target_include_directories(core PUBLIC include)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)
