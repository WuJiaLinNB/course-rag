add_library(core STATIC src/distance.cpp src/vector_store.cpp src/brute_index.cpp src/kmeans.cpp src/ivf_index.cpp src/hnsw_index.cpp src/engine.cpp)
target_include_directories(core PUBLIC include)
# engine.cpp 用 std::thread（后台重建线程）：MSVC 无需显式库，但 Linux CI（gcc）必须 -pthread，
# 否则链接报 undefined reference。PUBLIC 传递，下游（rag/bench/tests）不再各自找。
find_package(Threads REQUIRED)
target_link_libraries(core PUBLIC Threads::Threads)

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(tests)
