#include <gtest/gtest.h>
#include <core/distance.hpp>
#include <stdexcept>
#include <vector>

using namespace core;

TEST(Distance, DotProduct) {
    std::vector<float> a{1, 2, 3};
    std::vector<float> b{4, 5, 6};
    EXPECT_FLOAT_EQ(dot(a, b), 32.0f);
}

TEST(Distance, DotSizeMismatch) {
    std::vector<float> a{1, 2, 3};
    std::vector<float> b{1, 2};
    EXPECT_THROW(dot(a, b), std::invalid_argument);
}

TEST(Distance, CosineIdentical) {
    EXPECT_NEAR(cosine({1, 0, 0}, {1, 0, 0}), 1.0f, 1e-6f);
}

TEST(Distance, CosineOrthogonal) {  // 正交
    EXPECT_NEAR(cosine({1, 0, 0}, {0, 1, 0}), 0.0f, 1e-6f);
}

TEST(Distance, CosineOpposite) {
    EXPECT_NEAR(cosine({1, 0, 0}, {-1, 0, 0}), -1.0f, 1e-6f);
}

TEST(Distance, CosineZeroVectorThrows) {
    EXPECT_THROW(cosine({0, 0, 0}, {1, 0, 0}), std::invalid_argument);
}
