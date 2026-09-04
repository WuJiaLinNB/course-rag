#include "core/distance.hpp"

#include <cmath>
#include <stdexcept>

namespace core {

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot: vectors must have the same size");
    }
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("cosine: vectors must have the same size");
    }
    const float dot_ab = dot(a, b);
    const float norm_a = std::sqrt(dot(a, a));
    const float norm_b = std::sqrt(dot(b, b));
    if (norm_a == 0.0f || norm_b == 0.0f) {
        throw std::invalid_argument("cosine: zero vector has no direction");
    }
    return dot_ab / (norm_a * norm_b);
}

}  // namespace core
