#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

struct Point {
  double x = 0, y = 0;
};

inline double euclid(const Point& a, const Point& b) {
  double dx = a.x - b.x, dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

// Instancia TSP euclidiana. Soporta matriz densa (n pequeno)
// y distancia bajo demanda (n grande).
struct TSPInstance {
  int n = 0;
  std::vector<Point> pts;
  std::vector<float> mat;          // n*n, solo si has_matrix
  bool has_matrix = false;
  std::vector<std::vector<int>> cand;  // k-NN por ciudad
  int cand_k = 0;

  void generate_random(int n_, uint32_t seed, double coord_max = 1000.0) {
    n = n_;
    pts.resize(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, coord_max);
    for (int i = 0; i < n; ++i) { pts[i].x = u(rng); pts[i].y = u(rng); }
    has_matrix = false;
    mat.clear();
    cand.clear();
  }

  inline float dist(int i, int j) const {
    if (i == j) return 0.0f;
    if (has_matrix) return mat[(size_t)i * n + j];
    return (float)euclid(pts[i], pts[j]);
  }

  double tour_length(const std::vector<int>& tour) const {
    double L = 0;
    for (int i = 0; i < n; ++i)
      L += dist(tour[i], tour[(i + 1) % n]);
    return L;
  }

  // Memoria estimada matriz densa en MB (float).
  static double matrix_mb(int n_) {
    return (double)(size_t)n_ * (size_t)n_ * sizeof(float) / (1024.0 * 1024.0);
  }

  void build_matrix() {
    mat.resize((size_t)n * n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j)
        mat[(size_t)i * n + j] = (float)euclid(pts[i], pts[j]);
    }
    has_matrix = true;
  }

  // k-NN exacto O(n^2 log k). Solo usar con n <= ~5000.
  void build_candidates(int k) {
    cand_k = std::min(k, n - 1);
    cand.assign(n, {});
    if (!has_matrix) build_matrix();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      std::vector<int> idx(n - 1);
      int p = 0;
      for (int j = 0; j < n; ++j) if (j != i) idx[p++] = j;
      int kk = cand_k;
      std::nth_element(idx.begin(), idx.begin() + kk, idx.end(),
        [&](int a, int b) { return mat[(size_t)i * n + a] < mat[(size_t)i * n + b]; });
      idx.resize(kk);
      std::sort(idx.begin(), idx.end(),
        [&](int a, int b) { return mat[(size_t)i * n + a] < mat[(size_t)i * n + b]; });
      cand[i] = std::move(idx);
    }
  }
};
