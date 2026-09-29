#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

struct Point {
  float x = 0, y = 0;
};

inline float euclid(const Point& a, const Point& b) {
  float dx = a.x - b.x, dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

// Instancia TSP euclidiana, baja memoria:
// - puntos en float (O(n))
// - k-NN plano n*k (O(n*k)), sin matriz n*n
// - distancias fuera de candidatos: al vuelo (O(1) extra)
struct TSPInstance {
  int n = 0;
  std::vector<Point> pts;
  std::vector<int> knn_idx;    // n*cand_k, plano
  std::vector<float> knn_dist; // n*cand_k, plano
  int cand_k = 0;

  void generate_random(int n_, uint32_t seed, double coord_max = 1000.0) {
    n = n_;
    pts.resize(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, coord_max);
    for (int i = 0; i < n; ++i) { pts[i].x = (float)u(rng); pts[i].y = (float)u(rng); }
    cand_k = 0;
    knn_idx.clear();
    knn_dist.clear();
  }

  inline float dist(int i, int j) const {
    if (i == j) return 0.0f;
    return euclid(pts[i], pts[j]);
  }

  double tour_length(const std::vector<int>& tour) const {
    double L = 0;
    for (int i = 0; i < n; ++i)
      L += (double)dist(tour[i], tour[(i + 1) % n]);
    return L;
  }

  // Memoria estimada en MB.
  static double dense_mb_theoretical(int n_) {
    return (double)(size_t)n_ * (size_t)n_ * sizeof(float) / (1024.0 * 1024.0);
  }
  static double sparse_mb(int n_, int k_) {
    // idx(int) + dist(float) + tau(float) + eta(float) = 4 bytes * 4 * n*k
    return (double)(size_t)n_ * (size_t)k_ * 4.0 * sizeof(float) / (1024.0 * 1024.0);
  }
  // Compat: antes se llamaba matrix_mb().
  static double matrix_mb(int n_) { return dense_mb_theoretical(n_); }

  bool has_candidates() const { return cand_k > 0 && !knn_idx.empty(); }

  // k-NN exacto con buffer O(n) por hilo, sin materializar n*n.
  // Solo usar con n <= ~5000 en modo full; en hier se llama por celda (~600).
  void build_candidates(int k) {
    cand_k = std::min(k, n - 1);
    if (cand_k <= 0) { knn_idx.clear(); knn_dist.clear(); cand_k = 0; return; }
    int K = cand_k;
    knn_idx.assign((size_t)n * K, 0);
    knn_dist.assign((size_t)n * K, 0.0f);
    #pragma omp parallel
    {
      std::vector<float> d2(n);
      std::vector<int> id(n);
      #pragma omp for schedule(static)
      for (int i = 0; i < n; ++i) {
        float xi = pts[i].x, yi = pts[i].y;
        for (int j = 0; j < n; ++j) {
          float dx = xi - pts[j].x, dy = yi - pts[j].y;
          d2[j] = dx * dx + dy * dy;
        }
        d2[i] = 1e30f;
        std::iota(id.begin(), id.end(), 0);
        std::nth_element(id.begin(), id.begin() + K, id.end(),
          [&](int a, int b) { return d2[a] < d2[b]; });
        // ordena solo el top-k para ruleta estable
        std::sort(id.begin(), id.begin() + K,
          [&](int a, int b) { return d2[a] < d2[b]; });
        for (int t = 0; t < K; ++t) {
          knn_idx[(size_t)i * K + t] = id[t];
          knn_dist[(size_t)i * K + t] = std::sqrt(d2[id[t]]);
        }
      }
    }
  }
};
