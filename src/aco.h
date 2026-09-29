#pragma once
#include "tsp.h"
#include <random>
#include <vector>
#include <limits>
#include <cmath>
#include <cstdio>
#include <omp.h>

struct ACOParams {
  int ants = 20;
  int iters = 100;
  double alpha = 1.0;
  double beta = 2.5;
  double rho = 0.1;        // evaporacion
  int cand_k = 20;         // 0 = sin lista de candidatos (solo heuristica)
  uint32_t seed = 42;
  bool use_2opt = true;
  int q0_decimals = -1;    // reservado
};

struct ACOResult {
  std::vector<int> tour;
  double length = 1e300;
};

// 2-opt first-improvement, con ventana limitada si window>0.
inline bool two_opt(TSPInstance& inst, std::vector<int>& tour, int max_passes = 2, int window = 0) {
  int n = inst.n;
  if (n < 4) return false;
  bool any = false;
  std::vector<int> pos(n);
  for (int pass = 0; pass < max_passes; ++pass) {
    bool improved = false;
    for (int i = 0; i < n; ++i) pos[tour[i]] = i;
    for (int i = 0; i < n - 2; ++i) {
      int a = tour[i], b = tour[(i + 1) % n];
      int jmax = (window > 0) ? std::min(n, i + 2 + window) : n;
      for (int j = i + 2; j < jmax; ++j) {
        int jj = j % n;
        if (i == 0 && jj == n - 1) continue;
        int c = tour[jj], d = tour[(jj + 1) % n];
        double before = inst.dist(a, b) + inst.dist(c, d);
        double after = inst.dist(a, c) + inst.dist(b, d);
        if (after + 1e-9 < before) {
          int l = (i + 1) % n, r = jj;
          int len = (r - l + n) % n + 1;
          for (int s = 0; s < len / 2; ++s) {
            std::swap(tour[l], tour[r]);
            l = (l + 1) % n;
            r = (r - 1 + n) % n;
          }
          for (int k2 = 0; k2 < n; ++k2) pos[tour[k2]] = k2;
          a = tour[i]; b = tour[(i + 1) % n];
          improved = true; any = true;
        }
      }
    }
    if (!improved) break;
  }
  return any;
}

// MMAS simplificado disperso: feromona n*k (no n*n), eta precalculada.
// Memoria O(n*k). Tiempo O(t*m*n*k) paralelo en hormigas.
class ACOSolver {
 public:
  TSPInstance& inst;
  ACOParams p;

  ACOSolver(TSPInstance& inst_, ACOParams p_) : inst(inst_), p(p_) {}

  ACOResult solve(FILE* trace = nullptr) {
    int n = inst.n;
    int K = inst.cand_k; // 0 => sin candidatos
    ACOResult best;
    best.tour.resize(n);
    std::iota(best.tour.begin(), best.tour.end(), 0);

    // Inicializa best con NN greedy desde 0 para tener cota.
    {
      std::vector<char> vis(n, 0);
      std::vector<int> t; t.reserve(n);
      int cur = 0; t.push_back(cur); vis[cur] = 1;
      for (int s = 1; s < n; ++s) {
        int nxt = -1; float bd = std::numeric_limits<float>::max();
        if (inst.has_candidates()) {
          const int* row = &inst.knn_idx[(size_t)cur * K];
          const float* rd = &inst.knn_dist[(size_t)cur * K];
          for (int tt = 0; tt < K; ++tt) {
            int c = row[tt];
            if (!vis[c] && rd[tt] < bd) { bd = rd[tt]; nxt = c; }
          }
        }
        if (nxt < 0) {
          for (int j = 0; j < n; ++j) if (!vis[j]) {
            float d = inst.dist(cur, j);
            if (d < bd) { bd = d; nxt = j; }
          }
        }
        t.push_back(nxt); vis[nxt] = 1; cur = nxt;
      }
      best.tour = t;
      best.length = inst.tour_length(t);
    }

    double tau_max = 1.0 / (p.rho * best.length + 1e-300);
    double tau_min = tau_max / (2.0 * n);

    // Feromona dispersa + heuristica precalculada (solo si hay candidatos).
    std::vector<float> tau, eta;
    if (K > 0) {
      tau.assign((size_t)n * K, (float)tau_max);
      eta.assign((size_t)n * K, 1.0f);
      #pragma omp parallel for schedule(static)
      for (int i = 0; i < n; ++i) {
        for (int tt = 0; tt < K; ++tt) {
          float d = inst.knn_dist[(size_t)i * K + tt];
          eta[(size_t)i * K + tt] = std::pow(1.0f / (d + 1e-9f), (float)p.beta);
        }
      }
    }

    std::vector<std::vector<int>> ant_tours(p.ants, std::vector<int>(n));
    std::vector<double> ant_len(p.ants);

    std::vector<uint32_t> seeds(p.ants);
    for (int a = 0; a < p.ants; ++a) seeds[a] = p.seed + 7919u * a + 17u;

    auto deposit_edge = [&](std::vector<float>& T, int u, int v, float q) {
      if (K <= 0) return;
      const int* row = &inst.knn_idx[(size_t)u * K];
      for (int tt = 0; tt < K; ++tt)
        if (row[tt] == v) { T[(size_t)u * K + tt] += q; return; }
      // arista fuera de candidatos: se ignora (ahorro de memoria)
    };

    for (int it = 0; it < p.iters; ++it) {
      #pragma omp parallel for schedule(dynamic, 1)
      for (int a = 0; a < p.ants; ++a) {
        std::mt19937 rng(seeds[a] + (uint32_t)it * 104729u);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<char> vis(n, 0);
        std::vector<int> tour; tour.reserve(n);
        int start = (int)(rng() % (uint32_t)n);
        tour.push_back(start); vis[start] = 1;
        int cur = start;
        // buffers reutilizados por paso (sin alloc dentro del loop caliente)
        std::vector<int> cands; cands.reserve(64);
        std::vector<float> probs; probs.reserve(64);
        for (int step = 1; step < n; ++step) {
          cands.clear(); probs.clear();
          double sum = 0;
          if (K > 0) {
            const int* row = &inst.knn_idx[(size_t)cur * K];
            for (int tt = 0; tt < K; ++tt) {
              int c = row[tt];
              if (!vis[c]) {
                float tauv = tau[(size_t)cur * K + tt];
                float tp = (p.alpha == 1.0) ? tauv : std::pow(tauv, (float)p.alpha);
                float v = tp * eta[(size_t)cur * K + tt];
                cands.push_back(c);
                probs.push_back(v);
                sum += v;
              }
            }
          }
          int nxt = -1;
          if (!cands.empty()) {
            double r = uni(rng) * sum;
            double acc = 0;
            nxt = cands.back();
            if (sum > 0 && std::isfinite(sum)) {
              for (size_t k = 0; k < cands.size(); ++k) {
                acc += probs[k];
                if (acc >= r) { nxt = cands[k]; break; }
              }
            } else {
              nxt = cands[rng() % cands.size()];
            }
          } else {
            // Fallback sin alloc: vecino mas cercano entre no visitados.
            // O(n) por paso, pero solo ocurre al final del tour.
            float bd = std::numeric_limits<float>::max();
            for (int j = 0; j < n; ++j) if (!vis[j]) {
              float d = inst.dist(cur, j);
              if (d < bd) { bd = d; nxt = j; }
            }
          }
          tour.push_back(nxt); vis[nxt] = 1; cur = nxt;
        }
        ant_tours[a] = tour;
        ant_len[a] = inst.tour_length(tour);
      }

      // mejor de la iteracion
      int bi = 0;
      for (int a = 1; a < p.ants; ++a) if (ant_len[a] < ant_len[bi]) bi = a;

      // 2-opt opcional al mejor de iteracion (barato y mejora mucho)
      if (p.use_2opt) {
        std::vector<int> t = ant_tours[bi];
        two_opt(inst, t, 1, (n > 500 ? 40 : 0));
        double L = inst.tour_length(t);
        if (L < ant_len[bi]) { ant_tours[bi] = t; ant_len[bi] = L; }
      }

      if (ant_len[bi] < best.length) {
        best.length = ant_len[bi];
        best.tour = ant_tours[bi];
        tau_max = 1.0 / (p.rho * best.length + 1e-300);
        tau_min = tau_max / (2.0 * n);
      }

      if (K > 0) {
        float decay = (float)(1.0 - p.rho);
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < tau.size(); ++k) tau[k] *= decay;

        // deposito: mejor global (MMAS)
        float dep = (float)(1.0 / (best.length + 1e-300));
        for (int i = 0; i < n; ++i) {
          int u = best.tour[i], v = best.tour[(i + 1) % n];
          deposit_edge(tau, u, v, dep);
          deposit_edge(tau, v, u, dep);
        }
        float tmax = (float)tau_max, tmin = (float)tau_min;
        #pragma omp parallel for schedule(static)
        for (size_t k = 0; k < tau.size(); ++k) {
          if (tau[k] > tmax) tau[k] = tmax;
          else if (tau[k] < tmin) tau[k] = tmin;
        }
      }

      if (trace) fprintf(trace, "%d,%.2f\n", it + 1, best.length);
    }

    if (p.use_2opt) {
      two_opt(inst, best.tour, 2, (n > 500 ? 40 : 0));
      best.length = inst.tour_length(best.tour);
    }
    return best;
  }
};
