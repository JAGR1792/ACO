#pragma once
#include "tsp.h"
#include <random>
#include <vector>
#include <limits>
#include <cmath>
#include <omp.h>

struct ACOParams {
  int ants = 20;
  int iters = 100;
  double alpha = 1.0;
  double beta = 2.5;
  double rho = 0.1;        // evaporacion
  int cand_k = 20;         // 0 = sin lista de candidatos
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
      // j recorre aristas no adyacentes; truco con posiciones para reversa O(1) amortizado
      for (int j = i + 2; j < jmax; ++j) {
        int jj = j % n;
        if (i == 0 && jj == n - 1) continue;
        int c = tour[jj], d = tour[(jj + 1) % n];
        double before = inst.dist(a, b) + inst.dist(c, d);
        double after = inst.dist(a, c) + inst.dist(b, d);
        if (after + 1e-9 < before) {
          // reversar segmento (i+1 .. jj)
          int lo = i + 1, hi = jj;
          if (lo < hi) {
            // maneja wrap cuando jj < i (no ocurre aqui porque j>i lineal, pero con modulo si)
            // como jmax<=n e i<n, el segmento es lineal salvo cruce del 0; lo tratamos lineal
            // Reconstruccion simple por indice lineal sobre tour duplicado:
          }
          // Implementacion robusta: reversa circular entre i+1 y jj
          // Convertimos a bucle con posiciones circulares.
          int l = (i + 1) % n, r = jj;
          // numero de swaps = dist circular(l,r)/2
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

// MMAS simplificado: solo la mejor hormiga global deposita, con limites tau.
class ACOSolver {
 public:
  TSPInstance& inst;
  ACOParams p;

  ACOSolver(TSPInstance& inst_, ACOParams p_) : inst(inst_), p(p_) {}

  ACOResult solve(FILE* trace = nullptr) {
    int n = inst.n;
    ACOResult best;
    best.tour.resize(n);
    std::iota(best.tour.begin(), best.tour.end(), 0);

    // Feromona densa: solo viable si n moderado. El llamante decide.
    std::vector<float> tau((size_t)n * n, 1.0f);
    auto TAU = [&](int i, int j) -> float& { return tau[(size_t)i * n + j]; };

    // Inicializa best con NN greedy desde 0 para tener cota.
    {
      std::vector<char> vis(n, 0);
      std::vector<int> t; t.reserve(n);
      int cur = 0; t.push_back(cur); vis[cur] = 1;
      for (int s = 1; s < n; ++s) {
        int nxt = -1; float bd = std::numeric_limits<float>::max();
        // si hay candidatos, buscar ahi primero
        if (!inst.cand.empty()) {
          for (int c : inst.cand[cur]) if (!vis[c]) {
            float d = inst.dist(cur, c);
            if (d < bd) { bd = d; nxt = c; }
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
    std::fill(tau.begin(), tau.end(), (float)tau_max);

    std::vector<std::vector<int>> ant_tours(p.ants, std::vector<int>(n));
    std::vector<double> ant_len(p.ants);

    std::vector<uint32_t> seeds(p.ants);
    for (int a = 0; a < p.ants; ++a) seeds[a] = p.seed + 7919u * a + 17u;

    std::vector<double> pow_tau_cache;
    (void)pow_tau_cache;

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
        std::vector<double> probs;
        std::vector<int> cands;
        probs.reserve(64); cands.reserve(64);
        for (int step = 1; step < n; ++step) {
          cands.clear(); probs.clear();
          double sum = 0;
          // 1) intenta lista de candidatos no visitados
          if (!inst.cand.empty()) {
            for (int c : inst.cand[cur]) if (!vis[c]) { cands.push_back(c); }
          }
          // 2) si lista vacia o todos visitados: usa todos los no visitados
          //    (para n<=2000 es OK; para n grande este solver no se usa)
          if (cands.empty()) {
            for (int j = 0; j < n; ++j) if (!vis[j]) cands.push_back(j);
          }
          for (int c : cands) {
            double t = std::pow((double)TAU(cur, c), p.alpha);
            double e = std::pow(1.0 / ((double)inst.dist(cur, c) + 1e-9), p.beta);
            double v = t * e;
            probs.push_back(v); sum += v;
          }
          int nxt = cands.back();
          double r = uni(rng) * sum;
          double acc = 0;
          for (size_t k = 0; k < cands.size(); ++k) {
            acc += probs[k];
            if (acc >= r) { nxt = cands[k]; break; }
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

      // evaporacion
      #pragma omp parallel for schedule(static)
      for (size_t k = 0; k < tau.size(); ++k) tau[k] *= (float)(1.0 - p.rho);

      // deposito: mejor global (MMAS)
      double dep = 1.0 / (best.length + 1e-300);
      for (int i = 0; i < n; ++i) {
        int u = best.tour[i], v = best.tour[(i + 1) % n];
        TAU(u, v) += (float)dep;
        TAU(v, u) += (float)dep;
      }
      // clamp
      #pragma omp parallel for schedule(static)
      for (size_t k = 0; k < tau.size(); ++k) {
        if (tau[k] > tau_max) tau[k] = (float)tau_max;
        else if (tau[k] < tau_min) tau[k] = (float)tau_min;
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
