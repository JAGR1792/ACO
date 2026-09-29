#include "tsp.h"
#include "aco.h"
#include "metrics.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

struct Args {
  int n = 20;
  int ants = 20;
  int iters = 100;
  int cand_k = 20;
  uint32_t seed = 42;
  bool use_2opt = true;
  std::string mode = "auto";  // auto|full|hier|nn
  int target_cluster = 600;
  std::string out;            // csv append
  std::string trace_out;      // csv de convergencia (iter,best) solo modo full
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { fprintf(stderr, "falta valor para %s\n", name); exit(2); }
      return argv[++i];
    };
    if (s == "--n") a.n = std::stoi(need("n"));
    else if (s == "--ants") a.ants = std::stoi(need("ants"));
    else if (s == "--iters") a.iters = std::stoi(need("iters"));
    else if (s == "--cand") a.cand_k = std::stoi(need("cand"));
    else if (s == "--seed") a.seed = (uint32_t)std::stoul(need("seed"));
    else if (s == "--no-2opt") a.use_2opt = false;
    else if (s == "--mode") a.mode = need("mode");
    else if (s == "--target-cluster") a.target_cluster = std::stoi(need("target"));
    else if (s == "--out") a.out = need("out");
    else if (s == "--trace-out") a.trace_out = need("trace-out");
    else if (s == "--help" || s == "-h") {
      printf("Uso: ./aco --n 20 --ants 20 --iters 100 --cand 20 --seed 42 --mode auto|full|hier|nn [--no-2opt] [--target-cluster 600] [--out results.csv] [--trace-out trace.csv]\n");
      exit(0);
    } else { fprintf(stderr, "arg desconocido: %s\n", s.c_str()); exit(2); }
  }
  return a;
}

// NN greedy global (baseline, O(n^2) sin matriz: usa puntos directos).
static std::vector<int> nn_tour(const TSPInstance& inst, int start = 0) {
  int n = inst.n;
  std::vector<char> vis(n, 0);
  std::vector<int> tour; tour.reserve(n);
  int cur = start; tour.push_back(cur); vis[cur] = 1;
  for (int s = 1; s < n; ++s) {
    int best = -1; double bd = 1e300;
    for (int j = 0; j < n; ++j) if (!vis[j]) {
      double d = euclid(inst.pts[cur], inst.pts[j]);
      if (d < bd) { bd = d; best = j; }
    }
    tour.push_back(best); vis[best] = 1; cur = best;
  }
  return tour;
}

struct FullResult { std::vector<int> tour; double len; };

// ACO disperso completo. Memoria O(n*k), sin matriz n*n.
static FullResult run_full(TSPInstance& inst, const Args& a) {
  double dense = TSPInstance::dense_mb_theoretical(inst.n);
  // cota de seguridad sobre la densa teorica (aunque ya no se materializa)
  if (dense > 400.0 && a.cand_k <= 0) {
    fprintf(stderr, "[ERROR] n=%d sin candidatos pediria densa %.1f MB, usa --mode hier o --cand 8..25\n", inst.n, dense);
    exit(3);
  }
  if (a.cand_k > 0) inst.build_candidates(a.cand_k);
  ACOParams p;
  p.ants = a.ants; p.iters = a.iters;
  p.cand_k = a.cand_k; p.seed = a.seed; p.use_2opt = a.use_2opt;
  ACOSolver solver(inst, p);
  FILE* tr = nullptr;
  if (!a.trace_out.empty()) {
    tr = fopen(a.trace_out.c_str(), "w");
    if (tr) fprintf(tr, "iter,best\n");
  }
  ACOResult r = solver.solve(tr);
  if (tr) fclose(tr);
  return {r.tour, r.length};
}

// ACO jerarquico para n grande: grid -> ACO por celda -> orden por centroides.
static FullResult run_hier(TSPInstance& inst, const Args& a) {
  int n = inst.n;
  int G = (int)std::ceil(std::sqrt((double)n / (double)a.target_cluster));
  if (G < 1) G = 1;
  if (G > 100) G = 100;  // evita miles de celdas diminutas
  double minx = 1e30, miny = 1e30, maxx = -1e30, maxy = -1e30;
  for (auto& pt : inst.pts) {
    minx = std::min(minx, (double)pt.x); maxx = std::max(maxx, (double)pt.x);
    miny = std::min(miny, (double)pt.y); maxy = std::max(maxy, (double)pt.y);
  }
  double wx = (maxx - minx) / G + 1e-9, wy = (maxy - miny) / G + 1e-9;
  std::map<std::pair<int,int>, std::vector<int>> cells;
  for (int i = 0; i < n; ++i) {
    int gx = std::min(G - 1, (int)((inst.pts[i].x - minx) / wx));
    int gy = std::min(G - 1, (int)((inst.pts[i].y - miny) / wy));
    cells[{gx, gy}].push_back(i);
  }
  // vector de clusters
  std::vector<std::vector<int>> clusters;
  for (auto& kv : cells) clusters.push_back(std::move(kv.second));
  int C = clusters.size();
  fprintf(stderr, "[hier] n=%d G=%d clusters=%d\n", n, G, C);

  // 1) ACO intra-cluster (cada cluster es pequeno: ~target_cluster)
  // Humillacion total: k=10 concentra mejor + mas esfuerzo (12x35).
  // Sigue en segundos vs minutos de Camilo (180x mas rapido).
  std::vector<std::vector<int>> cluster_tours(C);
  #pragma omp parallel for schedule(dynamic, 1)
  for (int c = 0; c < C; ++c) {
    TSPInstance sub;
    sub.n = (int)clusters[c].size();
    sub.pts.resize(sub.n);
    for (int i = 0; i < sub.n; ++i) sub.pts[i] = inst.pts[clusters[c][i]];
    int ants = std::min(12, std::max(6, sub.n / 15));
    int iters = (sub.n < 100) ? 20 : 35;
    int ck = std::min(10, sub.n - 1);
    if (sub.n >= 3) {
      if (ck > 0) sub.build_candidates(ck);
      ACOParams p; p.ants = ants; p.iters = iters; p.cand_k = ck;
      p.seed = a.seed + (uint32_t)c * 1009u; p.use_2opt = true;
      ACOSolver s(sub, p);
      ACOResult r = s.solve();
      // mapea a indices globales
      std::vector<int> gt(r.tour.size());
      for (size_t i = 0; i < r.tour.size(); ++i) gt[i] = clusters[c][r.tour[i]];
      cluster_tours[c] = std::move(gt);
    } else {
      cluster_tours[c] = clusters[c];  // 1-2 puntos
    }
  }

  // 2) ordena clusters por centroide con NN (C es pequeno, ~400)
  std::vector<Point> cent(C);
  for (int c = 0; c < C; ++c) {
    double sx = 0, sy = 0;
    for (int g : clusters[c]) { sx += inst.pts[g].x; sy += inst.pts[g].y; }
    cent[c].x = (float)(sx / clusters[c].size()); cent[c].y = (float)(sy / clusters[c].size());
  }
  std::vector<char> vis(C, 0);
  std::vector<int> order; order.reserve(C);
  int cur = 0; order.push_back(cur); vis[cur] = 1;
  for (int s = 1; s < C; ++s) {
    int best = -1; double bd = 1e300;
    for (int j = 0; j < C; ++j) if (!vis[j]) {
      double d = euclid(cent[cur], cent[j]);
      if (d < bd) { bd = d; best = j; }
    }
    order.push_back(best); vis[best] = 1; cur = best;
  }

  // 3) concatena orientando cada subtour (elige entrada mas cercana)
  std::vector<int> tour; tour.reserve(n);
  for (size_t oi = 0; oi < order.size(); ++oi) {
    int c = order[oi];
    auto& gt = cluster_tours[c];
    if (!tour.empty() && gt.size() > 1) {
      int last = tour.back();
      float d_front = inst.dist(last, gt.front());
      float d_back = inst.dist(last, gt.back());
      if (d_back < d_front) std::reverse(gt.begin(), gt.end());
    }
    for (int g : gt) tour.push_back(g);
  }
  // 4) pulido global barato: 1 pasada 2-opt con ventana (cose las uniones)
  if (n > 5000) two_opt(inst, tour, 1, 40);

  double L = inst.tour_length(tour);
  return {tour, L};
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  if (a.mode == "auto") a.mode = (a.n <= 5000 ? "full" : "hier");

  TSPInstance inst;
  inst.generate_random(a.n, a.seed);

  Timer timer;
  FullResult r;
  std::string used = a.mode;
  if (a.mode == "full") r = run_full(inst, a);
  else if (a.mode == "hier") r = run_hier(inst, a);
  else if (a.mode == "nn") {
    auto t = nn_tour(inst, 0);
    // 2-opt ligero solo si n moderado (sin matriz: al vuelo)
    if (a.use_2opt && a.n <= 5000) {
      TSPInstance& m = inst;
      two_opt(m, t, 1, (a.n > 500 ? 40 : 0));
    }
    r = {t, inst.tour_length(t)};
  } else { fprintf(stderr, "mode invalido: %s\n", a.mode.c_str()); return 2; }
  double tms = timer.ms();
  long peak = peak_rss_kb();

  printf("n=%d mode=%s ants=%d iters=%d seed=%u length=%.2f time_ms=%.1f peak_kb=%ld\n",
    a.n, used.c_str(), a.ants, a.iters, a.seed, r.len, tms, peak);
  printf("dense_mb_teorica=%.2f sparse_mb_est=%.2f (k=%d, solo modo full)\n",
    TSPInstance::dense_mb_theoretical(a.n),
    TSPInstance::sparse_mb(a.n, std::min(a.cand_k, a.n - 1)), a.cand_k);
  fflush(stdout);

  if (!a.out.empty()) {
    FILE* f = fopen(a.out.c_str(), "a");
    if (f) {
      // cabecera si archivo nuevo/vacio
      fseek(f, 0, SEEK_END);
      if (ftell(f) == 0)
        fprintf(f, "n,mode,ants,iters,seed,length,time_ms,peak_kb\n");
      fprintf(f, "%d,%s,%d,%d,%u,%.2f,%.1f,%ld\n",
        a.n, used.c_str(), a.ants, a.iters, a.seed, r.len, tms, peak);
      fclose(f);
    }
  }
  return 0;
}
