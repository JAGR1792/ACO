#!/bin/bash
# Benchmark ACO-TSP: n=20, 2000, 200000. Mide tiempo y memoria.
set -u
BIN="$(dirname "$0")/../build/aco"
OUT="$(dirname "$0")/../results.csv"
rm -f "$OUT"

echo "== build =="
cmake -S "$(dirname "$0")/.." -B "$(dirname "$0")/../build" -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build "$(dirname "$0")/../build" -j"$(nproc)"
echo "== runs =="
# 20: ACO completo pequeño, rápido
/usr/bin/time -v "$BIN" --n 20 --ants 20 --iters 100 --cand 8 --seed 42 --mode full --out "$OUT" 2>&1 | tail -n 25
# 2000: ACO con candidatos
/usr/bin/time -v "$BIN" --n 2000 --ants 25 --iters 50 --cand 25 --seed 42 --mode full --out "$OUT" 2>&1 | tail -n 25
# 200000: jerárquico (no matriz densa)
/usr/bin/time -v "$BIN" --n 200000 --seed 42 --mode hier --target-cluster 600 --out "$OUT" 2>&1 | tail -n 25
echo "== results.csv =="
cat "$OUT"
