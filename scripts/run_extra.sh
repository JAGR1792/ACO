#!/bin/bash
# Experimentos extra: escalamiento, sensibilidad a k y trazas de convergencia.
set -u
cd "$(dirname "$0")/.."
BIN=build/aco

cmake --build build -j"$(nproc)" > /dev/null

echo "== escalamiento (ants=25 iters=50 cand=25 seed=42) =="
rm -f scaling.csv
for n in 100 500 1000 2000 5000; do
  ./$BIN --n $n --ants 25 --iters 50 --cand 25 --seed 42 --mode full --out scaling.csv
done

echo "== trazas de convergencia =="
./$BIN --n 20 --ants 20 --iters 100 --cand 8 --seed 42 --mode full --trace-out trace_20.csv > /dev/null
./$BIN --n 2000 --ants 25 --iters 50 --cand 25 --seed 42 --mode full --trace-out trace_2000.csv > /dev/null

echo "== sensibilidad a k (n=2000, seed 42) =="
rm -f sensibilidad.csv
for k in 5 15 25 50; do
  ./$BIN --n 2000 --ants 25 --iters 50 --cand $k --seed 42 --mode full --out sensibilidad.csv
done
./$BIN --n 2000 --ants 25 --iters 50 --cand 25 --seed 42 --mode full --no-2opt --out sensibilidad.csv

echo "== demo sin candidatos (n=200, 10x20) =="
./$BIN --n 200 --ants 10 --iters 20 --cand 0 --seed 42 --mode full --out sensibilidad.csv
./$BIN --n 200 --ants 10 --iters 20 --cand 10 --seed 42 --mode full --out sensibilidad.csv

echo "== listo =="
cat scaling.csv
cat sensibilidad.csv
