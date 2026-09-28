# ACO para el agente viajero (TSP)

Implementación en C++17 con OpenMP. Todo en español.

## Estructura

- `src/tsp.h`: instancia euclidiana, matriz densa y lista k-NN.
- `src/aco.h`: MMAS (solo mejor global deposita, límites tau) + 2-opt.
- `src/main.cpp`: modos `full` (n ≤ 5000), `hier` (n grande), `nn` (base).
- `scripts/run_benchmark.sh`: corre n=20, 2000, 200000 y guarda `results.csv`.
- `scripts/plot.py`: grafica tiempo, memoria y longitud.

## Compilar

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Usar

```bash
./build/aco --n 20 --ants 20 --iters 100 --cand 8 --seed 42 --mode full
./build/aco --n 2000 --ants 25 --iters 50 --cand 25 --seed 42 --mode full --out results.csv
./build/aco --n 200000 --seed 42 --mode hier --target-cluster 600 --out results.csv
./build/aco --help
```

## Notas de escala

- ACO denso es O(n²) en memoria. n=2000 ≈ 16 MB (float). n=200000 ≈ 160 GB: imposible.
- Por eso `hier` particiona en rejilla (~600 pts por celda), corre ACO por celda y une por centroides.
