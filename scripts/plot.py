import csv, sys, os
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results.csv")
rows = list(csv.DictReader(open(csv_path)))
if not rows:
    print("sin datos en", csv_path); sys.exit(1)

ns = [int(r["n"]) for r in rows]
times = [float(r["time_ms"]) / 1000.0 for r in rows]
peaks = [float(r["peak_kb"]) / 1024.0 for r in rows]
lens = [float(r["length"]) for r in rows]
modes = [r["mode"] for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].bar([str(n) + "\n" + m for n, m in zip(ns, modes)], times)
ax[0].set_title("Tiempo (s)")
ax[0].set_ylabel("segundos")

ax[1].bar([str(n) + "\n" + m for n, m in zip(ns, modes)], peaks)
ax[1].set_title("Memoria pico (MB)")

ax[2].bar([str(n) + "\n" + m for n, m in zip(ns, modes)], lens)
ax[2].set_title("Longitud tour")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "benchmark.png")
plt.savefig(out)
print("guardado:", out)
