"""Graficas extra: convergencia, escalamiento y sensibilidad a k."""
import csv, os
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 14})

BASE = os.path.join(os.path.dirname(__file__), "..")
IMG = os.path.join(BASE, "img")

def read_csv(path):
    with open(os.path.join(BASE, path)) as f:
        return list(csv.DictReader(f))

# ---------- 1. Convergencia: mejor longitud vs iteracion ----------
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for i, (trace, title) in enumerate([("trace_20.csv", "n=20 (20 hormigas x 100 iters)"),
                                    ("trace_2000.csv", "n=2000 (25 hormigas x 50 iters)")]):
    it, best = [], []
    with open(os.path.join(BASE, trace)) as f:
        for row in csv.DictReader(f):
            it.append(int(row["iter"])); best.append(float(row["best"]))
    ax[i].plot(it, best)
    ax[i].set_title(title); ax[i].set_xlabel("iteracion"); ax[i].set_ylabel("mejor longitud")
    ax[i].grid(True)
    ax[i].annotate(f"ini {best[0]:.0f} -> fin {best[-1]:.0f}\nmejora {(1-best[-1]/best[0])*100:.1f}%",
                   xy=(0.6, 0.85), xycoords="axes fraction", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "convergencia.png")); print("img/convergencia.png")

# ---------- 2. Escalamiento: tiempo y memoria vs n (log-log) ----------
scal = read_csv("scaling.csv")
ns = [int(r["n"]) for r in scal]
t = [float(r["time_ms"]) / 1000.0 for r in scal]
mem = [float(r["peak_kb"]) / 1024.0 for r in scal]

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].loglog(ns, t, "o-", label="full (25x50, k=25)")
ax[0].loglog([200000], [6.988], "s", label="hier (200000)")
ax[0].set_title("Tiempo vs n"); ax[0].set_xlabel("n (ciudades)"); ax[0].set_ylabel("segundos")
ax[0].grid(True, which="both"); ax[0].legend(fontsize=8)
# referencia O(n): recta pendiente 1 anclada en n=1000
ref = [t[2] * (n / ns[2]) for n in ns]
ax[0].loglog(ns, ref, "--", label="referencia O(n)", linewidth=1)
ax[0].legend(fontsize=8)
ax[1].loglog(ns, mem, "o-", label="full (25x50, k=25)")
ax[1].loglog([200000], [75.0], "s", label="hier (200000)")
ax[1].set_title("Memoria pico vs n"); ax[1].set_xlabel("n (ciudades)"); ax[1].set_ylabel("MB")
ax[1].grid(True, which="both")
ref2 = [mem[2] * (n / ns[2]) ** 2 for n in ns]
ax[1].loglog(ns, ref2, "--", label="referencia O(n^2)", linewidth=1)
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(IMG, "escalamiento.png")); print("img/escalamiento.png")

# ---------- 3. Sensibilidad a k (n=2000) ----------
# Orden de filas en sensibilidad.csv: k=5,15,25,50 (n=2000), k=25 sin 2-opt,
# luego n=200 con k=0 y k=10.
sens = read_csv("sensibilidad.csv")
labels = ["k=5", "k=15", "k=25", "k=50", "k=25\nsin 2-opt"]
ln = [float(r["length"]) for r in sens[:5]]
tm = [float(r["time_ms"]) / 1000.0 for r in sens[:5]]
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].bar(labels, tm); ax[0].set_title("Tiempo vs k (n=2000)"); ax[0].set_ylabel("segundos")
ax[1].bar(labels, ln); ax[1].set_title("Longitud vs k (n=2000)"); ax[1].set_ylabel("longitud tour")
plt.tight_layout(); plt.savefig(os.path.join(IMG, "candidatos.png")); print("img/candidatos.png")
print("demo n=200: k=0 -> %.1fs / k=10 -> %.1fs" % (
    float(sens[5]["time_ms"]) / 1000.0, float(sens[6]["time_ms"]) / 1000.0))
