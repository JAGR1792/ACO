"""Visualizacion de la regla de transicion p_ij^k con el ejemplo de clase.
Desde la ciudad actual hay 3 candidatas (alpha=1, beta=2):
  A: d=4, tau=2 | B: d=6, tau=3 | C: d=8, tau=1
Muestra: peso = tau^alpha * eta^beta por candidata, normalizacion y ruleta.
"""
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13})
ALPHA, BETA = 1, 2
dests = ["A", "B", "C"]
d = {"A": 4.0, "B": 6.0, "C": 8.0}
tau = {"A": 2.0, "B": 3.0, "C": 1.0}
eta = {k: 1.0 / d[k] for k in dests}
w = {k: (tau[k] ** ALPHA) * (eta[k] ** BETA) for k in dests}
S = sum(w.values())
p = {k: w[k] / S for k in dests}
print({k: (round(w[k], 4), round(p[k], 3)) for k in dests}, "suma =", round(S, 4))

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

# 1. Pesos: tau^alpha * eta^beta
x = range(3)
ax[0].bar(x, [w[k] for k in dests], tick_label=dests)
for i, k in enumerate(dests):
    ax[0].text(i, w[k], f"{tau[k]}x(1/{d[k]:.0f})^2\n={w[k]:.4f}",
               ha="center", va="bottom", fontsize=12)
ax[0].set_title("1. Peso de cada candidata\ntau^alpha x eta^beta")
ax[0].set_ylabel("peso")

# 2. Normalizacion: peso / suma
ax[1].bar(x, [p[k] for k in dests], tick_label=dests)
for i, k in enumerate(dests):
    ax[1].text(i, p[k], f"{w[k]:.4f}/{S:.4f}\n={p[k]:.3f}",
               ha="center", va="bottom", fontsize=12)
ax[1].set_title("2. Probabilidad\npeso / suma de pesos")
ax[1].set_ylabel("probabilidad")

# 3. Ruleta con r=0.71 -> elige B
acum, colors = [0.0], ["#2ca02c", "#ff7f0e", "#1f77b4"]
for k in dests:
    acum.append(acum[-1] + p[k])
for i, k in enumerate(dests):
    ax[2].barh(0, p[k], left=acum[i], color=colors[i])
    ax[2].text(acum[i] + p[k] / 2, 0, k, ha="center", va="center",
               fontsize=14, color="white", weight="bold")
ax[2].plot([0.71, 0.71], [-0.4, 0.4], "r--")
ax[2].text(0.71, 0.45, "r=0.71 -> B", ha="center", fontsize=13, color="red")
ax[2].set_xlim(0, 1)
ax[2].set_xticks([0, acum[1], acum[2], 1.0])
ax[2].set_xticklabels(["0", f"{acum[1]:.3f}", f"{acum[2]:.3f}", "1"])
ax[2].set_yticks([])
ax[2].set_title("3. Ruleta: cae en B\n(aunque A era mas probable)")

plt.tight_layout()
plt.savefig("img/ruleta.png")
print("img/ruleta.png")
