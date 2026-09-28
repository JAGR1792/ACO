"""Visualizacion de la regla de transicion p_ij^k con el ejemplo de clase.
Desde la ciudad actual hay 3 candidatas (alpha=1, beta=2):
  A: d=4, tau=2 | B: d=6, tau=3 | C: d=8, tau=1
Muestra: peso = tau^alpha * eta^beta por candidata, normalizacion y ruleta.
"""
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})
ALPHA, BETA = 1, 2
dests = ["A", "B", "C"]
d = {"A": 4.0, "B": 6.0, "C": 8.0}
tau = {"A": 2.0, "B": 3.0, "C": 1.0}
eta = {k: 1.0 / d[k] for k in dests}
w = {k: (tau[k] ** ALPHA) * (eta[k] ** BETA) for k in dests}
S = sum(w.values())
p = {k: w[k] / S for k in dests}
print({k: (round(w[k], 4), round(p[k], 3)) for k in dests}, "suma =", round(S, 4))

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
fig.subplots_adjust(top=0.82, bottom=0.12)

# 1. Pesos: tau^alpha * eta^beta (anotacion en una linea, con aire arriba)
x = range(3)
ax[0].bar(x, [w[k] for k in dests], tick_label=dests)
ax[0].set_ylim(0, max(w.values()) * 1.6)
for i, k in enumerate(dests):
    ax[0].text(i, w[k] * 1.05, f"{tau[k]:.0f}*(1/{d[k]:.0f})^2={w[k]:.4f}",
               ha="center", va="bottom", fontsize=11)
ax[0].set_title("1. Peso = tau^alpha x eta^beta")
ax[0].set_ylabel("peso")

# 2. Normalizacion: peso / suma
ax[1].bar(x, [p[k] for k in dests], tick_label=dests)
ax[1].set_ylim(0, max(p.values()) * 2.0)
for i, k in enumerate(dests):
    ax[1].text(i, p[k] * 1.05, f"{w[k]:.4f}/{S:.4f}={p[k]:.3f}",
               ha="center", va="bottom", fontsize=11)
ax[1].set_title("2. Probabilidad = peso/suma")
ax[1].set_ylabel("probabilidad")

# 3. Ruleta con r=0.71 -> elige B (etiqueta DEBAJO de la barra, lejos del titulo)
acum, colors = [0.0], ["#2ca02c", "#ff7f0e", "#1f77b4"]
for k in dests:
    acum.append(acum[-1] + p[k])
for i, k in enumerate(dests):
    ax[2].barh(0, p[k], left=acum[i], color=colors[i])
    ax[2].text(acum[i] + p[k] / 2, 0, k, ha="center", va="center",
               fontsize=16, color="white", weight="bold")
ax[2].plot([0.71, 0.71], [-0.4, 0.4], "r--")
ax[2].plot([acum[2], acum[2]], [-0.4, 0.4], "k:")
ax[2].text(0.71, -0.75, "r=0.71 cae en B", ha="center", fontsize=13, color="red")
ax[2].text(acum[2], 0.55, "0.93", ha="center", fontsize=11)
ax[2].set_xlim(0, 1)
ax[2].set_xticks([0, acum[1], 1.0])
ax[2].set_xticklabels(["0", f"{acum[1]:.3f}", "1"])
ax[2].set_ylim(-1.1, 1.0)
ax[2].set_yticks([])
ax[2].set_title("3. Ruleta: gana B (A era mas probable)")

plt.savefig("img/ruleta.png")
print("img/ruleta.png")
