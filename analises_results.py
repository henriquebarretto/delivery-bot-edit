import pandas as pd
import matplotlib.pyplot as plt

# Carregar os resultados
df = pd.read_csv("resultados.csv")

# Garantir que a seed é string
df["seed"] = df["seed"].astype(str)

# ==============================
# 1) Score médio por agente
# ==============================
mean_scores = df.groupby("agent")["score"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
ax = mean_scores.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Score médio por agente (todas as seeds e configs)")
plt.ylabel("Score médio")
plt.xlabel("Agente")
plt.xticks(rotation=0)

# Rótulos em cima das barras
ax.bar_label(ax.containers[0], fmt="%.1f")

plt.tight_layout()
plt.savefig("1score_medio_por_agente.png")
plt.show()

# ==============================
# 2) Score médio por agente e seed
# ==============================
plt.figure(figsize=(10,6))
ax = df.groupby(["seed","agent"])["score"].mean().unstack().plot(kind="bar", ax=plt.gca())
plt.title("Score médio por seed e por agente")
plt.ylabel("Score médio")
plt.xlabel("Seed")
plt.legend(title="Agente")
plt.xticks(rotation=0)
ax.axhline(0, color="black", linewidth=1)

# Rótulos em cima de cada barra
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.savefig("2score_por_seed.png")
plt.show()

# ==============================
# 3) Comparação Sticky vs Non-Sticky
# ==============================
plt.figure(figsize=(8,5))
ax = df.groupby(["agent","sticky_target"])["score"].mean().unstack().plot(kind="bar", ax=plt.gca())
plt.title("Impacto do Sticky Target no Score médio")
plt.ylabel("Score médio")
plt.xlabel("Agente")
plt.xticks(rotation=0)

# Rótulos
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f")

plt.tight_layout()
plt.savefig("3impacto_sticky.png")
plt.show()

# ==============================
# 4) Smart: tuning de parâmetros
# ==============================
smart_df = df[df["agent"]=="smart"].copy()
if not smart_df.empty:
    # Agrupamento já está correto
    scores = smart_df.groupby(["max_carry","urgent_threshold"])["score"].mean()

    plt.figure(figsize=(10,6))
    ax = scores.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Smart agent - comparação de parâmetros (média do Score)")
    plt.ylabel("Score médio")
    plt.xlabel("(max_carry, urgent_threshold)")
    plt.xticks(rotation=0)

    # Rótulos
    ax.bar_label(ax.containers[0], fmt="%.1f")

    plt.tight_layout()
    plt.savefig("4smart_param_tuning.png")
    plt.show()

print("->> Gráficos gerados com rótulos: score_medio_por_agente.png, score_por_seed.png, impacto_sticky.png, smart_param_tuning.png")
