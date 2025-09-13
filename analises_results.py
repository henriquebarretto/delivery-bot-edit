import pandas as pd
import matplotlib.pyplot as plt
import os

# Criar pasta de saída
output_dir = "analysis"
os.makedirs(output_dir, exist_ok=True)

# Carregar os resultados
df = pd.read_csv("resultados.csv")

# ==============================
# 1) Score médio por agente
# ==============================
mean_scores = df.groupby("agent")["score"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
ax = mean_scores.plot(kind="bar", color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], edgecolor="black")
plt.title("Score médio por agente (todas as seeds e configs)", fontsize=13, fontweight="bold")
plt.ylabel("Score médio")
plt.xlabel("Agente")
plt.xticks(rotation=0)
ax.axhline(0, color="black", linewidth=1)

# Rótulos
ax.bar_label(ax.containers[0], fmt="%.1f", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "1score_medio_por_agente.png"))
plt.close()

# ==============================
# 2) Score médio por agente e seed (ordenado por seed crescente)
# ==============================
scores_by_seed = df.groupby(["seed","agent"])["score"].mean().unstack()

# ordenar seeds numericamente
scores_by_seed.index = scores_by_seed.index.astype(int)
scores_by_seed = scores_by_seed.sort_index()

plt.figure(figsize=(12,6))
ax = scores_by_seed.plot(kind="bar", ax=plt.gca(), colormap="tab10", edgecolor="black")
plt.title("Score médio por seed e agente", fontsize=13, fontweight="bold")
plt.ylabel("Score médio")
plt.xlabel("Seed")
plt.xticks(rotation=0)
ax.axhline(0, color="black", linewidth=1)

# Rótulos
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", fontsize=8, rotation=90, label_type="edge", padding=2)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "2score_por_seed.png"))
plt.close()

# ==============================
# 3) Comparação Sticky vs Non-Sticky
# ==============================
sticky_scores = df.groupby(["agent","sticky_target"])["score"].mean().unstack()

plt.figure(figsize=(8,5))
ax = sticky_scores.plot(kind="bar", ax=plt.gca(), color=["#E69F00", "#56B4E9"], edgecolor="black")
plt.title("Impacto do Sticky Target no Score médio", fontsize=13, fontweight="bold")
plt.ylabel("Score médio")
plt.xlabel("Agente")
plt.xticks(rotation=0)
ax.axhline(0, color="black", linewidth=1)

# Rótulos
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "3impacto_sticky.png"))
plt.close()

# ==============================
# 4) Smart: tuning de parâmetros
# ==============================
smart_df = df[df["agent"]=="smart"].copy()
if not smart_df.empty:
    scores = smart_df.groupby(["max_carry","urgent_threshold"])["score"].mean()

    plt.figure(figsize=(12,6))
    ax = scores.plot(kind="bar", color="#FF7F0E", edgecolor="black")
    plt.title("Smart agent - tuning de parâmetros (média do Score)", fontsize=13, fontweight="bold")
    plt.ylabel("Score médio")
    plt.xlabel("(max_carry, urgent_threshold)")
    plt.xticks(rotation=0)
    ax.axhline(0, color="black", linewidth=1)

    # Rótulos
    ax.bar_label(ax.containers[0], fmt="%.1f", fontsize=9, rotation=90, label_type="edge", padding=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4smart_param_tuning.png"))
    plt.close()

print("->> Gráficos gerados e salvos na pasta 'analysis'")
