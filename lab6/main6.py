# =============================================================
# Задание 6. Построение матрицы корреляции
# Дисциплина: Введение в анализ больших данных
# Датасет: NMES1988 (медицинское обследование)
# =============================================================

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")


# -------------------------------------------------------------
# 1. Загрузка и подготовка данных
# -------------------------------------------------------------
data = pd.read_csv("NMES1988.csv", index_col=0)

cols = ["visits", "health", "chronic", "adl", "region", "age",
        "gender", "married", "school", "income", "employed", "insurance"]
data = data[cols]

print("Размерность:", data.shape)
print("Столбцы:", data.columns.tolist())

# Количественные показатели (по заданию)
quant_cols = ["visits", "chronic", "age", "school", "income"]
print("\nКоличественные переменные:", quant_cols)


# -------------------------------------------------------------
# 2. Проверка нормальности (Shapiro–Wilk)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. Проверка нормальности (Shapiro–Wilk)")
print("=" * 60)
print("H0: выборка из нормального распределения")
print("p > 0.05 → H0 не отвергается\n")

for col in quant_cols:
    x = data[col].dropna()
    W, p = stats.shapiro(x)
    verdict = "нормально" if p > 0.05 else "НЕ нормально"
    print(f"{col:10s}: W={W:.4f}, p={p:.4e} → {verdict}")

# Гистограммы для визуальной проверки
fig, axes = plt.subplots(1, len(quant_cols), figsize=(3.5 * len(quant_cols), 4))
for ax, col in zip(axes, quant_cols):
    ax.hist(data[col], bins=30, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel("")
plt.suptitle("Распределения количественных переменных", fontsize=13)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# 3. Корреляция Спирмена для visits и age
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Спирмен: visits ~ age")
print("=" * 60)

rho, p = stats.spearmanr(data["visits"], data["age"], nan_policy="omit")
print(f"rho = {rho:.4f}")
print(f"p   = {p:.4e}")
print(f"Интерпретация: {'значимая' if p < 0.05 else 'незначимая'} связь, "
      f"{'положительная' if rho > 0 else 'отрицательная'}, "
      f"сила: {'слабая' if abs(rho) < 0.3 else 'средняя' if abs(rho) < 0.5 else 'сильная'}")


# -------------------------------------------------------------
# 4. Матрица корреляций + heatmap
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Матрица корреляций (Спирмен)")
print("=" * 60)

corr = data[quant_cols].corr(method="spearman")
print(corr.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            vmin=-1, vmax=1, square=True, ax=axes[0])
axes[0].set_title("Матрица корреляций Спирмена")

# Только числа без аннотаций — «чистый» вид
sns.heatmap(corr, annot=False, cmap="RdYlBu_r", center=0,
            vmin=-1, vmax=1, square=True, ax=axes[1])
axes[1].set_title("Матрица корреляций (цветами)")
plt.tight_layout()
plt.show()


# -------------------------------------------------------------
# 5. Значимость коэффициентов < 0.5
#    Вся выборка / мужчины / женщины
# -------------------------------------------------------------
def significant_pairs(df, quant_cols, threshold=0.5):
    """Возвращает список пар с |r| < threshold и их значимостью."""
    rows = []
    for i, c1 in enumerate(quant_cols):
        for c2 in quant_cols[i + 1:]:
            sub = df[[c1, c2]].dropna()
            rho, p = stats.spearmanr(sub[c1], sub[c2])
            if abs(rho) < threshold:
                rows.append({
                    "pair": f"{c1} ~ {c2}",
                    "rho": rho,
                    "p": p,
                    "significant": p < 0.05,
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


print("\n" + "=" * 60)
print("5. Значимость корреляций |r| < 0.5")
print("=" * 60)

# Вся выборка
all_pairs = significant_pairs(data, quant_cols)
print("\n--- Вся выборка ---")
print(all_pairs.round(4).to_string(index=False))

# По полу
results_by_gender = {}
for g in data["gender"].dropna().unique():
    sub = data[data["gender"] == g]
    res = significant_pairs(sub, quant_cols)
    results_by_gender[g] = res
    print(f"\n--- gender = {g} (n={len(sub)}) ---")
    print(res.round(4).to_string(index=False))

# Визуализация: сравнение rho для пар по полу
if "male" in results_by_gender and "female" in results_by_gender:
    m = results_by_gender["male"].set_index("pair")["rho"]
    f = results_by_gender["female"].set_index("pair")["rho"]
    compare = pd.DataFrame({"male": m, "female": f}).dropna()
    if len(compare) > 0:
        compare.plot(kind="bar", figsize=(10, 5), edgecolor="black")
        plt.title("Сравнение rho для пар |r| < 0.5: мужчины vs женщины")
        plt.ylabel("Spearman rho")
        plt.xticks(rotation=30, ha="right")
        plt.axhline(0, color="black", lw=0.8)
        plt.tight_layout()
        plt.show()


# -------------------------------------------------------------
# 6. visits vs остальные количественные, группировка по region
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("6. visits vs другие количественные показатели по region")
print("=" * 60)

others = ["chronic", "age", "school", "income"]
rows = []
for region in data["region"].dropna().unique():
    sub = data[data["region"] == region]
    for col in others:
        s = sub[["visits", col]].dropna()
        if len(s) < 3:
            continue
        rho, p = stats.spearmanr(s["visits"], s[col])
        rows.append({
            "region": region,
            "variable": col,
            "rho": rho,
            "p": p,
            "n": len(s),
        })

res_region = pd.DataFrame(rows)
print(res_region.round(4).to_string(index=False))

# Heatmap: region × variable, значения rho
pivot = res_region.pivot(index="region", columns="variable", values="rho")
print("\nСводная таблица rho:")
print(pivot.round(3))

plt.figure(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu_r", center=0,
            vmin=-1, vmax=1)
plt.title("Spearman: visits ~ другие переменные, по region")
plt.tight_layout()
plt.show()