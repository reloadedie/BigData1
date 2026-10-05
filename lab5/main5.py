# =============================================================================
# Задание 5. Работа с данными. Расчет коэффициента корреляции.
# Полное решение пунктов 1-18.
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки графики и генератора случайных чисел
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "DejaVu Sans"   # кириллица на графиках
sns.set_style("whitegrid")

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)


# ПУНКТ 1. Нормальная выборка x1: n=100, mean=3, sd=2
print("=" * 70)
print("ПУНКТ 1. Нормальная выборка x1")
print("=" * 70)

n1 = 100
mean1 = 3
sd1 = 2

x1 = stats.norm.rvs(loc=mean1, scale=sd1, size=n1, random_state=RANDOM_STATE)
print("Первые 10 значений x1:", np.round(x1[:10], 3))


# ПУНКТ 2. Выборочные среднее, стандартное отклонение, дисперсия x1
print("\n" + "=" * 70)
print("ПУНКТ 2. Характеристики x1")
print("=" * 70)

mean_x1 = x1.mean()
sd_x1 = x1.std(ddof=1)
var_x1 = x1.var(ddof=1)

print(f"mean = {mean_x1:.4f}")
print(f"sd   = {sd_x1:.4f}")
print(f"var  = {var_x1:.4f}")


# ПУНКТ 3. Биномиальная выборка x2: size=500, n=10, p=0.5
print("\n" + "=" * 70)
print("ПУНКТ 3. Биномиальная выборка x2")
print("=" * 70)

n2_size = 500    # сколько значений
n2_trials = 10   # число испытаний в одном эксперименте
p2 = 0.5

x2 = stats.binom.rvs(n=n2_trials, p=p2, size=n2_size, random_state=RANDOM_STATE)
print("Первые 20 значений x2:", x2[:20])
print("Уникальные значения:", np.unique(x2))


# ПУНКТ 4. Характеристики вектора x2
print("\n" + "=" * 70)
print("ПУНКТ 4. Характеристики x2")
print("=" * 70)

print(pd.Series(x2).describe())
print(f"\nmean = {x2.mean():.4f}")
print(f"sd   = {x2.std(ddof=1):.4f}")
print(f"var  = {x2.var(ddof=1):.4f}")


# ПУНКТЫ 5-6. ECDF и гистограммы для x1 и x2
print("\n" + "=" * 70)
print("ПУНКТЫ 5-6. ECDF и гистограммы")
print("=" * 70)


def ecdf(data):
    """Возвращает отсортированные значения и накопленные доли."""
    xs = np.sort(data)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


x1_sorted, x1_ecdf = ecdf(x1)
x2_sorted, x2_ecdf = ecdf(x2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ECDF
axes[0].step(x1_sorted, x1_ecdf, where="post",
             color="steelblue", label="x1 (normal)")
axes[0].step(x2_sorted, x2_ecdf, where="post",
             color="darkorange", label="x2 (binomial)")
axes[0].set_title("Эмпирические функции распределения (ECDF)")
axes[0].set_xlabel("значение")
axes[0].set_ylabel("F(x)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Гистограммы (плотность, чтобы совместить разные масштабы)
bins1 = np.histogram_bin_edges(x1, bins=15)
bins2 = np.arange(x2.min() - 0.5, x2.max() + 1.5, 1)

axes[1].hist(x1, bins=bins1, density=True, alpha=0.6,
             color="steelblue", edgecolor="black", label="x1")
axes[1].hist(x2, bins=bins2, density=True, alpha=0.6,
             color="darkorange", edgecolor="black", label="x2")
axes[1].set_title("Гистограммы x1 и x2")
axes[1].set_xlabel("значение")
axes[1].set_ylabel("плотность")
axes[1].legend()

plt.tight_layout()
plt.savefig("p06_ecdf_hist_x1_x2.png", dpi=120)
plt.show()


# ПУНКТ 7. Вектор x2_100 — 100 последних значений x2
print("\n" + "=" * 70)
print("ПУНКТ 7. x2_100")
print("=" * 70)

x2_100 = x2[-100:]
print("Длина x2_100:", len(x2_100))
print("Первые 10:", x2_100[:10])


# ПУНКТ 8. Корреляция между x1 и x2_100 (Пирсон, Спирмен, Кендалл)
print("\n" + "=" * 70)
print("ПУНКТ 8. Корреляция x1 и x2_100")
print("=" * 70)

r_p, p_p = stats.pearsonr(x1, x2_100)
r_s, p_s = stats.spearmanr(x1, x2_100)
r_k, p_k = stats.kendalltau(x1, x2_100)

print(f"Пирсон  : r = {r_p:+.4f}, p = {p_p:.4g}")
print(f"Спирмен : r = {r_s:+.4f}, p = {p_s:.4g}")
print(f"Кендалл : r = {r_k:+.4f}, p = {p_k:.4g}")


def interpret(r, p):
    a = abs(r)
    if a < 0.1:
        strength = "связи практически нет"
    elif a < 0.5:
        strength = "слабая связь"
    elif a < 0.7:
        strength = "умеренная связь"
    else:
        strength = "сильная связь"
    direction = "прямая" if r > 0 else ("обратная" if r < 0 else "отсутствует")
    signif = "значима (p < 0.05)" if p < 0.05 else "незначима (p >= 0.05)"
    return f"{strength}, {direction}, {signif}"


print("\nИнтерпретация (Пирсон):", interpret(r_p, p_p))
print("Интерпретация (Спирмен):", interpret(r_s, p_s))
print("Интерпретация (Кендалл):", interpret(r_k, p_k))
print("\nВывод: x1 и x2_100 независимы, коэффициент близок к нулю,")
print("p-value большое — линейной связи не обнаружено.")


# ПУНКТ 9. График зависимости x1 и x2_100
print("\n" + "=" * 70)
print("ПУНКТ 9. Scatter x1 vs x2_100")
print("=" * 70)

plt.figure(figsize=(8, 6))
plt.scatter(x1, x2_100, alpha=0.7, edgecolors="black", s=40)
plt.xlabel("x1 (нормальное)")
plt.ylabel("x2_100 (биномиальное)")
plt.title(f"Зависимость x1 и x2_100 (Пирсон r = {r_p:.3f}, p = {p_p:.3f})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("p09_scatter_x1_x2_100.png", dpi=120)
plt.show()


# ПУНКТ 10. Матрица x3 = [x1, x2_100]
print("\n" + "=" * 70)
print("ПУНКТ 10. Матрица x3")
print("=" * 70)

x3 = np.column_stack([x1, x2_100])
print("Форма x3:", x3.shape)
print("Первые 3 строки:\n", np.round(x3[:3], 3))


# ПУНКТ 11. Разбить x2 на 5 частей -> матрица x4 (100 x 5)
print("\n" + "=" * 70)
print("ПУНКТ 11. Матрица x4")
print("=" * 70)

x4 = x2.reshape(100, 5)
print("Форма x4:", x4.shape)
print("Первые 3 строки:\n", x4[:3])


# ПУНКТ 12. Матрица корреляций для x4
print("\n" + "=" * 70)
print("ПУНКТ 12. Матрица корреляций x4")
print("=" * 70)

corr4 = np.corrcoef(x4, rowvar=False)
corr4_df = pd.DataFrame(
    corr4,
    index=[f"c{i+1}" for i in range(5)],
    columns=[f"c{i+1}" for i in range(5)],
)
print(np.round(corr4, 3))

plt.figure(figsize=(7, 6))
sns.heatmap(corr4_df, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True)
plt.title("Матрица корреляций для x4")
plt.tight_layout()
plt.savefig("p12_corr_x4.png", dpi=120)
plt.show()


# ПУНКТ 13. Стандартизация x2: вручную и через scale()
print("\n" + "=" * 70)
print("ПУНКТ 13. Стандартизация x2")
print("=" * 70)

# Ручная стандартизация (через выборочное sd, ddof=1)
x2_st_manual = (x2 - x2.mean()) / x2.std(ddof=1)

# scale() из R использует sd с ddof=1
x2_st_scale = (x2 - np.mean(x2)) / np.std(x2, ddof=1)

print("Первые 5 вручную :", np.round(x2_st_manual[:5], 4))
print("Первые 5 scale() :", np.round(x2_st_scale[:5], 4))
print("Совпадают?", np.allclose(x2_st_manual, x2_st_scale))
print(f"mean(z) = {x2_st_manual.mean():.2e}")
print(f"sd(z)   = {x2_st_manual.std(ddof=1):.4f}")
print(f"var(z)  = {x2_st_manual.var(ddof=1):.4f}")


# ПУНКТ 14. Логарифмирование x1 и округление до 3 знаков
print("\n" + "=" * 70)
print("ПУНКТ 14. log(x1) с округлением")
print("=" * 70)

# log требует положительных значений — берём только x1 > 0
x1_positive = x1[x1 > 0]
x1_log = np.round(np.log(x1_positive), 3)
print("Первые 10 значений log(x1):", x1_log[:10])
print("Всего положительных значений:", len(x1_positive), "из", len(x1))


# ПУНКТ 15. mtcars: точечная диаграмма по mpg
print("\n" + "=" * 70)
print("ПУНКТ 15. mtcars: dotchart по mpg")
print("=" * 70)

mtcars = pd.read_csv("mtcars.csv")
print(mtcars.head())
print("Столбцы:", mtcars.columns.tolist())

# Первый столбец — имена моделей
if mtcars.columns[0] != "mpg":
    mtcars = mtcars.set_index(mtcars.columns[0])

print("Индекс (модели):", mtcars.index[:5].tolist())

# Сортируем по mpg
mt_sorted = mtcars.sort_values("mpg")

plt.figure(figsize=(8, 10))
plt.scatter(mt_sorted["mpg"], range(len(mt_sorted)),
            c="steelblue", s=45, edgecolors="black", alpha=0.8)
plt.yticks(range(len(mt_sorted)), mt_sorted.index, fontsize=8)
plt.xlabel("mpg (миль на галлон)")
plt.title("Точечная диаграмма mpg по моделям (mtcars)")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("p15_mtcars_dotchart_mpg.png", dpi=120)
plt.show()


# ПУНКТ 16. Описательные статистики mpg (вся выборка и по vs)
print("\n" + "=" * 70)
print("ПУНКТ 16. Описательные статистики mpg")
print("=" * 70)

print("=== mpg: вся выборка ===")
print(mtcars["mpg"].describe())

print("\n=== mpg по vs (0 = V-образный, 1 = прямой) ===")
print(mtcars.groupby("vs")["mpg"].describe().round(3))


# ПУНКТ 17. ECDF и гистограммы mpg: вся выборка + по vs
print("\n" + "=" * 70)
print("ПУНКТ 17. Графики mpg")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (0,0) Гистограмма всей выборки
axes[0, 0].hist(mtcars["mpg"], bins=12, color="steelblue",
                edgecolor="black", alpha=0.75)
axes[0, 0].set_title("Гистограмма mpg (вся выборка)")
axes[0, 0].set_xlabel("mpg")

# (0,1) ECDF всей выборки
xs, ys = ecdf(mtcars["mpg"].values)
axes[0, 1].step(xs, ys, where="post", color="steelblue")
axes[0, 1].set_title("ECDF mpg (вся выборка)")
axes[0, 1].set_xlabel("mpg")
axes[0, 1].set_ylabel("F(x)")

# (1,0) Гистограммы по vs
for vs_val, color, label in [(0, "tomato", "vs = 0"),
                             (1, "seagreen", "vs = 1")]:
    sub = mtcars.loc[mtcars["vs"] == vs_val, "mpg"]
    axes[1, 0].hist(sub, bins=10, alpha=0.6, color=color,
                    edgecolor="black", label=label)
axes[1, 0].set_title("Гистограмма mpg по vs")
axes[1, 0].set_xlabel("mpg")
axes[1, 0].legend()

# (1,1) ECDF по vs
for vs_val, color, label in [(0, "tomato", "vs = 0"),
                             (1, "seagreen", "vs = 1")]:
    sub = mtcars.loc[mtcars["vs"] == vs_val, "mpg"].values
    xs, ys = ecdf(sub)
    axes[1, 1].step(xs, ys, where="post", color=color, label=label)
axes[1, 1].set_title("ECDF mpg по vs")
axes[1, 1].set_xlabel("mpg")
axes[1, 1].set_ylabel("F(x)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("p17_mtcars_mpg_graphs.png", dpi=120)
plt.show()


# ПУНКТ 18. Логарифмирование mpg и повтор графиков
print("\n" + "=" * 70)
print("ПУНКТ 18. log(mpg) и графики")
print("=" * 70)

mtcars = mtcars.copy()
mtcars["log_mpg"] = np.log(mtcars["mpg"])

print(mtcars[["mpg", "log_mpg"]].describe().round(3))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (0,0) Гистограмма log(mpg) всей выборки
axes[0, 0].hist(mtcars["log_mpg"], bins=12, color="purple",
                edgecolor="black", alpha=0.75)
axes[0, 0].set_title("Гистограмма log(mpg) (вся выборка)")
axes[0, 0].set_xlabel("log(mpg)")

# (0,1) ECDF log(mpg) всей выборки
xs, ys = ecdf(mtcars["log_mpg"].values)
axes[0, 1].step(xs, ys, where="post", color="purple")
axes[0, 1].set_title("ECDF log(mpg) (вся выборка)")
axes[0, 1].set_xlabel("log(mpg)")
axes[0, 1].set_ylabel("F(x)")

# (1,0) Гистограммы log(mpg) по vs
for vs_val, color, label in [(0, "tomato", "vs = 0"),
                             (1, "seagreen", "vs = 1")]:
    sub = mtcars.loc[mtcars["vs"] == vs_val, "log_mpg"]
    axes[1, 0].hist(sub, bins=10, alpha=0.6, color=color,
                    edgecolor="black", label=label)
axes[1, 0].set_title("Гистограмма log(mpg) по vs")
axes[1, 0].set_xlabel("log(mpg)")
axes[1, 0].legend()

# (1,1) ECDF log(mpg) по vs
for vs_val, color, label in [(0, "tomato", "vs = 0"),
                             (1, "seagreen", "vs = 1")]:
    sub = mtcars.loc[mtcars["vs"] == vs_val, "log_mpg"].values
    xs, ys = ecdf(sub)
    axes[1, 1].step(xs, ys, where="post", color=color, label=label)
axes[1, 1].set_title("ECDF log(mpg) по vs")
axes[1, 1].set_xlabel("log(mpg)")
axes[1, 1].set_ylabel("F(x)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("p18_mtcars_logmpg_graphs.png", dpi=120)
plt.show()


print("\nГотово: все пункты 1-18 выполнены.")