# -*- coding: utf-8 -*-
"""
Задание 8. Подбор вида зависимости между переменными с помощью регрессии
Данные: water_date.csv (температура T_degC от глубины Depthm)

Модели:
  a) линейная:             y = a0 + a1*x
  b) полиномиальная 2-й:   y = a0 + a1*x + a2*x^2
  в) логарифмическая:      y = a0 + a1*log(x)

Для каждой модели: анализ остатков, MSE, AIC.
Попарное сравнение вложенных моделей через ANOVA.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ---------------------------------------------------------------------------
# 0. Настройки
# ---------------------------------------------------------------------------
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

DATA_FILE = "water_date.csv"

# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
print("Размер данных:", df.shape)
print(df.head())

# Работаем с нужными переменными: x = Depthm, y = T_degC
df = df[["Depthm", "T_degC"]].dropna().copy()
df.rename(columns={"Depthm": "depth", "T_degC": "temp"}, inplace=True)

print("\nОписательные статистики:")
print(df.describe())

# Для логарифмической модели нужно depth > 0.
# При глубине = 0 log не определён -> исключаем нулевые глубины
# для корректного сравнения всех трёх моделей на одних и тех же данных.
df = df[df["depth"] > 0].reset_index(drop=True)
print(f"\nПосле удаления depth = 0 осталось наблюдений: {len(df)}")

# ---------------------------------------------------------------------------
# 2. График зависимости температуры от глубины
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.scatter(df["depth"], df["temp"], s=25, alpha=0.5, edgecolors="k", label="данные")
plt.xlabel("Глубина, м")
plt.ylabel("Температура, °C")
plt.title("Зависимость температуры от глубины")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temp_vs_depth.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 3. Построение трёх моделей
# ---------------------------------------------------------------------------
m_lin = smf.ols("temp ~ depth", data=df).fit()
m_poly = smf.ols("temp ~ depth + I(depth ** 2)", data=df).fit()
m_log = smf.ols("temp ~ np.log(depth)", data=df).fit()

for name, m in [("Линейная", m_lin), ("Полином 2", m_poly), ("Логарифм", m_log)]:
    print(f"\n=== {name} ===")
    print(m.summary().tables[1])

# ---------------------------------------------------------------------------
# 4. Кривые на одном графике
# ---------------------------------------------------------------------------
x_grid = np.linspace(df["depth"].min(), df["depth"].max(), 300)
grid = pd.DataFrame({"depth": x_grid})

plt.figure(figsize=(9, 5))
plt.scatter(df["depth"], df["temp"], s=25, alpha=0.4, edgecolors="k", label="данные", zorder=3)
plt.plot(x_grid, m_lin.predict(grid), lw=2, label="линейная")
plt.plot(x_grid, m_poly.predict(grid), lw=2, label="полином 2")
plt.plot(x_grid, m_log.predict(grid), lw=2, label="логарифм")
plt.xlabel("Глубина, м")
plt.ylabel("Температура, °C")
plt.title("Сравнение моделей")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("models_comparison.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 5. Диагностика: R², MSE, AIC, Shapiro по остаткам
# ---------------------------------------------------------------------------
def diagnose(model, name):
    resid = model.resid
    mse = np.mean(resid ** 2)
    # Shapiro на большой выборке (>5000) нестабилен, ограничимся подвыборкой
    sample = resid if len(resid) <= 5000 else np.random.choice(resid, 5000, replace=False)
    _, p_sw = stats.shapiro(sample)
    print(f"=== {name} ===")
    print(f"  R²        = {model.rsquared:.4f}")
    print(f"  Adj.R²    = {model.rsquared_adj:.4f}")
    print(f"  MSE       = {mse:.4f}")
    print(f"  AIC       = {model.aic:.2f}")
    print(f"  Shapiro p = {p_sw:.4f}")
    print()
    return {
        "model": name,
        "R2": model.rsquared,
        "AdjR2": model.rsquared_adj,
        "MSE": mse,
        "AIC": model.aic,
        "Shapiro_p": p_sw,
    }

cmp = pd.DataFrame([
    diagnose(m_lin, "линейная"),
    diagnose(m_poly, "полином 2"),
    diagnose(m_log, "логарифм"),
])
print("Сводная таблица:")
print(cmp.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Графики остатков vs предсказанные значения
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, model, title in zip(
    axes,
    [m_lin, m_poly, m_log],
    ["линейная", "полином 2", "логарифм"],
):
    ax.scatter(model.fittedvalues, model.resid, s=15, alpha=0.5, edgecolors="k")
    ax.axhline(0, color="red", ls="--")
    ax.set_xlabel("Предсказанные значения")
    ax.set_ylabel("Остатки")
    ax.set_title(title)
plt.suptitle("Остатки vs предсказанные значения", y=1.02)
plt.tight_layout()
plt.savefig("residuals.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 7. ANOVA для вложенных моделей
# ---------------------------------------------------------------------------
# Линейная ⊂ Полином 2 (полином включает линейный член)
print("\nANOVA: линейная vs полином 2")
print(anova_lm(m_lin, m_poly))

print("\nРанжирование по AIC (меньше — лучше):")
print(cmp.sort_values("AIC")[["model", "AIC", "MSE", "AdjR2"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Итоговый вывод
# ---------------------------------------------------------------------------
best = cmp.loc[cmp["AIC"].idxmin(), "model"]
print(f"\nПо AIC предпочтительна модель: {best}")