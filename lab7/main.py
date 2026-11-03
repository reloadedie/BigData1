# ЗАДАНИЕ 7. Регрессионный анализ данных о черепахах

# ---------- 0. Импорт библиотек ----------
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan

# Настройки отображения
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# ---------- 1. Загрузка данных ----------
# Файл Turtle.csv использует разделитель ';'
df = pd.read_csv("Turtle.csv", sep=";")

print("=== Первые 5 строк данных ===")
print(df.head())
print("\n=== Описательные статистики ===")
print(df.describe().round(2))
print("\n=== Типы данных ===")
print(df.dtypes)

# ---------- 2. Проверка нормальности (Шапиро-Уилк) ----------
print("\n=== Тест Шапиро-Уилка на нормальность ===")
normality_results = {}
for col in df.columns:
    W, p = stats.shapiro(df[col])
    normality_results[col] = {"W": W, "p": p}
    conclusion = "нормально" if p > 0.05 else "НЕ нормально"
    print(f"{col:10s}: W = {W:.4f}, p = {p:.4g}  ->  {conclusion}")

# ---------- 3. Корреляционная матрица (Спирмен) ----------
print("\n=== Корреляционная матрица (Spearman) ===")
corr_spearman = df.corr(method="spearman")
print(corr_spearman.round(3))

# Матрица p-значений для корреляций (попарно)
print("\n=== p-значения корреляций (Spearman) ===")
cols = df.columns
p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
for i in cols:
    for j in cols:
        if i == j:
            p_matrix.loc[i, j] = 0.0
        else:
            r, p = stats.spearmanr(df[i], df[j])
            p_matrix.loc[i, j] = p
print(p_matrix.round(4))

# Визуализация корреляционной матрицы
plt.figure(figsize=(8, 6))
sns.heatmap(corr_spearman, annot=True, fmt=".3f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True)
plt.title("Корреляционная матрица (Spearman)")
plt.tight_layout()
plt.show()

# Парные графики
sns.pairplot(df)
plt.suptitle("Парные зависимости показателей черепах", y=1.02)
plt.show()

# ---------- 4. Множественная линейная регрессия ----------
# Зависимая переменная: weight
# Предикторы: length, width, height
print("\n=== Множественная линейная регрессия: weight ~ length + width + height ===")
model = smf.ols("weight ~ length + width + height", data=df).fit()
print(model.summary())

# Уравнение регрессии
coef = model.params
print("\n=== Уравнение регрессии ===")
print(f"weight = {coef['Intercept']:.4f} + {coef['length']:.4f}*length "
      f"+ {coef['width']:.4f}*width + {coef['height']:.4f}*height")

# Коэффициент детерминации
print(f"\nR² = {model.rsquared:.4f}")
print(f"Adj. R² = {model.rsquared_adj:.4f}")
print(f"F-статистика = {model.fvalue:.4f}, p-value = {model.f_pvalue:.4g}")

# ---------- 5. Анализ остатков ----------
resid = model.resid
fitted = model.fittedvalues

print("\n=== Анализ остатков ===")

# 5.1 Нормальность остатков (Шапиро-Уилк)
W_res, p_res = stats.shapiro(resid)
print(f"Шапиро-Уилк для остатков: W = {W_res:.4f}, p = {p_res:.4g}")
if p_res > 0.05:
    print("  -> Остатки распределены нормально")
else:
    print("  -> Остатки НЕ распределены нормально")

# 5.2 Автокорреляция (тест Бреуша-Годфри)
bg = acorr_breusch_godfrey(model, nlags=1)
print(f"\nТест Бреуша-Годфри (автокорреляция): LM = {bg[0]:.4f}, "
      f"p = {bg[1]:.4g}, df = {bg[2]}")
if bg[1] > 0.05:
    print("  -> Автокорреляция отсутствует (остатки независимы)")
else:
    print("  -> Обнаружена автокорреляция")

# 5.3 Гетероскедастичность (тест Бреуша-Пагана)
bp = het_breuschpagan(resid, model.model.exog)
print(f"\nТест Бреуша-Пагана (гетероскедастичность): BP = {bp[0]:.4f}, "
      f"p = {bp[1]:.4g}")
if bp[1] > 0.05:
    print("  -> Гетероскедастичность отсутствует (гомоскедастичность)")
else:
    print("  -> Обнаружена гетероскедастичность")

# 5.4 MSE / RMSE
mse = np.mean(resid ** 2)
rmse = np.sqrt(mse)
print(f"\nMSE = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")

# ---------- 6. Графики остатков ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 6.1 Остатки vs предсказанные значения
axes[0, 0].scatter(fitted, resid, edgecolors="k", alpha=0.7, s=40)
axes[0, 0].axhline(0, color="red", ls="--")
axes[0, 0].set_xlabel("Предсказанные значения (fitted)")
axes[0, 0].set_ylabel("Остатки (residuals)")
axes[0, 0].set_title("Остатки vs Предсказанные")

# 6.2 Q-Q plot
sm.qqplot(resid, line="s", ax=axes[0, 1])
axes[0, 1].set_title("Нормальный Q-Q график остатков")

# 6.3 Scale-Location
axes[1, 0].scatter(fitted, np.sqrt(np.abs(resid)),
                   edgecolors="k", alpha=0.7, s=40)
axes[1, 0].set_xlabel("Предсказанные значения (fitted)")
axes[1, 0].set_ylabel("sqrt(|Остатки|)")
axes[1, 0].set_title("Scale-Location")

# 6.4 Наблюдаемые vs Предсказанные
axes[1, 1].scatter(df["weight"], fitted, edgecolors="k", alpha=0.7, s=40)
mn = min(df["weight"].min(), fitted.min())
mx = max(df["weight"].max(), fitted.max())
axes[1, 1].plot([mn, mx], [mn, mx], "r--")
axes[1, 1].set_xlabel("Наблюдаемые значения (weight)")
axes[1, 1].set_ylabel("Предсказанные значения")
axes[1, 1].set_title("Наблюдаемые vs Предсказанные")

plt.tight_layout()
plt.show()

# ---------- 7. Дополнительно: простая регрессия для сравнения ----------
print("\n=== Простая регрессия: weight ~ length ===")
m_simple = smf.ols("weight ~ length", data=df).fit()
print(m_simple.summary().tables[1])
print(f"R² (простая) = {m_simple.rsquared:.4f}")
print(f"R² (множественная) = {model.rsquared:.4f}")

# ---------- 8. Итоговые выводы ----------
print("\n" + "=" * 60)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 60)
print(f"1. R² = {model.rsquared:.4f} — модель объясняет "
      f"{model.rsquared*100:.1f}% дисперсии веса.")
print(f"2. F-тест: p = {model.f_pvalue:.4g} < 0.05 — модель значима.")
print("3. Значимость коэффициентов (p < 0.05):")
for name in ["Intercept", "length", "width", "height"]:
    pval = model.pvalues[name]
    sign = "значим" if pval < 0.05 else "НЕ значим"
    print(f"   {name:10s}: p = {pval:.4g} -> {sign}")
print(f"4. Остатки: нормальность p = {p_res:.4g}; "
      f"автокорреляция p = {bg[1]:.4g}; "
      f"гетероскедастичность p = {bp[1]:.4g}")
print(f"5. MSE = {mse:.4f}, RMSE = {rmse:.4f}")