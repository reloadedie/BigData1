# Лабораторная работа №3: Описательные статистики

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")


FILE_NAME = "NMES1988.csv"

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"Файл '{FILE_NAME}' не найден в {os.getcwd()}. "
        f"Положите файл рядом со скриптом."
    )

# В файле первый столбец — rownames, его убираем через index_col=0
data = pd.read_csv(FILE_NAME, index_col=0)

print("Размерность исходного датасета:", data.shape)
print("Столбцы:", data.columns.tolist())
print(data.head())


# ПУНКТ 2. Открытие таблицы и отбор нужных переменных
needed_cols = ["visits", "health", "chronic", "adl", "region",
               "age", "gender", "married", "school", "income",
               "employed", "insurance"]

available = [c for c in needed_cols if c in data.columns]
missing = [c for c in needed_cols if c not in data.columns]

print("Доступные столбцы:", available)
print("Отсутствующие столбцы:", missing if missing else "нет")

df = data[available].copy()
print("\nРазмерность после отбора:", df.shape)
print(df.head())

# Общая информация и пропуски
df.info()
print("\nПропуски:\n", df.isnull().sum())

# Общая сводка
print("\nОписательные статистики (include='all'):")
print(df.describe(include="all"))


# ПУНКТ 3. Гистограмма visits с кривой плотности
fig, ax = plt.subplots()
ax.hist(df["visits"], bins=30, density=True, color="lightblue",
        edgecolor="black", alpha=0.7, label="гистограмма")

kde = stats.gaussian_kde(df["visits"].dropna())
xs = np.linspace(df["visits"].min(), df["visits"].max(), 300)
ax.plot(xs, kde(xs), color="crimson", lw=2, label="KDE")

ax.set_xlabel("visits (число посещений)")
ax.set_ylabel("плотность")
ax.set_title("Гистограмма visits + KDE")
ax.legend()
plt.tight_layout()
plt.show()

# То же через seaborn
plt.figure()
sns.histplot(df["visits"], bins=30, kde=True, color="steelblue",
             edgecolor="black", alpha=0.7)
plt.xlabel("visits")
plt.title("Гистограмма + KDE (seaborn)")
plt.tight_layout()
plt.show()


# ПУНКТ 4. Среднее значение и стандартная ошибка (SE) для visits
x = df["visits"]
n = x.count()
mean_x = x.mean()
sd_x = x.std(ddof=1)
se_x = sd_x / np.sqrt(n)

print(f"n     = {n}")
print(f"mean  = {mean_x:.4f}")
print(f"sd    = {sd_x:.4f}")
print(f"SE    = {se_x:.4f}")


# ПУНКТ 5. Медиана, квартили, summary для visits
print("Медиана:", x.median())
print("\nКвантили 0%, 25%, 50%, 75%, 100%:")
print(x.quantile([0, 0.25, 0.5, 0.75, 1.0]))
print()
print(x.describe())


# ПУНКТ 6. Эксцесс и асимметрия для visits
sk = stats.skew(x, bias=False)
ku = stats.kurtosis(x, bias=False)
print(f"skewness = {sk:.4f}")
print(f"kurtosis (excess) = {ku:.4f}")


# ПУНКТ 7. Медианы и квартили количественных данных по полу
quant_vars = ["visits", "chronic", "age", "school", "income"]

print("Медианы по полу:")
print(df.groupby("gender")[quant_vars].median().round(3))

print("\nКвартили visits по полу:")
print(df.groupby("gender")["visits"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .round(3))

# Полная сводка по каждому количественному показателю
for var in quant_vars:
    print(f"\n--- {var} ---")
    print(df.groupby("gender")[var].describe().round(3))


# ПУНКТ 8. Диаграмма размаха (boxplot) visits по полу
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="gender", y="visits", palette="Set2")
plt.title("Количество посещений врача по полу")
plt.xlabel("Пол")
plt.ylabel("visits")
plt.tight_layout()
plt.show()

# Вариант с изменённым параметром range (whis)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="gender", y="visits", ax=axes[0], palette="Set2")
axes[0].set_title("Boxplot (range=1.5, по умолчанию)")
axes[0].set_ylabel("visits")

sns.boxplot(data=df, x="gender", y="visits", ax=axes[1],
            palette="Set2", whis=3.0)
axes[1].set_title("Boxplot (range=3.0)")
axes[1].set_ylabel("visits")
plt.tight_layout()
plt.show()


# ПУНКТ 9. Медианы visits по gender и region одновременно
med_two = df.groupby(["region", "gender"])["visits"].median().unstack()
print("Медианы visits: region × gender")
print(med_two.round(2))

print("\nПолная сводка visits по region и gender:")
print(df.groupby(["region", "gender"])["visits"]
        .agg(["count", "mean", "median", "std"])
        .round(2))


# ПУНКТ 10. Столбиковые диаграммы медиан visits по gender и region
med_two.plot(kind="bar", figsize=(9, 5), edgecolor="black",
             color=["#66c2a5", "#fc8d62"])
plt.ylabel("Медиана visits")
plt.xlabel("Регион")
plt.title("Медианное число посещений: регион × пол")
plt.xticks(rotation=0)
plt.legend(title="Пол")
plt.tight_layout()
plt.show()

# Альтернатива через seaborn
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="region", y="visits", hue="gender",
            estimator=np.median, errorbar=None,
            palette="Set2", edgecolor="black")
plt.title("Медианное число посещений: регион × пол (seaborn)")
plt.ylabel("Медиана visits")
plt.xlabel("Регион")
plt.tight_layout()
plt.show()


# ПУНКТ 11. Медианный возраст по married: вся выборка и по полу
med_age_married = df.groupby("married")["age"].median()
print("Медианный возраст по married (вся выборка):")
print(med_age_married.round(2))

med_age_two = df.groupby(["married", "gender"])["age"].median().unstack()
print("\nМедианный возраст: married × gender")
print(med_age_two.round(2))

# Barplot: вся выборка
plt.figure(figsize=(7, 5))
med_age_married.plot(kind="bar", color="teal", edgecolor="black")
plt.ylabel("Медианный возраст (age)")
plt.xlabel("married")
plt.title("Медианный возраст по семейному положению")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Barplot: married × gender
med_age_two.plot(kind="bar", figsize=(9, 5), edgecolor="black",
                 color=["#66c2a5", "#fc8d62"])
plt.ylabel("Медианный возраст (age)")
plt.xlabel("married")
plt.title("Медианный возраст: married × gender")
plt.xticks(rotation=0)
plt.legend(title="Пол")
plt.tight_layout()
plt.show()

# Альтернатива через seaborn
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="married", y="age", hue="gender",
            estimator=np.median, errorbar=None,
            palette="Set2", edgecolor="black")
plt.title("Медианный возраст: married × gender (seaborn)")
plt.ylabel("Медианный возраст (age)")
plt.xlabel("married")
plt.tight_layout()
plt.show()