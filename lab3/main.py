# Описательные статистики в Python

## 0. Импорт и загрузка данных

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    print("Доступные файлы в рабочей директории:")
    print(os.listdir())

    file_name = "NMES1988.csv"

    if file_name.endswith('.csv'):
        data = pd.read_csv(file_name, index_col=0)   # index_col=0 убирает rownames
    elif file_name.endswith('.xlsx'):
        data = pd.read_excel(file_name, index_col=0)
    else:
        raise ValueError("Формат файла не поддерживается. Используйте CSV или Excel.")

    print(f"\nФайл '{file_name}' успешно загружен!")
    data.to_excel('data.xlsx')

    return data


data = load_data()
print("Размерность:", data.shape)
print("Столбцы:", data.columns.tolist())
data.head()

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

tips.describe(include="all")

## 1. Гистограмма числовой переменной + кривая плотности

fig, ax = plt.subplots()
ax.hist(tips["total_bill"], bins=25, density=True, color="lightblue",
        edgecolor="black", alpha=0.7, label="гистограмма")

# KDE по данным
kde = stats.gaussian_kde(tips["total_bill"].dropna())
xs = np.linspace(tips["total_bill"].min(), tips["total_bill"].max(), 300)
ax.plot(xs, kde(xs), color="crimson", lw=2, label="KDE")

ax.set_xlabel("total_bill")
ax.set_ylabel("плотность")
ax.set_title("Гистограмма total_bill + KDE")
ax.legend()
plt.tight_layout()
plt.show()

# То же через seaborn
plt.figure()
sns.histplot(tips["total_bill"], bins=25, kde=True, color="steelblue",
             edgecolor="black", alpha=0.7)
plt.xlabel("total_bill")
plt.title("Гистограмма + KDE (seaborn)")
plt.tight_layout()
plt.show()

## 2. Среднее и стандартная ошибка среднего (SE)

x = tips["total_bill"]
n = x.count()
mean_x = x.mean()
sd_x = x.std(ddof=1)
se_x = sd_x / np.sqrt(n)

print(f"n     = {n}")
print(f"mean  = {mean_x:.4f}")
print(f"sd    = {sd_x:.4f}")
print(f"SE    = {se_x:.4f}")

## 3. Медиана, квантили, summary


print("Медиана:", x.median())
print("Квантили 0%, 25%, 50%, 75%, 100%:")
print(x.quantile([0, 0.25, 0.5, 0.75, 1.0]))
print()
print(x.describe())

## 4. Асимметрия (skewness) и эксцесс (kurtosis)

sk = stats.skew(x, bias=False)
ku = stats.kurtosis(x, bias=False)
print(f"skewness = {sk:.4f}")
print(f"kurtosis (excess) = {ku:.4f}")

## 5. Описательные статистики **по группам**

# Сводка total_bill по полу клиента
print(tips.groupby("sex")["total_bill"].describe().round(2))

# Несколько статистик сразу
tips.groupby("day")["tip"].agg(["count", "mean", "median", "std"]).round(3)

## 6. Boxplot по категориям

plt.figure(figsize=(8, 5))
sns.boxplot(data=tips, x="day", y="total_bill", hue="sex")
plt.title("total_bill по дням недели и полу")
plt.tight_layout()
plt.show()

## 7. Медианы в разрезе двух факторов

med_two = tips.groupby(["day", "sex"])["tip"].median().unstack()
print(med_two.round(2))

med_two.plot(kind="bar", figsize=(9, 5), edgecolor="black")
plt.ylabel("Медиана tip")
plt.xlabel("День")
plt.title("Медианные чаевые: день × пол")
plt.xticks(rotation=0)
plt.legend(title="sex")
plt.tight_layout()
plt.show()

## 8. Barplot для одной группировки

med_by_time = tips.groupby("time")["total_bill"].median().sort_values(ascending=False)
print(med_by_time)

plt.figure(figsize=(6, 4))
med_by_time.plot(kind="bar", color="teal", edgecolor="black")
plt.ylabel("Медиана total_bill")
plt.title("Медианный счёт: обед vs ужин")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

## 9. Ещё один пример группировки: размер компании

# Вся выборка
med_size = tips.groupby("size")["tip"].median()
print("Медиана tip по size:")
print(med_size)

# По времени дня
med_size_time = tips.groupby(["time", "size"])["tip"].median().unstack()
print("\nМедиана tip: time × size")
print(med_size_time.round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

med_size.plot(kind="bar", ax=axes[0], color="coral", edgecolor="black")
axes[0].set_title("Медиана tip по size (вся выборка)")
axes[0].set_ylabel("Медиана tip")
axes[0].set_xlabel("size")

med_size_time.T.plot(kind="bar", ax=axes[1], edgecolor="black")
axes[1].set_title("Медиана tip: size × time")
axes[1].set_ylabel("Медиана tip")
axes[1].set_xlabel("size")
axes[1].legend(title="time")

for ax in axes:
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
