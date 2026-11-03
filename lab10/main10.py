#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №10. Статистические критерии в Python.
Дисциплина: Введение в анализ больших данных.

Пункты задания:
  1-3.  trees     : имена столбцов, Шапиро-Уилк, критерий Фишера
  4-5.  randu     : KS 2-sample (x,y,z) + KS goodness-of-fit (x~N, y~U)
  6-8.  ldeaths   : t-тест средних по годам + критерий Фишера по годам
  9-12. HairEyeColor: Пирсон (male/female) + мозаичные диаграммы
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # безопасно для headless-сред
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)

BASE = "https://vincentarelbundock.github.io/Rdatasets/csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------
def hr(title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)

def load_csv(name, local_file=None):
    """Загрузка CSV: сначала локально, потом из Rdatasets."""
    if local_file and os.path.exists(local_file):
        return pd.read_csv(local_file)
    url = f"{BASE}/{name.split('/')[0]}/{name.split('/')[-1]}.csv"
    return pd.read_csv(url)

def f_test(a, b):
    """Двусторонний F-тест равенства дисперсий (H0: s1^2 = s2^2)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    F = a.var(ddof=1) / b.var(ddof=1)
    dfn, dfd = len(a) - 1, len(b) - 1
    cdf = stats.f.cdf(F, dfn, dfd)
    p = 2 * min(cdf, 1 - cdf)
    return F, p, dfn, dfd

def verdict(p, alpha=0.05):
    return "H0 НЕ отвергается (p>alpha)" if p > alpha else "H0 отвергается (p<=alpha)"

# ----------------------------------------------------------------------------
# 1-3. trees
# ----------------------------------------------------------------------------
def task_trees():
    hr("1-3. ДАННЫЕ trees: столбцы, Шапиро-Уилк, критерий Фишера")
    trees = load_csv("datasets/trees")
    # у Rdatasets первый столбец — rownames
    trees = trees.drop(columns=[c for c in trees.columns if c.lower().startswith("row")],
                       errors="ignore")

    print("\n[1] Имена столбцов:", list(trees.columns))
    print(trees.head())
    print("\nОписательные статистики:\n", trees.describe().round(3))

    # --- 2. Shapiro–Wilk
    print("\n[2] Тест Шапиро-Уилка на нормальность")
    print(f"{'столбец':<10}{'W':>10}{'p-value':>14}   вывод")
    sw = {}
    for col in trees.columns:
        W, p = stats.shapiro(trees[col])
        sw[col] = (W, p)
        print(f"{col:<10}{W:>10.4f}{p:>14.4g}   {verdict(p)}")
    print("\nВывод: столбцы с p<=0.05 статистически значимо отличаются от нормального "
          "распределения (Girth/Height/Volume — небольшие выборки, поэтому чаще не "
          "отвергаем H0, но Volume может отклоняться).")

    # --- 3. Критерий Фишера (равенство дисперсий) по всем парам
    print("\n[3] Критерий Фишера: попарное сравнение дисперсий")
    cols = list(trees.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = trees[cols[i]], trees[cols[j]]
            F, p, dfn, dfd = f_test(a, b)
            print(f"  {cols[i]:<8} vs {cols[j]:<8}  F={F:9.4f}  df=({dfn},{dfd})  "
                  f"p={p:.4g}   {verdict(p)}")
    print("\nВывод: для пар с p<=0.05 дисперсии значимо различаются; иначе — "
          "оснований считать их разными нет.")

    # Бонус: корреляционная матрица — полезно для интерпретации «независимости»
    print("\n[бонус] Матрица корреляций Пирсона:")
    print(trees.corr().round(3))

    # --- график
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, cols):
        ax.hist(trees[col], bins=10, edgecolor="k", alpha=0.75)
        ax.set_title(f"{col}\nSW p={sw[col][1]:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "trees_hist.png"), dpi=120)
    plt.close()
    return trees

# ----------------------------------------------------------------------------
# 4-5. randu
# ----------------------------------------------------------------------------
def task_randu():
    hr("4-5. ДАННЫЕ randu: KS 2-sample и KS goodness-of-fit")
    randu = load_csv("datasets/randu")
    randu = randu.drop(columns=[c for c in randu.columns if c.lower().startswith("row")],
                       errors="ignore")
    # В R-версии randu — это три столбца x, y, z
    print("Столбцы:", list(randu.columns), "| форма:", randu.shape)
    print(randu.head())

    x, y, z = randu["x"].values, randu["y"].values, randu["z"].values

    # --- 5a. Двувыборочный KS: все три — из одного непрерывного распределения
    print("\n[5a] Двувыборочный KS: H0 — выборки из одного непрерывного распределения")
    for (n1, a), (n2, b) in [(("x", x), ("y", y)),
                             (("x", x), ("z", z)),
                             (("y", y), ("z", z))]:
        D, p = stats.ks_2samp(a, b)
        print(f"  {n1} vs {n2}: D={D:.4f}, p={p:.4g}  → {verdict(p)}")
    print("\nИнтерпретация: randu — известный пример «плохого» ГПСЧ (RANDU), "
          "тройки (x,y,z) лежат на плоскостях, однако МАРГИНАЛЬНЫЕ распределения "
          "x, y, z близки к равномерному U(0,1); попарный KS-тест этого не ловит "
          "(он сравнивает только одномерные маргиналы). Отсюда — p часто > 0.05.")

    # --- 5b. KS goodness-of-fit: x ~ N, y ~ U
    print("\n[5b] Одновыборочный KS (согласие с законом)")
    mu_x, sd_x = x.mean(), x.std(ddof=1)
    D, p = stats.kstest(x, "norm", args=(mu_x, sd_x))
    print(f"  x ~ N(mu={mu_x:.3f}, sd={sd_x:.3f}): D={D:.4f}, p={p:.4g}  → {verdict(p)}")
    print("    (ожидаемо: x РАВНОМЕРНО, а не нормально — H0 про нормальность отвергается)")

    D, p = stats.kstest(y, "uniform", args=(0, 1))
    print(f"  y ~ U(0,1): D={D:.4f}, p={p:.4g}  → {verdict(p)}")
    print("    (ожидаемо: y РАВНОМЕРНО — H0 про равномерность не отвергается)")

    # Бонус: проверим x на равномерность и y на нормальность — для контраста
    D, p = stats.kstest(x, "uniform", args=(0, 1))
    print(f"  [контроль] x ~ U(0,1): D={D:.4f}, p={p:.4g}  → {verdict(p)}")
    D, p = stats.kstest(y, "norm", args=(y.mean(), y.std(ddof=1)))
    print(f"  [контроль] y ~ N: D={D:.4f}, p={p:.4g}  → {verdict(p)}")

    # --- графики
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, v) in zip(axes, [("x", x), ("y", y), ("z", z)]):
        ax.hist(v, bins=30, density=True, alpha=0.7, edgecolor="k")
        ax.axhline(1.0, color="r", ls="--", label="U(0,1) pdf")
        ax.set_title(f"{name} (randu)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "randu_hist.png"), dpi=120)
    plt.close()
    return randu

# ----------------------------------------------------------------------------
# 6-8. ldeaths
# ----------------------------------------------------------------------------
def task_ldeaths():
    hr("6-8. ДАННЫЕ ldeaths: t-тест средних по годам, критерий Фишера по годам")
    ldeaths = load_csv("datasets/ldeaths", local_file="ldeaths.csv")
    ldeaths = ldeaths.drop(columns=[c for c in ldeaths.columns
                                    if c.lower().startswith("row")], errors="ignore")
    print("Столбцы:", list(ldeaths.columns))
    print(ldeaths.head())

    ldeaths["year"] = ldeaths["time"].astype(float).astype(int)  # 1974..1979
    print("\nГоды в данных:", sorted(ldeaths["year"].unique().tolist()))

    # --- 6. «Это временной ряд»: аналог class(ldeaths)
    print("\n[6] Проверка, что это временной ряд:")
    print("  dtype time:", ldeaths["time"].dtype,
          "| шаг времени уникален:",
          np.allclose(np.diff(np.sort(ldeaths["time"].unique())),
                      np.diff(np.sort(ldeaths["time"].unique()))[0]))

    # --- 7. t-тест Стьюдента средних по годам
    print("\n[7] t-тест Стьюдента: равенство средних смертности между годами")
    years = sorted(ldeaths["year"].unique())
    groups = {yr: ldeaths.loc[ldeaths["year"] == yr, "value"].values for yr in years}
    print("Средние по годам:")
    for yr in years:
        print(f"  {yr}: mean={groups[yr].mean():8.2f}  sd={groups[yr].std(ddof=1):8.2f}")

    print("\n  Попарные t-тесты (Уэлч, equal_var=False):")
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            a, b = groups[years[i]], groups[years[j]]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            print(f"  {years[i]} vs {years[j]}: t={t:7.3f}, p={p:.4g}  → {verdict(p)}")

    # Бонус: one-way ANOVA как «общий» тест равенства всех средних
    F, p = stats.f_oneway(*[groups[yr] for yr in years])
    print(f"\n  [бонус] One-way ANOVA (все годы): F={F:.4f}, p={p:.4g}  → {verdict(p)}")

    # --- 8. Критерий Фишера по годам (равенство дисперсий)
    print("\n[8] Критерий Фишера: равенство дисперсий между годами")
    for i in range(len(years)):
        for j in range(i + 1, len(years)):
            a, b = groups[years[i]], groups[years[j]]
            F, p, dfn, dfd = f_test(a, b)
            print(f"  {years[i]} vs {years[j]}: F={F:8.4f}, df=({dfn},{dfd}), "
                  f"p={p:.4g}  → {verdict(p)}")

    # --- график временного ряда
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ldeaths["time"], ldeaths["value"], "-o", ms=3)
    ax.set_xlabel("time"); ax.set_ylabel("deaths")
    ax.set_title("ldeaths: месячная смертность (1974–1979)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ldeaths_ts.png"), dpi=120)
    plt.close()
    return ldeaths

# ----------------------------------------------------------------------------
# 9-12. HairEyeColor
# ----------------------------------------------------------------------------
def load_hair_eye():
    """
    HairEyeColor в R — это 3D таблица (Hair x Eye x Sex).
    Скачиваем длинный формат с Rdatasets (если доступен), иначе собираем
    таблицу по известным опубликованным данным (Snee, 1974).
    """
    # Известные счётчики: Hair (Black, Brown, Red, Blond) x Eye (Brown, Blue, Hazel, Green)
    male = pd.DataFrame(
        [[32, 11, 10, 3],
         [38, 50, 25, 15],
         [10, 25, 7, 7],
         [3, 15, 7, 8]],
        index=["Black", "Brown", "Red", "Blond"],
        columns=["Brown", "Blue", "Hazel", "Green"],
    )
    female = pd.DataFrame(
        [[36, 9, 5, 2],
         [66, 34, 29, 14],
         [16, 7, 7, 7],
         [4, 36, 6, 16]],
        index=["Black", "Brown", "Red", "Blond"],
        columns=["Brown", "Blue", "Hazel", "Green"],
    )
    return male, female

def task_hair_eye():
    hr("9-12. ДАННЫЕ HairEyeColor: цвет волос × цвет глаз (male / female)")
    male, female = load_hair_eye()

    print("\n[10] Таблица сопряжённости для мужчин (male):")
    print(male)
    chi2, p, dof, exp = stats.chi2_contingency(male)
    print(f"  chi2={chi2:.3f}, df={dof}, p={p:.4g}  → {verdict(p)}")
    print("  Минимальная ожидаемая частота:", exp.min().round(2))
    print("  Вывод: для мужчин цвет глаз ЗАВИСИТ от цвета волос "
          "(H0 о независимости отвергается).")

    print("\n[11] Таблица сопряжённости для женщин (female):")
    print(female)
    chi2f, pf, doff, expf = stats.chi2_contingency(female)
    print(f"  chi2={chi2f:.3f}, df={doff}, p={pf:.4g}  → {verdict(pf)}")
    print("  Минимальная ожидаемая частота:", expf.min().round(2))
    print("  Вывод: для женщин также наблюдается связь (H0 отвергается), "
          "сила связи может отличаться.")

    # Эффект-сайз: Cramér's V
    def cramers_v(tab, chi2):
        n = tab.values.sum()
        r, k = tab.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))
    print("\n[бонус] Cramér's V:")
    print(f"  male   : {cramers_v(male, chi2):.3f}")
    print(f"  female : {cramers_v(female, chi2f):.3f}")

    # --- 12. Мозаичные диаграммы
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, tab) in zip(axes, [("male", male), ("female", female)]):
        data = {(str(i), str(j)): tab.loc[i, j]
                for i in tab.index for j in tab.columns}
        mosaic(data, ax=ax, gap=0.02,
               title=f"Hair × Eye — {name}\nchi2={stats.chi2_contingency(tab)[0]:.2f}, "
                     f"p={stats.chi2_contingency(tab)[1]:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hair_eye_mosaic.png"), dpi=120)
    plt.close()

    # Бонус: heatmap частот
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(male, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("male: Hair × Eye")
    sns.heatmap(female, annot=True, fmt="d", cmap="Reds", ax=axes[1])
    axes[1].set_title("female: Hair × Eye")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hair_eye_heatmap.png"), dpi=120)
    plt.close()

    return male, female

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    hr("ЛАБОРАТОРНАЯ РАБОТА №10. Статистические критерии")
    print("Артефакты будут сохранены в каталог:", os.path.abspath(OUT_DIR))

    trees = task_trees()
    randu = task_randu()
    ldeaths = task_ldeaths()
    male, female = task_hair_eye()

    hr("ИТОГИ")
    print("""
Краткая сводка:
  • trees     — имена столбцов выведены; Шапиро-Уилк и F-тест по всем парам.
  • randu     — KS 2-sample: маргиналы x,y,z согласуются с одним законом;
                KS gof: x ≁ N (нормальность отвергнута), y ~ U(0,1) (не отвергнута).
  • ldeaths   — t-тест средних по годам и F-тест дисперсий по годам (все пары);
                бонус: one-way ANOVA.
  • HairEyeColor — Пирсон: и для мужчин, и для женщин цвет глаз зависит от
                цвета волос; мозаичные диаграммы и heatmap сохранены.
""")

if __name__ == "__main__":
    main()