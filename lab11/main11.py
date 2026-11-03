# -*- coding: utf-8 -*-
"""
Дисперсионный анализ (ANOVA / MANOVA) — лабораторное задание.
Дисциплина: Введение в анализ больших данных

Задание 1: двухфакторный ANOVA по файлу DA.csv (ВесИглокожих)
    weight ~ priming * view
    weight ~ priming * region
    weight ~ priming * conditions
Задание 2: MANOVA по файлу university2.csv
    University ~ Yslov_educ + Yslov_job + Yslov_science
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # чтобы работало без GUI
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def find_col(df, candidates):
    """Возвращает первое найденное имя колонки из списка кандидатов."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Не найдена ни одна из колонок: {candidates}. "
                   f"Есть: {list(df.columns)}")


def prep_categorical(df, cols):
    """Приводит указанные колонки к категориальному типу (копия DataFrame)."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("category")
    return df


def save_table(df_table, name):
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df_table.to_csv(path, encoding="utf-8-sig")
    print(f"  [сохранено] {path}")


def interaction_plot(df, x, hue, y, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=axes[0])
    axes[0].set_title(f"Boxplot: {y} по {x} и {hue}")
    sns.pointplot(data=df, x=x, y=y, hue=hue, errorbar="se", ax=axes[1])
    axes[1].set_title("Interaction plot (средние ± SE)")
    plt.suptitle(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [сохранено] {path}")


def describe_anova(table, factors):
    """Печатает краткую интерпретацию таблицы ANOVA (typ=2)."""
    print("  Интерпретация (p < 0.05 → эффект значим):")
    for f in factors:
        row = None
        for idx in table.index:
            if f.lower() in str(idx).lower():
                row = table.loc[idx]
                break
        if row is None:
            continue
        p = row.get("PR(>F)", np.nan)
        F = row.get("F", np.nan)
        verdict = "значим" if (pd.notna(p) and p < 0.05) else "НЕ значим"
        print(f"    - {idx}: F = {F:.3f}, p = {p:.4g} → {verdict}")


# ---------------------------------------------------------------------------
# ЗАДАНИЕ 1. Двухфакторный ANOVA — DA.csv
# ---------------------------------------------------------------------------
def task1_anova():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 1. Двухфакторный ANOVA — DA.csv (ВесИглокожих)")
    print("=" * 70)

    if not os.path.exists("DA.csv"):
        print("  [!] Файл DA.csv не найден. Пропуск задания 1.")
        return

    df = pd.read_csv("DA.csv")
    # На случай нестандартных имён — ищем по кандидатам (в т.ч. русские)
    weight = find_col(df, ["weight", "Вес", "вес"])
    priming = find_col(df, ["priming", "Грунт", "грунт"])
    view = find_col(df, ["view", "Вид", "вид"])
    region = find_col(df, ["region", "Район", "район"])
    conditions = find_col(df, ["conditions", "Условия", "условия"])

    # Переименуем к латинским именам для удобства формул
    df = df.rename(columns={weight: "weight", priming: "priming",
                            view: "view", region: "region",
                            conditions: "conditions"})
    df = prep_categorical(df, ["priming", "view", "region", "conditions"])

    print("\nПервые строки датасета:")
    print(df.head())
    print("\nРазмер:", df.shape)

    analyses = [
        ("priming * view",       "Грунт × Вид",       "anova_priming_view"),
        ("priming * region",     "Грунт × Район",     "anova_priming_region"),
        ("priming * conditions", "Грунт × Условия",   "anova_priming_conditions"),
    ]

    for rhs, label, fname in analyses:
        print("\n" + "-" * 70)
        print(f"ANOVA: weight ~ {rhs}   ({label})")
        print("-" * 70)

        # Средние по группам
        factors = [c.strip() for c in rhs.split("*")]
        agg = df.groupby(factors, observed=True)["weight"].agg(
            ["mean", "std", "count"]).round(2)
        print("\nСредние по группам:")
        print(agg)

        # Модель и таблица ANOVA
        model = smf.ols(f"weight ~ {rhs}", data=df).fit()
        table = anova_lm(model, typ=2)
        print("\nТаблица ANOVA (typ=2):")
        print(table)

        describe_anova(table, factors)
        save_table(table, fname)
        interaction_plot(df, factors[0], factors[1], "weight",
                         f"weight ~ {rhs}", f"{fname}.png")


# ---------------------------------------------------------------------------
# ЗАДАНИЕ 2. MANOVA — university2.csv
# ---------------------------------------------------------------------------
def task2_manova():
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2. MANOVA — university2.csv")
    print("=" * 70)

    if not os.path.exists("university2.csv"):
        print("  [!] Файл university2.csv не найден. Пропуск задания 2.")
        return

    df = pd.read_csv("university2.csv")
    # Ищем кандидатов, включая русские названия
    uni = find_col(df, ["University", "Университет", "университет"])
    y1 = find_col(df, ["Yslov_educ", "Yslov_educ_rank",
                       "условия для получения качественного образования, ранг"])
    y2 = find_col(df, ["Yslov_job", "Yslov_job_rank",
                       "уровень востребованности выпускников работодателями, ранг"])
    y3 = find_col(df, ["Yslov_science", "Yslov_science_rank",
                       "уровень научно-исследовательской деятельности, ранг"])

    df = df.rename(columns={uni: "University", y1: "Yslov_educ",
                            y2: "Yslov_job", y3: "Yslov_science"})
    df["University"] = df["University"].astype("category")

    print("\nПервые строки датасета:")
    print(df.head())
    print("\nРазмер:", df.shape)

    ys = ["Yslov_educ", "Yslov_job", "Yslov_science"]

    print("\nСредние ранги по университетам:")
    means = df.groupby("University", observed=True)[ys].mean().round(2)
    print(means)
    save_table(means, "manova_group_means")

    # --- MANOVA ---
    formula = "Yslov_educ + Yslov_job + Yslov_science ~ C(University)"
    maov = MANOVA.from_formula(formula, data=df)
    mv = maov.mv_test()
    print("\nMANOVA (mv_test):")
    print(mv)

    # Сохраняем таблицу MANOVA (для C(University))
    try:
        mv_tab = mv.results["C(University)"]["stat"]
        save_table(mv_tab, "manova_results")
    except Exception as e:
        print(f"  [warn] не удалось сохранить MANOVA-таблицу: {e}")

    # --- Одномерные ANOVA после значимого MANOVA ---
    print("\n" + "-" * 70)
    print("Одномерные ANOVA по каждому отклику (post-hoc)")
    print("-" * 70)
    uni_anova = {}
    for y in ys:
        m = smf.ols(f"{y} ~ C(University)", data=df).fit()
        t = anova_lm(m, typ=2)
        print(f"\n{y}:")
        print(t)
        uni_anova[y] = t
        save_table(t, f"anova_{y}")

    # --- Графики ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, y in zip(axes, ys):
        sns.boxplot(data=df, x="University", y=y, ax=ax)
        ax.set_title(y)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "manova_boxplots.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\n  [сохранено] {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    task1_anova()
    task2_manova()
    print("\nГотово. Результаты в папке 'output/'.")