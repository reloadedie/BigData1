# Задание 2. Вероятностные распределения
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки графиковs
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
sns.set_style("whitegrid")

np.random.seed(2024)


# ------------------------------------------------------------
# Пример 1. Стандартное нормальное N(0, 1), n = 10
# ------------------------------------------------------------
def example_1():
    print("=" * 60)
    print("Пример 1. Стандартное нормальное N(0, 1), n = 10")
    print("=" * 60)

    x1 = stats.norm.rvs(loc=0, scale=1, size=10)

    print("Выборка x1     :", np.round(x1, 4))
    print("Среднее        :", round(x1.mean(), 4))
    print("Ст. отклонение :", round(x1.std(ddof=1), 4))
    print()

    fig, ax = plt.subplots()
    ax.hist(x1, bins=5, density=True, color="steelblue",
            alpha=0.7, edgecolor="white", label="гистограмма")

    xs = np.linspace(-4, 4, 300)
    ax.plot(xs, stats.norm.pdf(xs, loc=0, scale=1),
            color="crimson", lw=2, label="плотность N(0, 1)")

    ax.set_title("Пример 1. Стандартное нормальное N(0, 1), n=10")
    ax.set_xlabel("x")
    ax.set_ylabel("плотность")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Пример 2. Два вектора N(1, 3) и N(-1, 2), n = 100
# ------------------------------------------------------------
def example_2():
    print("=" * 60)
    print("Пример 2. Нормальные векторы, n = 100")
    print("=" * 60)

    vec_a = stats.norm.rvs(loc=1, scale=3, size=100)   # N(1, 3)
    vec_b = stats.norm.rvs(loc=-1, scale=2, size=100)  # N(-1, 2)

    print("а) N(1, 3)  : mean =", round(vec_a.mean(), 3),
          " sd =", round(vec_a.std(ddof=1), 3))
    print("б) N(-1, 2) : mean =", round(vec_b.mean(), 3),
          " sd =", round(vec_b.std(ddof=1), 3))
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sample, mu, sigma, title in [
        (axes[0], vec_a, 1, 3, "а) N(mean=1, sd=3), n=100"),
        (axes[1], vec_b, -1, 2, "б) N(mean=-1, sd=2), n=100"),
    ]:
        ax.hist(sample, bins=15, density=True, color="mediumseagreen",
                alpha=0.7, edgecolor="white", label="гистограмма")

        xs = np.linspace(sample.min() - 2, sample.max() + 2, 300)
        ax.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma),
                color="crimson", lw=2, label="теоретическая плотность")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("плотность")
        ax.legend()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Пример 3. Равномерное распределение U[0, 1], n = 20
# ------------------------------------------------------------
def example_3():
    print("=" * 60)
    print("Пример 3. Равномерное U[0, 1], n = 20")
    print("=" * 60)

    u = stats.uniform.rvs(loc=0, scale=1, size=20)

    print("Выборка u      :", np.round(u, 4))
    print("min            :", round(u.min(), 4))
    print("max            :", round(u.max(), 4))
    print("Среднее        :", round(u.mean(), 4), " (теория: 0.5)")
    print()

    fig, ax = plt.subplots()
    ax.hist(u, bins=8, density=True, color="goldenrod",
            alpha=0.7, edgecolor="white", label="гистограмма")

    xs = np.linspace(-0.2, 1.2, 300)
    ax.plot(xs, stats.uniform.pdf(xs, loc=0, scale=1),
            color="darkred", lw=2, label="плотность U[0, 1]")

    ax.set_title("Пример 3. Равномерное U[0, 1], n=20")
    ax.set_xlabel("x")
    ax.set_ylabel("плотность")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Пример 4. Распределение Стьюдента, df = 1 и df = 5
# ------------------------------------------------------------
def example_4():
    print("=" * 60)
    print("Пример 4. Распределение Стьюдента, df = 1 и df = 5")
    print("=" * 60)

    # Сетка значений из задания: x = range(-5, 5, 0.01)
    x_grid = np.arange(-5, 5, 0.01)

    np.random.seed(42)
    t_df1 = stats.t.rvs(df=1, size=200)
    t_df5 = stats.t.rvs(df=5, size=200)

    print("t(df=1): mean =", round(t_df1.mean(), 3),
          " sd =", round(t_df1.std(ddof=1), 3))
    print("t(df=5): mean =", round(t_df5.mean(), 3),
          " sd =", round(t_df5.std(ddof=1), 3))
    print("Теоретические sd: df=1 ->", round(stats.t(df=1).std(), 3),
          "| df=5 ->", round(stats.t(df=5).std(), 3))
    print()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sample, df, title in [
        (axes[0], t_df1, 1, "t-распределение, df=1, n=200"),
        (axes[1], t_df5, 5, "t-распределение, df=5, n=200"),
    ]:
        ax.hist(sample, bins=30, density=True, color="mediumpurple",
                alpha=0.7, edgecolor="white", label="гистограмма")

        ax.plot(x_grid, stats.t.pdf(x_grid, df=df),
                color="crimson", lw=2, label=f"плотность t(df={df})")

        ax.set_xlim(-5, 5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("плотность")
        ax.legend()

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Дополнительно: сравнение плотностей t(df=1), t(df=5) и N(0,1)
# ------------------------------------------------------------
def extra_comparison():
    print("=" * 60)
    print("Дополнительно: сравнение t(df=1), t(df=5) и N(0, 1)")
    print("=" * 60)

    x_grid = np.arange(-5, 5, 0.01)

    plt.figure(figsize=(10, 5))
    plt.plot(x_grid, stats.t.pdf(x_grid, df=1), lw=2, label="t(df=1)")
    plt.plot(x_grid, stats.t.pdf(x_grid, df=5), lw=2, label="t(df=5)")
    plt.plot(x_grid, stats.norm.pdf(x_grid, 0, 1), "k--", lw=2,
             alpha=0.7, label="N(0, 1)")

    plt.title("Сравнение плотностей: t(df=1), t(df=5) и N(0, 1)")
    plt.xlabel("x")
    plt.ylabel("плотность")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------
if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4()
    extra_comparison()