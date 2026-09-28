. # Конспект по лабораторной №3
. ## Описательные статистики (датасет NMES1988)

**Дисциплина:** Введение в анализ больших данных

---

. ## 0. О чём лаба

Реальный датасет **NMES1988** — медицинское обследование (4406 наблюдений).  
Анализируем 12 переменных: `visits`, `health`, `chronic`, `adl`, `region`, `age`, `gender`, `married`, `school`, `income`, `employed`, `insurance`.

**Задачи:**
1. Гистограмма + KDE для `visits`
2. Mean + SE
3. Медиана, квартили, `describe()`
4. Skewness и kurtosis
5. Группировка по полу
6. Boxplot
7. Медианы по двум факторам + barplot
8. Медиана `age` по `married` (общая и по полу)

---

. ## 1. Загрузка данных

```python
data = pd.read_csv("NMES1988.csv", index_col=0)   # убираем rownames
data = data[cols]                                 # оставляем нужные столбцы
```

- `index_col=0` — первая колонка `rownames` становится индексом.
- Отбираем только 12 переменных из задания.

---

. ## 2. Описательные статистики: что считаем

| Метрика | Формула / код | Что показывает |
|---------|---------------|----------------|
| Среднее | `s.mean()` | центр распределения |
| SD | `s.std(ddof=1)` | разброс (n−1) |
| **SE** | `sd / sqrt(n)` | точность оценки среднего |
| Медиана | `s.median()` | робастный центр |
| Квартили | `s.quantile([0.25, 0.5, 0.75])` | разброс + робастность |
| Skewness | `stats.skew(s)` | асимметрия |
| Kurtosis | `stats.kurtosis(s)` | «тяжесть» хвостов (excess) |

**SE** — стандартная ошибка среднего. Показывает, насколько выборочное среднее может «прыгать» вокруг истинного среднего.

**Skewness** > 0 → правый хвост длиннее (типично для `visits` — большинство людей ходят к врачу редко, но есть «частые посетители»).

**Kurtosis** > 0 → хвосты тяжелее нормального.

---

. ## 3. Гистограмма + KDE

```python
ax.hist(visits, bins=40, density=True, alpha=0.7)
kde = stats.gaussian_kde(visits.dropna())
xs = np.linspace(visits.min(), visits.max(), 300)
ax.plot(xs, kde(xs), color="crimson", lw=2)
```

- `density=True` — нормируем гистограмму, чтобы её можно было накладывать на плотность.
- `gaussian_kde` — ядерная оценка плотности по данным.

---

. ## 4. Группировка по категориям

```python
data.groupby("gender")["visits"].describe()
data.groupby(["gender", "region"])["visits"].median().unstack()
data.groupby(["gender", "married"])["age"].median().unstack()
```

- `groupby(["A", "B"])` — сводка по двум факторам.
- `.unstack()` — превращает «длинную» таблицу в матрицу (один фактор — строки, другой — столбцы).

---

. ## 5. Boxplot (диаграмма размаха)

```python
sns.boxplot(data=data, x="gender", y="visits")
```

- **Ящик** — от Q1 до Q3 (интерквартильный размах, IQR = Q3 − Q1).
- **Медиана** — линия внутри ящика.
- **Усы** — до крайних значений в пределах 1.5 × IQR от границ ящика.
- **Точки за усами** — потенциальные выбросы.

---

. ## 6. Barplot по медианам

```python
med = data.groupby("gender")["visits"].median()
med.plot(kind="bar", edgecolor="black")
```

Идея: сначала агрегируем (берём медиану), потом строим столбики.

---

. ## 7. Как читать результаты по NMES1988

- **visits** имеет сильно **правостороннее распределение** (skewness > 0): большинство людей ходят к врачу 0–5 раз, но есть редкие «частые посетители» с десятками визитов.
- **Среднее > медианы** — это следствие правого хвоста.
- **Kurtosis > 0** — тяжёлые хвосты, больше выбросов, чем у нормального распределения.
- **Медиана возраста** обычно выше у `married=yes` — женатые в среднем старше.

---

. ## 🧠 Шпаргалка

| Задача | Код |
|--------|-----|
| Загрузить CSV | `pd.read_csv(path, index_col=0)` |
| Среднее / sd | `s.mean()`, `s.std(ddof=1)` |
| SE среднего | `s.std(ddof=1) / np.sqrt(s.count())` |
| Медиана, квартили | `s.median()`, `s.quantile([0.25, 0.5, 0.75])` |
| Summary | `s.describe()` |
| Skew / kurtosis | `stats.skew(x, bias=False)`, `stats.kurtosis(x, bias=False)` |
| По группам | `df.groupby("A")["y"].mean()` |
| Два фактора | `df.groupby(["A","B"])["y"].median().unstack()` |
| Гистограмма + KDE | `plt.hist(..., density=True)` + `stats.gaussian_kde` |
| Boxplot | `sns.boxplot(data=df, x="A", y="y")` |
| Barplot | `series.plot(kind="bar")` |

---

. ## 📚 Что почитать

1. **Wes McKinney — «Python for Data Analysis»** — глава 5 (Pandas: `describe`, `groupby`), глава 7 (категориальные данные).
2. **Jake VanderPlas — «Python Data Science Handbook»**, глава 4 — статистика и распределения.
3. **Документация Pandas** — раздел «Group by: split-apply-combine».