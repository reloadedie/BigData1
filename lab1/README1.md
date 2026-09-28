. # Конспект по лабораторной №1
. ## Создание объектов для хранения данных в Python

**Дисциплина:** Введение в анализ больших данных
**Тема:** Основные структуры данных для анализа данных в Python

---

. ## 0. Зачем это нужно

В R есть свои встроенные типы: `vector`, `data.frame`, `matrix`, `array`, `list`, `factor`.
В Python аналогов «из коробки» нет — их заменяют библиотеки:

| R                | Python                               | Библиотека |
|------------------|--------------------------------------|------------|
| `vector`         | `list`, `numpy.ndarray`, `pd.Series` | встроенный / NumPy / Pandas |
| `data.frame`     | `pandas.DataFrame`                   | Pandas     |
| `matrix`         | `numpy.ndarray` (ndim=2)             | NumPy      |
| `array`          | `numpy.ndarray` (ndim=N)             | NumPy      |
| `list`           | `list`, `dict`                       | встроенный |
| `factor`         | `pandas.Categorical`                 | Pandas     |

**Две главные библиотеки:**
- **NumPy** — быстрые числовые массивы и матрицы.
- **Pandas** — таблицы (`DataFrame`) и одномерные структуры (`Series`).

Импорт:
```python
import numpy as np
import pandas as pd
```

---

. ## 1. Вектор

. ### Способы создания
```python
. # 1) Обычный список Python
S1_list = list(range(1, 11))         # [1, 2, ..., 10]

. # 2) NumPy-массив
S1 = np.arange(1, 11)                # array([1, 2, ..., 10])

. # 3) Pandas Series
S1_series = pd.Series(range(1, 11))
```

. ### `arange` vs `linspace`
```python
np.arange(start, stop, step)   # stop НЕ включается
np.linspace(start, stop, num)  # stop ВКЛЮЧАЕТСЯ, задаём число точек
```

Пример — от 1 до 10 с шагом 0.5 (19 значений):
```python
S2 = np.arange(1, 10.5, 0.5)     # пишем 10.5, чтобы 10.0 попало в диапазон
S2 = np.linspace(1, 10, 19)      # эквивалент
```

. ### Объединение векторов
```python
S = np.concatenate([S1, S2])     # аналог R: c(S1, S2)
```

. ### Индексация (важно!)
В Python индексация **с нуля**.
```python
S[0]        # первый элемент
S[[2,3,4]]  # 3-й, 4-й, 5-й (fancy indexing)
S[[0,3]]    # 1-й и 4-й
S[0:3]      # срез: элементы с индексами 0,1,2 (3 не включается)
```

. ### Полезные методы
- `len(arr)` / `arr.shape` — длина / размерность
- `arr.dtype` — тип данных
- `arr.min()`, `arr.max()`, `arr.mean()`, `arr.sum()`

---

. ## 2. Таблица (DataFrame)

. ### Создание из словаря
```python
CITY = pd.DataFrame({
    "City":   Cities,
    "Sex":    Sex,
    "Number": Number
})
```

. ### Просмотр структуры
| Метод / атрибут      | Что делает                                |
|----------------------|-------------------------------------------|
| `df.head(n)`         | первые n строк                            |
| `df.tail(n)`         | последние n строк                         |
| `df.info()`          | типы, пропуски, объём памяти              |
| `df.describe()`      | описательная статистика                   |
| `df.shape`           | (строки, столбцы)                         |
| `df.columns`         | имена столбцов                            |
| `df.dtypes`          | типы столбцов                             |

. ### Доступ к данным — `.loc` vs `.iloc`
```python
df.iloc[0:3]          # по позиции (правая граница не включается)
df.loc[0:2]           # по метке  (правая граница ВКЛЮЧАЕТСЯ)
df["Number"]          # столбец -> Series
df[["City","Number"]] # несколько столбцов -> DataFrame
```

. ### Фильтрация строк (булева маска)
```python
CITY[CITY["Number"] > 10000]                 # все столбцы, где условие True
CITY.loc[CITY["Number"] > 10000, "Number"]   # только столбец Number
CITY[CITY["Sex"] == "Male"]                  # по категории
```

. ### Извлечение среза из столбца
```python
CITY["Number"].iloc[0:3]   # элементы 1–3 (индексы 0,1,2)
```

---

. ## 3. Матрица

В Python матрица — это `numpy.ndarray` с `ndim == 2`.

. ### Создание
```python
m1 = np.arange(1, 17).reshape(2, 8)   # 2×8 (16 элементов)
m3 = np.arange(1, 25).reshape(4, 6)   # 4×6 (24 элемента)
```
⚠️ `reshape(rows, cols)` требует, чтобы **произведение совпадало** с числом элементов.

. ### Транспонирование
```python
m3_T = m3.T     # 6×4
```

. ### Умножение
```python
A @ B        # матричное умножение (оператор @)
np.dot(A,B)  # то же самое
A * B        # поэлементное умножение
```

. ### Определитель (только для квадратных!)
```python
square = m3 @ m3_T       # 4×4
det = np.linalg.det(square)
```

. ### Ещё полезное
- `np.linalg.inv(A)` — обратная матрица
- `np.linalg.eig(A)` — собственные значения и векторы
- `A.ravel()` / `A.flatten()` — «развернуть» в одномерный массив

---

. ## 4. Массив (многомерный)

В R `array()` — массив произвольной размерности. В Python — тот же `ndarray`.

```python
M = np.array(range(1, 25)).reshape(3, 2, 4)  # 3×2×4 = 24 элемента
```

. ### Проверка, что это массив
R: `is.array(x)`
Python:
```python
isinstance(M, np.ndarray)   # True
```
Атрибуты:
- `M.ndim`  — число измерений
- `M.shape` — форма
- `M.size`  — общее число элементов

---

. ## 5. Список

В R `list` может хранить элементы **разных типов и длин**.
В Python есть два близких инструмента:
- `list` — упорядоченная коллекция (доступ по индексу).
- `dict` — «именованный список» (доступ по ключу).

```python
integers = list(range(1, 7))       # [1..6]
booleans = [True, False] * 3       # [T,F,T,F,T,F]

. # Список списков
L1 = [integers, booleans]

. # Именованный (аналог R-списка с names)
L1 = {
    "integers": integers,
    "booleans": booleans
}
```

---

. ## 6. Фактор (категориальная переменная)

В R — `factor`, в Pandas — `Categorical`.

```python
f1 = pd.Categorical(
    ["Yes", "No", "Parhaps", "Yes", "No", "Parhaps"],
    categories=["Yes", "No", "Parhaps"]
)
```

- `f1.categories` — уровни (levels в R)
- `f1.codes` — числовые коды уровней
- Для столбца в DataFrame: `df["col"].astype("category")`

**Зачем нужно:**
- экономия памяти,
- явное задание порядка (`ordered=True`),
- корректная работа статистических и ML-методов.

---

. ## 7. Фрейм данных (DataFrame)

`data.frame` в R ↔ `pandas.DataFrame` в Python.

Может хранить **разнотипные столбцы** (числа, строки, bool, категории):

```python
b = ["a", "b", "c", "a"]
d = [(i % 2 == 0) for i in range(1, 5)]     # [False, True, False, True]
e = pd.Categorical(["soft","hard","soft","medium"],
                   categories=["soft","hard","medium"])

F1 = pd.DataFrame({"b": b, "d": d, "e": e})
```

`F1.dtypes` покажет, что столбцы имеют разные типы, а `e` — категориальный.

---

. ## 🧠 Шпаргалка «одной строкой»

| Задача                        | Python                                |
|-------------------------------|---------------------------------------|
| Вектор 1..10                  | `np.arange(1, 11)`                    |
| Вектор с шагом 0.5            | `np.arange(1, 10.5, 0.5)`             |
| Объединить векторы            | `np.concatenate([a, b])`              |
| Взять 3-й, 4-й, 5-й элемент   | `S[[2, 3, 4]]`                        |
| Таблица из векторов           | `pd.DataFrame({...})`                 |
| Первые n строк                | `df.head(n)`                          |
| Фильтр по условию             | `df[df["col"] > value]`               |
| Матрица 4×6                   | `np.arange(1,25).reshape(4,6)`        |
| Транспонирование              | `A.T`                                 |
| Определитель                  | `np.linalg.det(A)`                    |
| Проверка на массив            | `isinstance(x, np.ndarray)`           |
| Именованный список            | `{"key": value}`                      |
| Категориальный вектор (фактор)| `pd.Categorical(vals, categories=[…])`|

---

. ## 📚 Что почитать

1. **Jake VanderPlas — «Python Data Science Handbook»**
   - Глава 2 — NumPy (`arange`, `linspace`, `reshape`, `linalg`)
   - Глава 3 — Pandas (`Series`, `DataFrame`, `.loc`/`.iloc`, `head`/`tail`)
   - Бесплатно: https://jakevdp.github.io/PythonDataScienceHandbook/

2. **Wes McKinney — «Python for Data Analysis» (3-е изд.)**
   - Главы 4–5 — основы NumPy и Pandas
   - Глава 7 — категориальные данные (`Categorical`)

3. **Документация:**
   - `numpy.arange` / `numpy.linspace` — разница между ними
   - `pandas.DataFrame.loc` vs `iloc`