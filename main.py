import display
import numpy as np
import pandas as pd

# 1.1 Создание вектора S1 со значениями от 1 до 10

# Способ 1: обычный список Python
S1_list = list(range(1, 6))          # range(1, 11) → 1..10
print("Список:", S1_list)

# Способ 2: NumPy-массив (предпочтительно для вычислений)
S1 = np.arange(1, 6)
print("NumPy array:", S1)
print("Тип:", type(S1))

# Способ 3: Pandas Series (удобно, если нужны метки/имена)
S1_series = pd.Series(range(1, 6))
print("Pandas Series:\n", S1_series)

# 1.2 Вектор S2 от 1 до 10 с шагом 0.5
S2 = np.arange(1, 6, 1.5)          # stop не включается, поэтому 10.5
# или
S2_alt = np.linspace(1, 5.5, num=4)   # 19 точек от 1 до 10 включительно

print("S2 (arange):", S2)
print("Длина S2:", len(S2))
print()
print("S2_alt (linspace):", S2_alt)
print("Длина S2_alt:", len(S2_alt))

# 1.3 Объединение векторов S1 и S2 в вектор S
# В R: c(S1, S2)

S = np.concatenate([S1, S2])

print("Объединённый вектор S:")
print(S)
print("Длина S:", len(S))

# 1.4 Вывести 1-е, 2-е и 3-е значения вектора S
# В Python индексация с 0 → S[0:3]

print("Элементы 1, 2, 3:", S[0:3])

# или явно:
print("Через fancy indexing:", S[[0, 1, 2]])

print("5-й и 6-й элементы:", S[[4, 5]])

# 2.1 Создание векторов (столбцов)
MaleStruct = ["Female", "Male"]
Number = [10,1,50,5,20,22]
City = [ "Владивосток", "Москва", "СПБ", "Челябинск", "Ульяновск", "" ]

print(names)
print(subject)
print(points)

EXAM = pd.DataFrame({
    "Names": names,
    "Subject": subject,
    "Points": points
})

EXAM

# 2.4 Просмотр структуры таблицы
# В R: str(EXAM)
# В Python: .info() или .dtypes + .shape

print("Информация о таблице:")
EXAM.info()

print("\nТипы столбцов:")
print(EXAM.dtypes)

print("\nРазмерность (строки, столбцы):", EXAM.shape)

print(EXAM.columns.tolist())
# или просто
EXAM.columns

# 2.6 Извлечь элементы 1–3 из столбца Points

print(EXAM["Points"].iloc[0:3])

# альтернативы:
EXAM.Points.iloc[0:3]
EXAM.loc[0:2, "Points"]   # .loc использует метки (здесь совпадают с индексами)

# 2.7 Все значения баллов, превышающие 50


print(EXAM[EXAM["Points"] > 50])

# только столбец Number:
print("\nТолько Points > 50:")
print(EXAM.loc[EXAM["Points"] > 50, "Points"])
#print(EXAM.loc(EXAM["Points" > 40, 2])) # пока не получается

# 2.8 Все значения баллов по математике
print(EXAM[EXAM["Subject"] == "Math"])
# или только числа:
print("\nБаллы по математике:")
print(EXAM.loc[EXAM["Subject"] == "Math", "Points"])

# 2.9 Первые и последние строки


print("Первые 3 строки:")
display(EXAM.head(1))

print("\nПоследние 2 строки:")
display(EXAM.tail(3))

# 3.1 Создание матрицы размерности 2×8
m1 = np.arange(1, 7).reshape(2, 3)   # 16 элементов → 2 строки × 8 столбцов
print("Матрица 2×3:")
print(m1)
print("Форма:", m1.shape)

# 3.3 Транспонирование матрицы


m2 = m1.T
print("Транспонированная матрица (3×2):")
print(m2)
print("Форма:", m2.shape)

# 3.4 Квадратная матрица и определитель

square = np.array([[1, 2],
                   [3, 4]])
print("Квадратная матрица:")
print(square)

det = np.linalg.det(square)
print("Определитель:", det)

# 4.1 Массив из 24 элементов

# Создадим массив размерности (3, 2, 4) — 3×2 матрицы, 4 «слоя»
M = np.arange(1, 13).reshape(3, 2, 2)
print("Массив M формы", M.shape)
print(M)

# Альтернатива: явно через np.array и reshape

# 4.2 Проверка, является ли объект массивом

print("Является ли M ndarray?", isinstance(M, np.ndarray))
print("Количество измерений (ndim):", M.ndim)
print("Форма:", M.shape)
print("Общее число элементов:", M.size)

# 5.1 Создание списка L1: числа от 1 до 6 + TRUE/FALSE три раза


numbers = list(range(1, 5))
full_name = ["ФИО"] * 4

L1 = [numbers, full_name]          # обычный список списков
# или
L1_dict = {
    "integers": numbers,
    "full_name": full_name
}

print("L1 (list):")
print(L1)

print("\nL1_dict:")
print(L1_dict)
# 6.1 Вектор из 6 объектов трёх классов: Low, Medium, High → фактор

classes = ["Low", "Medium", "High"]
answers1 = [classes, classes]

answers = ["Low", "Medium", "High", "Low", "Medium", "High"]

# Создание Categorical
f1 = pd.Categorical(answers, categories=["Low", "Medium", "High"])
print(f1)
print("Уровни (categories):", f1.categories.tolist())
print("Коды:", f1.codes)

df_cat = pd.DataFrame({"answer": answers})
df_cat["answer"] = df_cat["answer"].astype("category")
print(df_cat.dtypes)
print(df_cat["answer"])