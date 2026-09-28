
# Введение в анализ больших данных
# Задание 1

import numpy as np
import pandas as pd

# 1.1 Создаём вектор S1 со значениями от 1 до 10 всеми возможными способами

# Способ 1: обычный список Python
S1_list = list(range(1, 11))                # range(1, 11) -> 1..10
print("Способ 1 (list):", S1_list)

# Способ 2: NumPy-массив (предпочтительно для вычислений)
S1 = np.arange(1, 11)
print("Способ 2 (np.arange):", S1)
print("Тип:", type(S1))

# Способ 3: Pandas Series (удобно, если нужны метки/имена)
S1_series = pd.Series(range(1, 11))
print("Способ 3 (pd.Series):")
print(S1_series)
print()

# 1.2 Вектор S2 от 1 до 10 с шагом 0.5
S2 = np.arange(1, 10.5, 0.5)                # 19 значений: 1.0, 1.5, ..., 10.0
# Альтернатива через linspace: np.linspace(1, 10, 19)
print("S2:", S2)
print("Длина S2:", len(S2))
print()

# 1.3 Объединяем S1 и S2 в вектор S (в R это было бы c(S1, S2))
S = np.concatenate([S1, S2])
print("Объединённый вектор S:")
print(S)
print("Длина S:", len(S))
print()

# 1.4 Выводим 3-е, 4-е и 5-е значения вектора S
# Python индексирует с 0 -> элементы с индексами 2, 3, 4
print("3-е, 4-е, 5-е значения S:", S[[2, 3, 4]])

# 1.5 Выбираем только первое и четвёртое значения вектора S
# индексы 0 и 3
print("1-е и 4-е значения S:", S[[0, 3]])
print()

# 2. ТАБЛИЦА (DataFrame)

# 2.1 Создаём текстовые векторы City и Sex,
#     а также вектор с численностью Number
MaleStruct = ["Female", "Male"]                                   # «пол»
Cities = ["Владивосток", "Москва", "СПБ", "Челябинск", "Ульяновск", "Казань"]
Number = [12000, 8000, 15000, 5000, 20000, 22000]                 # численность

print("Cities:", Cities)
print("MaleStruct:", MaleStruct)
print("Number:", Number)

# Собираем столбец Sex, повторяя MaleStruct до нужной длины
Sex = (MaleStruct * 3)[:len(Number)]
print("Sex:", Sex)
print()

# 2.3 Объединяем три вектора в таблицу данных CITY с заголовками
CITY = pd.DataFrame({
    "City": Cities,
    "Sex": Sex,
    "Number": Number
})

# 2.4 Визуализируем содержимое таблицы CITY
print("Таблица CITY:")
print(CITY)
print()

# 2.5 Выводим имена переменных, входящих в таблицу CITY
print("Имена столбцов CITY:", CITY.columns.tolist())
print()

# 2.6 Извлекаем элементы 1–3 из столбца Number таблицы CITY
print("Элементы 1–3 столбца Number:")
print(CITY["Number"].iloc[0:3])
print()

# 2.7 Извлекаем все значения численности, превышающие 10000
print("Строки, где Number > 10000:")
print(CITY[CITY["Number"] > 10000])
print()

# 2.8 Извлекаем все значения численности мужского населения
print("Строки, где Sex == 'Male':")
print(CITY[CITY["Sex"] == "Male"])
print()

# 2.9 Первые 3 и последние 2 строки таблицы CITY
print("Первые 3 строки (head):")
print(CITY.head(3))
print()

print("Последние 2 строки (tail):")
print(CITY.tail(2))
print()

# 3. МАТРИЦА

# 3.1 Создаём матрицу размерности 2×8
m1 = np.arange(1, 17).reshape(2, 8)
print("Матрица m1 (2×8):")
print(m1)
print("Форма:", m1.shape)
print()

# 3.2 Создаём числовую матрицу размерности 4×6
m3 = np.arange(1, 25).reshape(4, 6)
print("Матрица m3 (4×6):")
print(m3)
print("Форма:", m3.shape)
print()

# 3.3 Транспонируем матрицу m3
m3_T = m3.T
print("Транспонированная m3 (6×4):")
print(m3_T)
print("Форма:", m3_T.shape)
print()

# 3.4 Создаём квадратную матрицу и вычисляем её определитель
#     Берём m3 @ m3.T -> получится матрица 4×4
square = m3 @ m3_T
print("Квадратная матрица (m3 @ m3.T):")
print(square)

det = np.linalg.det(square)
print("Определитель:", det)
print()


# 4.1 Создаём массив M из 24 элементов через np.array + reshape
M = np.array(range(1, 25)).reshape(3, 2, 4)     # 3×2×4 = 24
print("Массив M формы", M.shape)
print(M)
print()

# 4.2 Проверяем, является ли M массивом
print("Является ли M ndarray? ->", isinstance(M, np.ndarray))
print("ndim:", M.ndim)
print("shape:", M.shape)
print("size:", M.size)
print()


# 5. СПИСОК

# 5.1 Создаём список L1 из чисел 1..6 и значений (TRUE, FALSE) три раза
integers = list(range(1, 7))             # [1, 2, 3, 4, 5, 6]
booleans = [True, False] * 3             # [True, False, True, False, True, False]

L1 = [integers, booleans]
print("Список L1 (без имён):")
print(L1)
print()

# 5.2 Присваиваем имена элементам списка L1
#     В Python «именованный список» удобнее всего делать через dict
L1 = {
    "integers": integers,
    "booleans": booleans
}
print("Список L1 (с именами):")
print(L1)
print()


# 6. ФАКТОР

# 6.1 Вектор из 6 объектов трёх классов: Yes, No, Parhaps -> фактор
answers = ["Yes", "No", "Parhaps", "Yes", "No", "Parhaps"]

f1 = pd.Categorical(answers, categories=["Yes", "No", "Parhaps"])
print("Фактор f1:")
print(f1)
print("Уровни (categories):", f1.categories.tolist())
print("Коды:", f1.codes)
print()


# 7. ФРЕЙМ

# 7.1 Создаём фрейм данных F1 из элементов задания
b = ["a", "b", "c", "a"]
d = [(i % 2 == 0) for i in range(1, 5)]          # [False, True, False, True]
e = pd.Categorical(["soft", "hard", "soft", "medium"],
                   categories=["soft", "hard", "medium"])

F1 = pd.DataFrame({
    "b": b,
    "d": d,
    "e": e
})

print("Фрейм F1:")
print(F1)
print()
print("Типы столбцов F1:")
print(F1.dtypes)