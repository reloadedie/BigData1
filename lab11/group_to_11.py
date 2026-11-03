import pandas as pd

df = pd.read_csv("university2.csv", sep=";", encoding="cp1251")
print("До:", df.shape, "| Уровней University:", df["University"].nunique())

def to_11(m):
    s = str(m).strip()
    if s in ('-', '', 'nan', 'None'):
        return 11                      # нет места → 11-я группа
    try:
        r = int(float(s.replace(',', '.')))
    except ValueError:
        return 11
    return min(11, (r - 1) // 8 + 1)   # 1-8→1, 9-16→2, ..., 81+→11

df["University"] = df["mesto2016"].map(to_11).astype(int)

print("После:", df.shape, "| Уровней University:", df["University"].nunique())
print(df["University"].value_counts().sort_index())

df.to_csv("university_11.csv", sep=";", index=False, encoding="cp1251")
print("\n✅ Сохранено: university_11.csv  — 83 строки, 11 уровней University")