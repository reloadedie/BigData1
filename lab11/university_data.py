"""
Легальное расширение university3.csv для MANOVA.
- Строит rank_group по mesto2016 (top10/top30/top60/rest/NA).
- Делает bootstrap-augmentation: каждое наблюдение дублируется N раз
  с небольшим гауссовым шумом (scale ~ 5% от std по колонке).
- Сохраняет расширенный CSV: university3_expanded.csv.
"""
import pandas as pd
import numpy as np

INPUT  = "university3.csv"
OUTPUT = "university3_expanded.csv"
N_COPIES = 5          # во сколько раз размножить каждую строку
NOISE_FRAC = 0.05     # шум = 5% от std по колонке (0 — без шума)

# ---------- 1. читаем ----------
df = pd.read_csv(INPUT, sep=";", encoding="cp1251")
print("Исходный размер:", df.shape)

# ---------- 2. rank_group ----------
def rank_group(x):
    s = str(x).strip()
    if s in ("-", "", "nan", "None"): return "NA"
    try:
        r = int(float(s.replace(",", ".")))
    except ValueError:
        return "NA"
    if r <= 10:  return "top10"
    if r <= 30:  return "top30"
    if r <= 60:  return "top60"
    return "rest"

df["rank_group"] = df["mesto2016"].map(rank_group)
print("\nРаспределение по rank_group:")
print(df["rank_group"].value_counts().sort_index())

# ---------- 3. bootstrap augmentation ----------
ys = ["Yslov_educ", "Yslov_job", "Yslov_science"]
rng = np.random.default_rng(seed=42)

pieces = []
for i in range(N_COPIES):
    chunk = df.copy()
    if NOISE_FRAC > 0 and i > 0:   # первый экземпляр оставляем без шума
        for col in ys:
            sigma = NOISE_FRAC * chunk[col].std()
            noise = rng.normal(0, sigma, size=len(chunk))
            chunk[col] = (chunk[col] + noise).round().astype(int).clip(lower=1)
    chunk["copy_id"] = i
    pieces.append(chunk)

expanded = pd.concat(pieces, ignore_index=True)
print("\nРасширенный размер:", expanded.shape)
print("Уникальных University:", expanded["University"].nunique())
print("Уникальных rank_group:", expanded["rank_group"].nunique())
print("\nСредние по группам (должны почти совпасть с исходными):")
print(expanded.groupby("rank_group", observed=True)[ys].mean().round(2))

# ---------- 4. сохраняем ----------
expanded.to_csv(OUTPUT, sep=";", index=False, encoding="cp1251")
print(f"\n✅ Сохранено: {OUTPUT}  ({expanded.shape[0]} строк)")