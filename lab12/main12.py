# -*- coding: utf-8 -*-
"""
Кластерный анализ животных по Klaster.csv
Признаки: температура, расход энергии, относительная поверхность, масса,
          кислород в органах: мозг, почки, печень, селезёнка, мышцы.
Перед дендрограммой удаляется первый столбец (vid).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
np.random.seed(42)

# ----------------------------------------------------------------------------
# 1. Загрузка данных
# ----------------------------------------------------------------------------
# Файл с разделителем ';' и десятичной запятой
df = pd.read_csv("Klaster.csv", sep=";", decimal=",")

print("Исходные данные:")
print(df)
print("\nРазмер таблицы:", df.shape)

# ----------------------------------------------------------------------------
# 2. Удаляем первый столбец (vid) — как требует задание
# ----------------------------------------------------------------------------
labels_names = df["vid"].astype(str).tolist()   # сохраняем имена для подписей
X = df.drop(columns=[df.columns[0]]).copy()      # признаки

features = X.columns.tolist()
print("\nПризнаки для кластеризации:")
print(features)

# ----------------------------------------------------------------------------
# 3. Стандартизация (z-оценки)
# ----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)

print("\nСредние после стандартизации:", X_scaled.mean().round(3).tolist())
print("Ст.откл. после стандартизации:", X_scaled.std(ddof=0).round(3).tolist())

# ----------------------------------------------------------------------------
# 4. Иерархическая кластеризация (Ward) + дендрограмма
# ----------------------------------------------------------------------------
Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=labels_names, leaf_rotation=45, leaf_font_size=10)
plt.title("Дендрограмма (метод Ward)")
plt.ylabel("Расстояние объединения")
plt.xlabel("Животные")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=150)
plt.show()

# Срез на 3 кластера
labels_h = fcluster(Z, t=3, criterion="maxclust")
df_h = X.copy()
df_h["cluster_h"] = labels_h
df_h["vid"] = labels_names

print("\n=== Иерархическая кластеризация (3 кластера) ===")
print("Размеры кластеров:")
print(df_h["cluster_h"].value_counts().sort_index())
print("\nСредние признаки по кластерам:")
print(df_h.groupby("cluster_h")[features].mean().round(2))
print("\nСостав кластеров:")
for c in sorted(df_h["cluster_h"].unique()):
    print(f"  Кластер {c}: {', '.join(df_h.loc[df_h['cluster_h']==c, 'vid'])}")

# ----------------------------------------------------------------------------
# 5. K-means: выбор числа кластеров (elbow + silhouette)
# ----------------------------------------------------------------------------
inertias, silhouettes = [], []
K_range = range(2, 7)   # всего 11 объектов
for k in K_range:
    km = KMeans(n_clusters=k, n_init=25, random_state=42)
    lab = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lab))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, "o-")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow")
axes[1].plot(list(K_range), silhouettes, "o-")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette"); axes[1].set_title("Silhouette")
plt.tight_layout()
plt.savefig("kmeans_choice.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------------
# 6. Финальный k-means
# ----------------------------------------------------------------------------
k_opt = 3
km = KMeans(n_clusters=k_opt, n_init=25, random_state=42)
labels_km = km.fit_predict(X_scaled)

df_km = X.copy()
df_km["cluster"] = labels_km + 1
df_km["vid"] = labels_names

print(f"\n=== K-means (k={k_opt}) ===")
print("Размеры кластеров:", df_km["cluster"].value_counts().sort_index().to_dict())
print("\nЦентры (стандартизированные координаты):")
print(pd.DataFrame(km.cluster_centers_, columns=features).round(3))
print("\nСредние в исходных единицах:")
print(df_km.groupby("cluster")[features].mean().round(2))
print("\nСостав кластеров:")
for c in sorted(df_km["cluster"].unique()):
    print(f"  Кластер {c}: {', '.join(df_km.loc[df_km['cluster']==c, 'vid'])}")

# ----------------------------------------------------------------------------
# 7. Визуализация в PCA-плоскости
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels_km, cmap="tab10",
                 s=120, edgecolors="k", alpha=0.85)
for i, name in enumerate(labels_names):
    plt.annotate(name, (coords[i, 0], coords[i, 1]),
                 textcoords="offset points", xytext=(6, 6), fontsize=9)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"K-means, k={k_opt} (PCA 2D)")
plt.colorbar(sc, label="cluster")
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=150)
plt.show()
print("Доля дисперсии PC1+PC2:", round(pca.explained_variance_ratio_.sum(), 3))

# ----------------------------------------------------------------------------
# 8. Тепловая карта средних по кластерам
# ----------------------------------------------------------------------------
means = df_km.groupby("cluster")[features].mean()
plt.figure(figsize=(10, 4))
sns.heatmap(means, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Средние значения признаков по кластерам (k-means)")
plt.tight_layout()
plt.savefig("cluster_means_heatmap.png", dpi=150)
plt.show()

print("\nГотово. Сохранены файлы: dendrogram.png, kmeans_choice.png, "
      "pca_clusters.png, cluster_means_heatmap.png")