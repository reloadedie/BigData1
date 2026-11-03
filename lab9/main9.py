"""
Лабораторная работа №9. Бинарная (логистическая) регрессия.
Данные: datlg.csv (кардиология: y — делать операцию или нет).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------
df = pd.read_csv("datlg.csv", index_col=0)
print("Размер данных:", df.shape)
print("\nПервые строки:")
print(df.head())
print("\nРаспределение y:")
print(df["y"].value_counts())
print("\nДоля y=1: {:.3f}".format(df["y"].mean()))

# ---------------------------------------------------------------------------
# 2. Разделение на обучающую и тестовую выборки (стратифицированное)
# ---------------------------------------------------------------------------
train, test = train_test_split(
    df, test_size=0.3, random_state=7, stratify=df["y"]
)
print("\ntrain:", train.shape, " test:", test.shape)
print("доля y=1 train: {:.3f}".format(train["y"].mean()))
print("доля y=1 test : {:.3f}".format(test["y"].mean()))

formula = "y ~ sex + x1 + x2 + x3 + x4"

# ---------------------------------------------------------------------------
# 3. Логит-модель на обучающей выборке
# ---------------------------------------------------------------------------
model = smf.logit(formula, data=train).fit(disp=False)
print("\n" + "=" * 70)
print("ЛОГИТ-МОДЕЛЬ (обучающая выборка)")
print("=" * 70)
print(model.summary())

# Значимость модели в целом (LR-тест)
lr = 2 * (model.llf - model.llnull)
df_lr = model.df_model
p_lr = stats.chi2.sf(lr, df_lr)
print(f"\nLR chi2 = {lr:.3f}, df = {df_lr:.0f}, p = {p_lr:.4e}")
print("Вывод: модель " + ("значима в целом" if p_lr < 0.05 else "НЕ значима в целом"))

# Пробит-модель для сравнения
m_probit = smf.probit(formula, data=train).fit(disp=False)
print("\n" + "=" * 70)
print("ПРОБИТ-МОДЕЛЬ (обучающая выборка)")
print("=" * 70)
print(m_probit.summary().tables[1])
print(f"AIC logit = {model.aic:.3f}, AIC probit = {m_probit.aic:.3f}")

# ---------------------------------------------------------------------------
# 4. Оценка качества на train и test
# ---------------------------------------------------------------------------
def evaluate(m, data, name):
    y = data["y"]
    p = m.predict(data)
    yhat = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, yhat)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, yhat)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    auc = roc_auc_score(y, p)

    print(f"\n=== {name} ===")
    print("Матрица неточностей:\n", cm)
    print(f"Accuracy    = {acc:.4f}")
    print(f"Sensitivity = {sens:.4f}")
    print(f"Specificity = {spec:.4f}")
    print(f"AUC         = {auc:.4f}")
    print(classification_report(y, yhat, digits=3, zero_division=0))
    return p, auc

p_tr, auc_tr = evaluate(model, train, "TRAIN")
p_te, auc_te = evaluate(model, test, "TEST")

print(f"\nПросадка AUC train -> test: {auc_tr - auc_te:+.4f}")

# ---------------------------------------------------------------------------
# 5. ROC-кривые train/test
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, data, p, title, auc_v in [
    (axes[0], train, p_tr, "Train", auc_tr),
    (axes[1], test, p_te, "Test", auc_te),
]:
    fpr, tpr, _ = roc_curve(data["y"], p)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc_v:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC ({title})")
    ax.set_xlabel("FPR (1 - Specificity)")
    ax.set_ylabel("TPR (Sensitivity)")
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_train_test.png", dpi=120)
plt.show()

# Матрица неточностей на тесте
cm_te = confusion_matrix(test["y"], (p_te >= 0.5).astype(int))
ConfusionMatrixDisplay(cm_te, display_labels=[0, 1]).plot(cmap="Blues")
plt.title("Матрица неточностей (test, logit, thr=0.5)")
plt.tight_layout()
plt.savefig("cm_test.png", dpi=120)
plt.show()

# ---------------------------------------------------------------------------
# 6. Кросс-валидация (Stratified K-Fold)
# ---------------------------------------------------------------------------
X = df[["sex", "x1", "x2", "x3", "x4"]]
y = df["y"]
sk_model = LogisticRegression(max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = cross_val_score(sk_model, X, y, cv=cv, scoring="accuracy")
auc_scores = cross_val_score(sk_model, X, y, cv=cv, scoring="roc_auc")

print("\n" + "=" * 70)
print("КРОСС-ВАЛИДАЦИЯ (5-fold, StratifiedKFold)")
print("=" * 70)
print("Accuracy по фолдам:", np.round(acc_scores, 4))
print(f"Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print("AUC по фолдам    :", np.round(auc_scores, 4))
print(f"AUC:      {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 7. Итоговый вывод
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 70)
print(f"1. Модель в целом значима (LR p = {p_lr:.2e}).")
print("2. Значимые предикторы (p<0.05):")
for name, pval in model.pvalues.items():
    if pval < 0.05 and name != "Intercept":
        coef = model.params[name]
        print(f"   {name}: coef={coef:+.4f}, p={pval:.4f} "
              f"({'рост' if coef > 0 else 'снижение'} шанса y=1)")
print(f"3. AUC train={auc_tr:.3f}, test={auc_te:.3f}, CV={auc_scores.mean():.3f}.")
print("4. Accuracy train={:.3f}, test={:.3f}, CV={:.3f}.".format(
    accuracy_score(train['y'], (p_tr >= 0.5).astype(int)),
    accuracy_score(test['y'], (p_te >= 0.5).astype(int)),
    acc_scores.mean(),
))
if auc_te >= 0.7:
    print("5. Качество приемлемое (AUC >= 0.7), модель пригодна для прогноза.")
else:
    print("5. Качество низкое (AUC < 0.7), модель ограниченно пригодна.")