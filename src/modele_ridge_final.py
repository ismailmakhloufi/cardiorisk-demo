"""
Modele Ridge final pour le risque de rehospitalisation a 3 mois.

Deux modeles sont sauvegardes :
1. modele_evaluation.joblib : entraine sur 80% et evalue sur 20% (rs=30)
2. modele_ridge_final.joblib : meme configuration, reentrainee sur 100% des donnees

Les performances rapportees viennent du modele d'evaluation.
"""
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from constrained_logistic import ConstrainedLogisticRidge

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "test.xlsx"
REPORT_DIR = PROJECT_DIR / "reports" / "modele_ridge"
FIGURES_DIR = REPORT_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models" / "modele_ridge"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CIBLE = "rehospit_3_mois"
RANDOM_STATE = 30
TEST_SIZE = 0.20
C_VALUE = 0.01
THRESHOLD = 0.22
QT_MIN_COEF = 0.01

FEATURES = [
    "cause_valvulaire",
    "PAS",
    "OG",
    "ATCD_d_hospitalisation",
    "dose_de_lasilix",
    "IMC",
    "IEC_dose",
    "FQ_ECG_sortie",
    "TG",
    "Hb",
    "lymphocyte",
    "duree_hospitalisation_jour",
    "FEVG",
    "creat",
    "cause_ischémique",
    "QT_corrige_sup_450",
]


def build_model():
    return ConstrainedLogisticRidge(
        c_value=C_VALUE,
        lower_bounds={FEATURES.index("QT_corrige_sup_450"): QT_MIN_COEF},
        random_state=42,
    )


def calc_metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "ap": float(average_precision_score(y_true, probs)),
        "sens": float(sens),
        "spec": float(spec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion(y_true, probs, title, file_name):
    pred = (probs >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() + 10)
    ax.set_xticks([0, 1], ["Non rehosp.", "Rehosp."])
    ax.set_yticks([0, 1], ["Non rehosp.", "Rehosp."])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Realite")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc(y_true, probs, title, file_name):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_value = roc_auc_score(y_true, probs)
    met = calc_metrics(y_true, probs, THRESHOLD)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.scatter(1 - met["spec"], met["sens"], c="red", s=80, zorder=5, label=f"Seuil = {THRESHOLD:.2f}")
    ax.set_xlabel("1 - Specificite")
    ax.set_ylabel("Sensibilite")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_coefficients(model, features, file_name, title):
    coef = model.coef_[0]
    order = np.argsort(np.abs(coef))[::-1]
    colors = ["#d32f2f" if c > 0 else "#1976d2" for c in coef[order]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(features)), coef[order], color=colors, height=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(np.array(features)[order], fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch

    ax.legend(
        [Patch(color="#d32f2f"), Patch(color="#1976d2")],
        ["Risque augmente", "Risque diminue"],
        loc="lower right",
    )
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 150})

df = pd.read_excel(DATA_PATH)
df["QT_corrige_sup_450"] = (df["QT_corrige"] > 450).astype(int)
missing = [col for col in FEATURES + [CIBLE] if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans test.xlsx: {missing}")

X = df[FEATURES].values.astype(float)
y = df[CIBLE].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Modele d'evaluation : train 80%, test 20%.
imputer_eval = SimpleImputer(strategy="median")
scaler_eval = RobustScaler()
X_train_eval = scaler_eval.fit_transform(imputer_eval.fit_transform(X_train))
X_test_eval = scaler_eval.transform(imputer_eval.transform(X_test))

model_eval = build_model()
model_eval.fit(X_train_eval, y_train)
train_probs = model_eval.predict_proba(X_train_eval)[:, 1]
test_probs = model_eval.predict_proba(X_test_eval)[:, 1]

train_metrics = calc_metrics(y_train, train_probs, THRESHOLD)
test_metrics = calc_metrics(y_test, test_probs, THRESHOLD)

coef_eval = pd.DataFrame({"variable": FEATURES, "coefficient": model_eval.coef_[0]})
coef_eval["abs_coef"] = coef_eval["coefficient"].abs()
coef_eval = coef_eval.sort_values("abs_coef", ascending=False)
coef_eval.to_excel(REPORT_DIR / "coefficients_evaluation.xlsx", index=False)

pd.DataFrame(
    {"y_reel": y_train, "proba": train_probs, "prediction": (train_probs >= THRESHOLD).astype(int)}
).to_csv(REPORT_DIR / "predictions_train.csv", index=False)
pd.DataFrame(
    {"y_reel": y_test, "proba": test_probs, "prediction": (test_probs >= THRESHOLD).astype(int)}
).to_csv(REPORT_DIR / "predictions_test.csv", index=False)

save_confusion(y_train, train_probs, f"Train (seuil={THRESHOLD:.2f})", "confusion_train.png")
save_confusion(y_test, test_probs, f"Test (seuil={THRESHOLD:.2f})", "confusion_test.png")
save_roc(y_train, train_probs, "ROC - Train", "roc_train.png")
save_roc(y_test, test_probs, "ROC - Test", "roc_test.png")
save_coefficients(model_eval, FEATURES, "coefficients_evaluation.png", "Coefficients Ridge - modele d'evaluation")

joblib.dump(
    {
        "model": model_eval,
        "imputer": imputer_eval,
        "scaler": scaler_eval,
        "features": FEATURES,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "C": C_VALUE,
        "class_weight": None,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    },
    MODELS_DIR / "modele_evaluation.joblib",
)

# Modele final d'utilisation : meme recette, entrainee sur 100% de la base.
imputer_final = SimpleImputer(strategy="median")
scaler_final = RobustScaler()
X_all_final = scaler_final.fit_transform(imputer_final.fit_transform(X))
model_final = build_model()
model_final.fit(X_all_final, y)

coef_final = pd.DataFrame({"variable": FEATURES, "coefficient": model_final.coef_[0]})
coef_final["abs_coef"] = coef_final["coefficient"].abs()
coef_final = coef_final.sort_values("abs_coef", ascending=False)
coef_final.to_excel(REPORT_DIR / "coefficients_modele_final.xlsx", index=False)
save_coefficients(model_final, FEATURES, "coefficients_modele_final.png", "Coefficients Ridge - modele final 100%")

joblib.dump(
    {
        "model": model_final,
        "imputer": imputer_final,
        "scaler": scaler_final,
        "features": FEATURES,
        "threshold": THRESHOLD,
        "C": C_VALUE,
        "class_weight": None,
        "evaluation_reference": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
    },
    MODELS_DIR / "modele_ridge_final.joblib",
)

summary = pd.DataFrame(
    [
        {"set": "train", **train_metrics},
        {"set": "test", **test_metrics},
    ]
)
summary.to_excel(REPORT_DIR / "resume_performances.xlsx", index=False)

print("=" * 80)
print("MODELE RIDGE FINAL")
print("=" * 80)
print(f"Base: {len(df)} patients | Variables: {len(FEATURES)}")
print(f"Configuration: Ridge L2 contraint, RobustScaler, C={C_VALUE}, QT>=+{QT_MIN_COEF}, seuil={THRESHOLD}")
print(f"Split evaluation: train/test {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, rs={RANDOM_STATE}")
print("\nTRAIN")
print(f"  AUC={train_metrics['auc']:.3f} AP={train_metrics['ap']:.3f}")
print(f"  Sens={train_metrics['sens']:.3f} Spec={train_metrics['spec']:.3f}")
print(f"  TN={train_metrics['tn']} FP={train_metrics['fp']} FN={train_metrics['fn']} TP={train_metrics['tp']}")
print("\nTEST")
print(f"  AUC={test_metrics['auc']:.3f} AP={test_metrics['ap']:.3f}")
print(f"  Sens={test_metrics['sens']:.3f} Spec={test_metrics['spec']:.3f}")
print(f"  TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}")

print("\nCoefficients modele evaluation:")
for _, row in coef_eval.iterrows():
    sign = "+" if row["coefficient"] > 0 else ""
    print(f"  {row['variable']}: {sign}{row['coefficient']:.4f}")

print(f"\nModele evaluation: {MODELS_DIR / 'modele_evaluation.joblib'}")
print(f"Modele final 100%: {MODELS_DIR / 'modele_ridge_final.joblib'}")
print(f"Figures: {FIGURES_DIR}")
