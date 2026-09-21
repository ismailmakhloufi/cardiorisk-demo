"""
Test option 1 : regression logistique Ridge avec contrainte de signe.

Contrainte imposee : coefficient de QT_corrige_sup_450 >= 0.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "test.xlsx"

RANDOM_STATE = 30
C_VALUE = 0.01
THRESHOLD = 0.22

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


class ConstrainedLogisticRidge:
    def __init__(self, c_value=0.01, positive_indices=None):
        self.c_value = c_value
        self.positive_indices = positive_indices or []
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        n, p = x.shape
        y = y.astype(float)
        alpha = 1.0 / (self.c_value * n)

        def objective(params):
            beta = params[:p]
            intercept = params[p]
            z = x @ beta + intercept
            # logloss stable: log(1 + exp(z)) - y*z
            loss = np.mean(np.logaddexp(0, z) - y * z)
            penalty = 0.5 * alpha * np.sum(beta**2)
            return loss + penalty

        def gradient(params):
            beta = params[:p]
            intercept = params[p]
            z = x @ beta + intercept
            prob = expit(z)
            error = prob - y
            grad_beta = (x.T @ error) / n + alpha * beta
            grad_intercept = np.mean(error)
            return np.r_[grad_beta, grad_intercept]

        bounds = [(None, None)] * (p + 1)
        for idx in self.positive_indices:
            bounds[idx] = (0.0, None)

        result = minimize(
            objective,
            x0=np.zeros(p + 1),
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(result.message)

        self.coef_ = result.x[:p].reshape(1, -1)
        self.intercept_ = np.array([result.x[p]])
        return self

    def decision_function(self, x):
        return x @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, x):
        p1 = expit(self.decision_function(x))
        return np.c_[1 - p1, p1]


def calc_metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    return {
        "auc": roc_auc_score(y_true, probs),
        "ap": average_precision_score(y_true, probs),
        "sens": sens,
        "spec": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


df = pd.read_excel(DATA_PATH)
df["QT_corrige_sup_450"] = (df["QT_corrige"] > 450).astype(int)
X = df[FEATURES].values.astype(float)
y = df["rehospit_3_mois"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(strategy="median")
scaler = RobustScaler()
X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_p = scaler.transform(imputer.transform(X_test))

qt_idx = FEATURES.index("QT_corrige_sup_450")
model = ConstrainedLogisticRidge(c_value=C_VALUE, positive_indices=[qt_idx])
model.fit(X_train_p, y_train)

train_probs = model.predict_proba(X_train_p)[:, 1]
test_probs = model.predict_proba(X_test_p)[:, 1]
train_metrics = calc_metrics(y_train, train_probs, THRESHOLD)
test_metrics = calc_metrics(y_test, test_probs, THRESHOLD)

print("=" * 80)
print("REGRESSION LOGISTIQUE CONTRAINTE : coef(QT_corrige_sup_450) >= 0")
print("=" * 80)
print(f"C={C_VALUE}, seuil={THRESHOLD}, rs={RANDOM_STATE}")
print(f"Coefficient QT_corrige_sup_450: {model.coef_[0][qt_idx]:+.6f}")
print("\nTRAIN")
print(f"AUC={train_metrics['auc']:.3f} AP={train_metrics['ap']:.3f} Sens={train_metrics['sens']:.3f} Spec={train_metrics['spec']:.3f}")
print(f"TN={train_metrics['tn']} FP={train_metrics['fp']} FN={train_metrics['fn']} TP={train_metrics['tp']}")
print("\nTEST")
print(f"AUC={test_metrics['auc']:.3f} AP={test_metrics['ap']:.3f} Sens={test_metrics['sens']:.3f} Spec={test_metrics['spec']:.3f}")
print(f"TN={test_metrics['tn']} FP={test_metrics['fp']} FN={test_metrics['fn']} TP={test_metrics['tp']}")
print("\nCoefficients:")
for feature, coef in sorted(zip(FEATURES, model.coef_[0]), key=lambda x: -abs(x[1])):
    print(f"  {feature}: {coef:+.4f}")
