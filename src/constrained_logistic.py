import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class ConstrainedLogisticRidge:
    """Logistic regression L2 with optional lower bounds on coefficients."""

    def __init__(self, c_value=0.01, lower_bounds=None, random_state=42):
        self.c_value = c_value
        self.lower_bounds = lower_bounds or {}
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        y = y.astype(float)
        alpha = 1.0 / (self.c_value * n_samples)

        def objective(params):
            beta = params[:n_features]
            intercept = params[n_features]
            logits = x @ beta + intercept
            loss = np.mean(np.logaddexp(0, logits) - y * logits)
            penalty = 0.5 * alpha * np.sum(beta**2)
            return loss + penalty

        def gradient(params):
            beta = params[:n_features]
            intercept = params[n_features]
            probs = expit(x @ beta + intercept)
            error = probs - y
            grad_beta = (x.T @ error) / n_samples + alpha * beta
            grad_intercept = np.mean(error)
            return np.r_[grad_beta, grad_intercept]

        bounds = [(None, None)] * (n_features + 1)
        for idx, lower_bound in self.lower_bounds.items():
            bounds[int(idx)] = (float(lower_bound), None)

        result = minimize(
            objective,
            x0=np.zeros(n_features + 1),
            jac=gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(result.message)

        self.coef_ = result.x[:n_features].reshape(1, -1)
        self.intercept_ = np.array([result.x[n_features]])
        return self

    def decision_function(self, x):
        return x @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, x):
        probs = expit(self.decision_function(x))
        return np.c_[1 - probs, probs]

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
