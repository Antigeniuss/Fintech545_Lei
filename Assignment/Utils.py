import numpy as np
import pandas as pd
from scipy import stats as st

# ---------- helpers ----------
def _symm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _to_corr_from_cov(S: np.ndarray):
    std = np.sqrt(np.diag(S))
    std = np.where(std <= 0.0, 1.0, std)
    D_inv = np.diag(1.0 / std)
    R = D_inv @ S @ D_inv
    return R, std

def _to_cov_from_corr(R: np.ndarray, std: np.ndarray):
    D = np.diag(std)
    return D @ R @ D

# ---------- Rebonato–Jäckel (near-PSD) ----------
def near_psd_correlation(R_df: pd.DataFrame, eps: float = 0.0) -> pd.DataFrame:
    C = _symm(R_df.values)
    eigvals, S = np.linalg.eigh(C)
    lam_p = np.maximum(eigvals, eps)           # clip eigenvalues

    # scaling t_i = 1 / sum_j S_{ij}^2 * lam'_j
    Si2 = S**2
    denom = Si2 @ lam_p
    denom = np.where(denom <= 0, 1.0, denom)
    t = 1.0 / denom

    # B = sqrt(T) S sqrt(Lam')
    B = (np.sqrt(t)[:, None]) * S * (np.sqrt(lam_p)[None, :])
    C_hat = _symm(B @ B.T)

    # normalize to diag=1
    d = np.sqrt(np.diag(C_hat))
    C_hat = C_hat / np.outer(d, d)
    np.fill_diagonal(C_hat, 1.0)

    return pd.DataFrame(C_hat, index=R_df.index, columns=R_df.columns)

def near_psd_covariance(S_df: pd.DataFrame, eps: float = 0.0) -> pd.DataFrame:
    S = _symm(S_df.values)
    R, std = _to_corr_from_cov(S)
    R_psd = near_psd_correlation(pd.DataFrame(R, index=S_df.index, columns=S_df.columns), eps=eps).values
    S_psd = _to_cov_from_corr(R_psd, std)
    return pd.DataFrame(_symm(S_psd), index=S_df.index, columns=S_df.columns)

# ---------- Higham (nearest correlation) ----------
def higham_correlation(R_df: pd.DataFrame, tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    X = _symm(R_df.values.copy())
    for _ in range(max_iter):
        # PSD projection
        w, V = np.linalg.eigh(_symm(X))
        w = np.maximum(w, 0.0)
        X_psd = V @ np.diag(w) @ V.T
        # set diag=1
        np.fill_diagonal(X_psd, 1.0)
        if np.linalg.norm(X_psd - X, ord='fro') < tol:
            X = X_psd
            break
        X = X_psd
    return pd.DataFrame(X, index=R_df.index, columns=R_df.columns)

def higham_covariance(S_df: pd.DataFrame, tol: float = 1e-8, max_iter: int = 200) -> pd.DataFrame:
    S = _symm(S_df.values)
    R, std = _to_corr_from_cov(S)
    R_h = higham_correlation(pd.DataFrame(R, index=S_df.index, columns=S_df.columns),
                             tol=tol, max_iter=max_iter).values
    S_h = _to_cov_from_corr(R_h, std)
    return pd.DataFrame(_symm(S_h), index=S_df.index, columns=S_df.columns)

# ---------- PD / PSD checker ----------
def check_pd_psd(cov, tol: float = 1e-10):
    """
    Check a symmetric matrix's definiteness with a tolerance.
    Return dict with status ('PD' / 'PSD' / 'Non-PSD'), min/max eigenvalues and all eigvals.
    """
    A = np.asarray(cov, dtype=float)
    A = _symm(A)                     # enforce symmetry
    eigvals = np.linalg.eigvalsh(A)
    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())
    if min_eig > tol:
        status = "PD"
    elif min_eig >= -tol:
        status = "PSD"                      # near-zero negatives treated as numerical noise
    else:
        status = "Non-PSD"
    return {"status": status, "min_eig": min_eig, "max_eig": max_eig, "eigvals": eigvals}

# ---------- Simulate Multivariate Normal Distribution ----------
def simulate_multivariate_normal(
    mean,
    cov,                                  # pd.DataFrame or np.ndarray
    n_samples: int = 100_000,
    seed: int = 42,
    tol: float = 1e-10,
    return_info: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Simulate X ~ N(mean, cov) robustly.
      - mean: scalar 0 or array-like (d,)
      - cov:  (d x d) covariance (DataFrame or ndarray)
      - If PD (min eig > tol): use Cholesky
      - Else: eigen-decompose and clip eigvals below tol to 0
    """
    rng = np.random.default_rng(seed)

    # Convert & symmetrize
    A = np.asarray(cov, dtype=float)
    A = _symm(A)
    d = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("cov must be square (d x d).")
    if not np.all(np.isfinite(A)):
        raise ValueError("cov contains NaN/Inf; clean your data first.")

    # Decide PD/PSD/Non-PSD
    chk = check_pd_psd(A, tol=tol)

    # Build factor L
    used = "chol"
    try:
        if chk["status"] == "PD":
            L = np.linalg.cholesky(A)                 # fast & stable
        else:
            raise np.linalg.LinAlgError("not PD")
    except np.linalg.LinAlgError:
        # Eigen fallback (handles PSD or slightly Non-PSD after clipping)
        w, V = np.linalg.eigh(A)
        w = np.where(w < tol, 0.0, w)                 # clip tiny negatives to 0
        L = V @ np.diag(np.sqrt(w))
        used = "eigh-clip"

    # Draw samples
    Z = rng.standard_normal((n_samples, d))
    mu = np.zeros(d) if np.isscalar(mean) else np.asarray(mean).reshape(1, -1)
    X = Z @ L.T + mu

    if return_info:
        info = {"factorization": used, "pd_check": chk, "tol": tol}
        return X, info
    return X

# ---------- calculate covariance matrix----------
def calculate_cov(X: np.ndarray) -> np.ndarray:
    """Sample covariance matrix (unbiased, n-1 in denominator)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


# --- PCA reduce covariance to >= 99% explained ---
def pca_covariance(cov_df: pd.DataFrame, threshold: float = 0.99):
    """Return (PCA-approximated covariance, k, cum_explained array)."""
    A = np.asarray(cov_df.values, dtype=float)
    A = _symm(A)            # enforce symmetry

    # eigen-decomposition (for symmetric covariance)
    w, V = np.linalg.eigh(A)                    # w ascending
    idx = np.argsort(w)[::-1]                   # sort descending
    w, V = w[idx], V[:, idx]

    explained_ratio = w / w.sum()
    cum = np.cumsum(explained_ratio)
    k = int(np.searchsorted(cum, threshold) + 1)

    # reconstruct covariance with top-k components
    Vk = V[:, :k]
    Wk = np.diag(w[:k])
    A_pca = Vk @ Wk @ Vk.T
    A_pca = _symm(A_pca)
    return pd.DataFrame(A_pca, index=cov_df.index, columns=cov_df.columns), k, cum

# ---- Unified VaR function (Normal / t distribution) ----
def var_from_returns(returns: pd.Series = None, 
                     alpha: float = 0.05, dist: str = "normal") -> dict:
    """
    Compute Value-at-Risk (VaR) under Normal or Student-t distribution.
    
    Parameters:
        returns : pandas Series of returns (if mu/sigma not provided, will be estimated)
        alpha   : significance level (e.g., 0.05 for 95% VaR)
        dist    : "normal" or "t"
        
    Returns:
        dict with:
            - VaR Absolute: absolute loss magnitude (positive number)
            - VaR Diff from Mean: VaR minus the mean
    """
    r = returns.squeeze().dropna().astype(float)

    # ---- Normal distribution VaR ----
    if dist.lower() == "normal":
        mu = r.mean()
        sigma = r.std(ddof=1)
        z = st.norm.ppf(alpha)
        var_quantile = mu + z * sigma
    
    # ---- t distribution VaR ----
    elif dist.lower() == "t":
        nu, mu, sigma = st.t.fit(returns)
        t_alpha = st.t.ppf(alpha, nu)
        print(nu)
        var_quantile = mu + t_alpha * sigma
    
    else:
        raise ValueError("dist must be 'normal' or 't'")
    
    return {
        "VaR Absolute": abs(var_quantile),
        "VaR Diff from Mean": mu - var_quantile
    }


import numpy as np
import pandas as pd
from scipy import stats as st

def var_mc_t_from_returns(returns: pd.Series,
                          alpha: float = 0.05,
                          n_samples: int = 100_000,
                          seed: int = 42) -> dict:
    """
    Monte Carlo Value-at-Risk (VaR) using a Student-t distribution fitted to input returns.

    Steps:
      1. Clean the input return series.
      2. Fit a Student-t distribution to returns via Maximum Likelihood Estimation (MLE).
      3. Generate N Monte Carlo simulated returns from the fitted t-distribution.
      4. Compute the alpha-quantile from the simulated returns as the VaR point.
      5. Return absolute VaR and deviation from the mean.

    Parameters
    ----------
    returns   : pd.Series
        Time series of returns.
    alpha     : float, default=0.05
        Significance level (e.g., 0.05 for 95% VaR).
    n_samples : int, default=100_000
        Number of Monte Carlo simulations.
    seed      : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict with:
        - "VaR Absolute": magnitude of loss (positive number).
        - "VaR Diff from Mean": quantile relative to mean (typically negative).
        - "params": dictionary of fitted t-distribution parameters (df, loc, scale).
    """
    r = returns.squeeze().dropna().astype(float)
    nu, loc, scale = st.t.fit(r.values) 

    # Monte Carlo simulation
    rng = np.random.default_rng(seed)
    sims = loc + scale * rng.standard_t(nu, size=n_samples)

    var_q = np.quantile(sims, alpha)

    return {
        "VaR Absolute": abs(var_q),
        "VaR Diff from Mean": sims.mean() - var_q,
    }