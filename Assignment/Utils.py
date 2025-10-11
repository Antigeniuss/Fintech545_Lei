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



# ---------- ES from Normal (closed-form; use empirical mean & std) ----------
def es_normal(returns: pd.Series, alpha: float = 0.05):
    """
    Compute ES under Normal using empirical mean and std.
    VaR_alpha = mu + sigma * Phi^{-1}(alpha)
    ES_alpha  = mu - sigma * phi(z_alpha) / alpha
    """
    returns = returns.squeeze().dropna().astype(float)
    mu_hat = returns.mean()
    sigma_hat = returns.std(ddof=1)
    z = st.norm.ppf(alpha)

    es_ = mu_hat - sigma_hat * st.norm.pdf(z) / alpha
    return {
        "ES Absolute": abs(es_),
        "ES Diff from Mean": returns.mean() - es_,
    }


# ---------- 8.5: ES from t (closed-form; fit df, loc, scale from data) ----------
def es_t(returns: pd.Series, alpha: float = 0.05):
    """
    Closed-form VaR/ES for Student-t with fitted parameters.
    ES_alpha  = mu - sigma * [ f_t(t_alpha)*(df + t_alpha^2) / ((df-1)*alpha) ]
                (valid for df > 1)
    """
    returns = returns.squeeze().dropna().astype(float)
    df_hat, mu_hat, sigma_hat = st.t.fit(returns.values)
    if df_hat <= 1:
        raise ValueError(f"Fitted df <= 1 ({df_hat:.3f}); ES formula requires df > 1.")
    t_alpha = st.t.ppf(alpha, df_hat)
    es_tail_term = (st.t.pdf(t_alpha, df_hat) * (df_hat + t_alpha**2)) / ((df_hat - 1) * alpha)
    es_ = mu_hat - sigma_hat * es_tail_term
    return {
        "ES Absolute": abs(es_),
        "ES Diff from Mean": returns.mean() - es_,
    }


# ---------- 8.6: ES from Simulation (draw from fitted t; compare to 8.5) ----------
def es_sim_from_fitted_t(returns: pd.Series, alpha: float = 0.05, n_sim: int = 100_000, random_state: int = 42):
    """
    Simulate returns from fitted t(df, loc=mu, scale=sigma) and compute empirical VaR/ES.
    """
    returns = returns.squeeze().dropna().astype(float)
    df_hat, mu_hat, sigma_hat = st.t.fit(returns.values)
    rng = np.random.default_rng(random_state)
    sims = mu_hat + sigma_hat * rng.standard_t(df_hat, size=n_sim)
    # empirical VaR and ES
    var_sim = np.quantile(sims, alpha, method="linear")
    es_sim = sims[sims <= var_sim].mean()
    return {
        "ES Absolute": abs(es_sim),
        "ES Diff from Mean": sims.mean() - es_sim,
    }


# ---------- generate Gaussian Copula samples ----------
def generate_copula_samples(
    n_assets: int,
    dist_types: list[str],
    data: pd.DataFrame | np.ndarray,
    corr_method: str = "spearman",
    n_samples: int = 100_000,
    random_state: int = 42,
    ):
    """
    Generate correlated samples using a Gaussian Copula.

    Parameters
    ----------
    n_assets : int
        Number of assets (must match number of columns in data).
    dist_types : list[str]
        List of marginal distribution types, one per asset.
        Supported: "normal", "t".
    n_samples : int
        Number of simulated samples to generate.
    data : DataFrame or ndarray
        Historical returns matrix (shape: [n_obs, n_assets]).
        Used to estimate marginal parameters and correlation.
    corr_method : {"spearman", "pearson"}, default "spearman"
        Method to estimate correlation matrix from data.
        Spearman is more robust to outliers.
    random_state : int or Generator or None, optional
        Random seed or NumPy random generator.

    Returns
    -------
    samples : np.ndarray
        Simulated joint samples (shape: [n_samples, n_assets]),
        each column follows its fitted marginal distribution.
    R : np.ndarray
        Estimated correlation matrix used in the Copula.
    params : list[dict]
        Fitted parameters of each marginal (mu, sigma, df if applicable).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(data, dtype=float)
    assert X.shape[1] == n_assets, "n_assets must match number of columns in data"
    assert len(dist_types) == n_assets, "dist_types length must equal n_assets"

    # ---- 1) Fit marginal distributions automatically ----
    marginals = []
    for j, dist_name in enumerate(dist_types):
        x = X[:, j]
        name = dist_name.lower()

        if name == "normal":
            mu, sigma = np.mean(x), np.std(x, ddof=1)
            F = lambda v, mu=mu, sigma=sigma: st.norm.cdf(v, mu, sigma)
            Finv = lambda u, mu=mu, sigma=sigma: st.norm.ppf(u, mu, sigma)
            params = {"mu": mu, "sigma": sigma}

        elif name == "t":
            df_hat, mu, sigma = st.t.fit(x)
            F = lambda v, df=df_hat, mu=mu, sigma=sigma: st.t.cdf((v - mu) / sigma, df)
            Finv = lambda u, df=df_hat, mu=mu, sigma=sigma: mu + sigma * st.t.ppf(u, df)
            params = {"mu": mu, "sigma": sigma, "df": df_hat}

        else:
            raise ValueError(f"Unsupported marginal type: {name}")

        marginals.append({"F": F, "Finv": Finv, "params": params})

    # ---- 2) Estimate correlation matrix R ----
    U = np.column_stack([marginals[j]["F"](X[:, j]) for j in range(n_assets)])

    if corr_method.lower() == "spearman":
        res = st.spearmanr(U, axis=0)
        rho = getattr(res, "correlation", res[0])

        # If it's a scalar (e.g., 2 assets), expand to 2×2 matrix
        if np.isscalar(rho):
            R = np.array([[1.0, rho],
                        [rho, 1.0]])
        else:
            R = np.asarray(rho, dtype=float)

    elif corr_method.lower() == "pearson":
        Z = st.norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
        R = np.corrcoef(Z, rowvar=False).astype(float)

    else:
        raise ValueError("corr_method must be 'spearman' or 'pearson'")

    R = near_psd_correlation(pd.DataFrame(R)).values

    # ---- 3) Sample from multivariate normal in Z-space ----
    Z = st.multivariate_normal.rvs(mean=np.zeros(n_assets), cov=R, size=n_samples, random_state=rng)
    if Z.ndim == 1:
        Z = Z[None, :]

    # ---- 4) Map to uniform, then to marginal space ----
    U_sim = st.norm.cdf(Z)
    samples = np.column_stack([marginals[j]["Finv"](U_sim[:, j]) for j in range(n_assets)])

    params_list = [m["params"] for m in marginals]
    return samples, R, params_list


def portfolio_var_es_sim(prices, holdings, returns, alpha = 0.05):
    """
    Compute VaR and ES for each asset and the total portfolio based on simulated returns.

    Parameters
    ----------
    prices : array-like
        Current prices of each asset.
    holdings : array-like
        Holdings (number of shares or units) of each asset.
    returns : np.ndarray
        Simulated or historical returns, shape = (n_samples, n_assets).
    alpha : float, default 0.05
        Tail probability (e.g. 0.05 for 95% confidence level).

    Returns
    -------
    out : pd.DataFrame
        Table containing VaR95, ES95 (monetary and percentage) for each asset and total.
    """
    prices   = np.asarray(prices, dtype=float)
    holdings = np.asarray(holdings, dtype=float)
    samples  = np.asarray(returns, dtype=float)
    n_assets = samples.shape[1]

    values0   = prices * holdings
    V0_total  = values0.sum()

    # Simulated prices and portfolio values
    prices_sim = (1.0 + samples) * prices
    values_sim = prices_sim * holdings
    pnl_assets = values_sim - values0
    pnl_total  = pnl_assets.sum(axis=1)

    # Helper function: VaR & ES
    def var_es(x, alpha):
        q = np.quantile(x, alpha)
        es = x[x <= q].mean()
        return -q, -es  # positive losses

    # Compute per-asset results
    rows = []
    for i in range(n_assets):
        VaR, ES = var_es(pnl_assets[:, i], alpha)
        rows.append([
            f"Asset_{i+1}",
            VaR, ES,
            VaR / values0[i],
            ES / values0[i]
        ])

    # Portfolio total
    VaR_tot, ES_tot = var_es(pnl_total, alpha)
    rows.append([
        "Total",
        VaR_tot, ES_tot,
        VaR_tot / V0_total,
        ES_tot / V0_total
    ])

    out = pd.DataFrame(rows, columns=["Stock", "VaR", "ES", "VaR_Pct", "ES_Pct"])
    return out