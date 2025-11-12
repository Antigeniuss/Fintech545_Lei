import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import minimize, OptimizeResult


# ---------- helpers ----------
def _symm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _to_corr_from_cov(S: np.ndarray):
    std = np.sqrt(np.diag(S))
    std = np.where(std <= 0.0, 1.0, std)
    D_inv = np.diag(1.0 / std)
    R = _symm(D_inv @ S @ D_inv)
    np.fill_diagonal(R, 1.0)
    return R, std

def _to_cov_from_corr(R: np.ndarray, std: np.ndarray):
    R = _symm(np.asarray(R, dtype=float))
    np.fill_diagonal(R, 1.0)
    D = np.diag(std)
    S = D @ R @ D
    return _symm(S)

def calculate_cov(X: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix (unbiased, denominator = n - 1).

    Parameters
    ----------
    X : np.ndarray, shape = (n_samples, n_features)
        Each row represents one observation (e.g., one time point or sample)
        Each column represents one variable / asset (e.g., returns of a stock)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    return _symm((Xc.T @ Xc) / (X.shape[0] - 1))


def fit_normal(data):
    data = np.asarray(data).ravel()
    mu = data.mean()
    sigma = data.std(ddof=1) 
    return mu, sigma

# ---------- Rebonato–Jäckel (near-PSD) ----------
def near_psd_correlation(R_df: pd.DataFrame, eps: float = 0.0) -> pd.DataFrame:
    C = _symm(R_df.values)
    np.fill_diagonal(C, 1.0)
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
    d = np.sqrt(np.clip(np.diag(C_hat), 1e-15, None))
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
    np.fill_diagonal(X, 1.0)
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
    Simulate samples from a multivariate normal distribution X ~ N(mean, cov).

    Parameters
    ----------
    mean : float or array-like of shape (d,)
        Mean vector of the distribution. If a scalar is given, it's broadcast to all dimensions.
        d = number of variables (features).
    cov : array-like of shape (d, d)
        Covariance matrix. Must be symmetric and positive semi-definite.
    n_samples : int, default=100_000
        Number of samples to generate (number of rows in the output).
    seed : int, default=42
        Random seed for reproducibility.
    tol : float, default=1e-10
        Tolerance for eigenvalue clipping in near-PSD handling.
    return_info : bool, default=False
        If True, also return diagnostic information.

    Returns
    -------
    X : np.ndarray of shape (n_samples, d)
        Simulated samples where:
          - Each **row** represents one observation (sample draw)
          - Each **column** represents one variable / dimension.
    info : dict (optional)
        Contains:
            - "factorization": method used ("chol" or "eigh-clip")
            - "pd_check": definiteness check result
            - "tol": tolerance used
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
    if np.isscalar(mean):
        mu = np.full((1, d), float(mean))
    else:
        mu = np.asarray(mean, dtype=float).reshape(1, -1)
        if mu.shape[1] != d:
            raise ValueError("mean length must equal cov dimension.")
    X = Z @ L.T + mu

    if return_info:
        info = {"factorization": used, "pd_check": chk, "tol": tol}
        return X, info
    return X


# --- PCA reduce covariance to >= 99% explained ---
def pca_covariance(cov_df: pd.DataFrame, threshold: float = 0.99):
    """Return (PCA-approximated covariance, k, cum_explained array)."""
    A = np.asarray(cov_df.values, dtype=float)
    A = _symm(A)            # enforce symmetry

    # eigen-decomposition (for symmetric covariance)
    w, V = np.linalg.eigh(A)                    # w ascending
    idx = np.argsort(w)[::-1]                   # sort descending
    w, V = w[idx], V[:, idx]

    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        # 退化情形：全零方差
        return cov_df.copy()*0.0, 0, np.zeros_like(w)

    explained_ratio = w / s
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
        nu, mu, sigma = st.t.fit(r)
        t_alpha = st.t.ppf(alpha, nu)
        var_quantile = mu + t_alpha * sigma
    
    else:
        raise ValueError("dist must be 'normal' or 't'")
    
    return {
        "VaR Absolute(distance from 0)": abs(var_quantile),
        # "VaR Diff from Mean": mu - var_quantile
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
    """
    r = returns.squeeze().dropna().astype(float)
    nu, loc, scale = st.t.fit(r.values) 

    # Monte Carlo simulation
    rng = np.random.default_rng(seed)
    sims = loc + scale * rng.standard_t(nu, size=n_samples)

    var_q = np.quantile(sims, alpha)

    return {
        "VaR Absolute(distance from 0)": abs(var_q),
        # "VaR Diff from Mean": sims.mean() - var_q,
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
        "ES Absolute(distance from 0)": abs(es_),
        # "ES Diff from Mean": returns.mean() - es_,
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
        "ES Absolute(distance from 0)": abs(es_),
        # "ES Diff from Mean": returns.mean() - es_,
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

def ew_cov_corr_normalized(df: pd.DataFrame, lam: float = 0.97, window: int | None = None):
    X = df.to_numpy(copy=False)
    if window is not None:
        X = X[-window:, :]
    n, k = X.shape

    # normalized weights: newest obs has exponent 0
    exponents = np.arange(n-1, -1, -1, dtype=float)
    w_raw = (1.0 - lam) * (lam ** exponents)
    w = w_raw / w_raw.sum()
    w = w.reshape(-1, 1)

    # mean over the same window (can switch to simple mean if needed)
    mu = (w.T @ X) / w.sum()
    # mu = X.mean(axis=0, keepdims=True)   # simple mean alternative

    XC = X - mu
    cov = (XC.T * w.ravel()) @ XC

    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / denom
    corr[np.isnan(corr)] = 0.0

    cov_df = pd.DataFrame(cov, index=df.columns, columns=df.columns)
    corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)
    return cov_df, corr_df

def pairwise_cov_corr(df):
    cols = df.columns
    n = len(cols)
    cov = pd.DataFrame(index=cols, columns=cols, dtype=float)
    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for i in range(n):
        for j in range(n):
            x = df.iloc[:, i]
            y = df.iloc[:, j]
            valid = x.notna() & y.notna()
            if valid.sum() > 1:
                cov.iloc[i, j] = x[valid].cov(y[valid])
                corr.iloc[i, j] = x[valid].corr(y[valid], method="pearson")
            else:
                cov.iloc[i, j] = None 
                corr.iloc[i, j] = None
    return cov,corr


def bs_european_greeks(S, K, T, r, q, sigma, option_type='call'):
    """
    Compute European option price and Greeks under the
    Black-Scholes-Merton (BSM) model with continuous dividend yield q.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free rate (continuous compounding).
    q : float
        Continuous dividend yield.
    sigma : float
        Annual volatility.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    dict
        {'Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'}
    """

    # Ensure numeric inputs
    S, K, T, r, q, sigma = map(float, [S, K, T, r, q, sigma])

    # Core BSM components
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * np.exp(-q * T) * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
        delta = np.exp(-q * T) * st.norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * st.norm.cdf(d2)
        theta = (-S * np.exp(-q * T) * st.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * st.norm.cdf(d2)
                 + q * S * np.exp(-q * T) * st.norm.cdf(d1))
    else:
        price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S * np.exp(-q * T) * st.norm.cdf(-d1)
        delta = np.exp(-q * T) * (st.norm.cdf(d1) - 1)
        rho = -K * T * np.exp(-r * T) * st.norm.cdf(-d2)
        theta = (-S * np.exp(-q * T) * st.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * st.norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * st.norm.cdf(-d1))

    gamma = np.exp(-q * T) * st.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * st.norm.pdf(d1) * np.sqrt(T)

    return {
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }


def bt_american_continuous_div(S, K, T, r, q, sigma, steps, option_type='call'):
    """
    American Option Binomial Tree Model with Continuous Dividend Yield (q).
    Uses the standard recombining tree approach (Cox-Ross-Rubinstein/Jarrow-Rudd type).
    
    Parameters
    ----------
    S : float         # Current underlying asset price
    K : float         # Strike price
    T : float         # Time to maturity (in years)
    r : float         # Risk-free interest rate (continuous compounding)
    q : float         # Continuous dividend yield
    sigma : float     # Annual volatility
    steps : int       # Number of time steps in the binomial tree
    option_type : str # 'call' or 'put'
    
    Returns
    -------
    float : American option price at t=0
    """
    
    # --- 1. Parameter Setup ---
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability (using r - q as the cost of carry)
    p = (np.exp((r - q) * dt) - d) / (u - d)     
    disc = np.exp(-r * dt)
    
    # Determine option type factor (z=1 for call, z=-1 for put)
    z = 1 if option_type.lower() == 'call' else -1

    # --- 2. Initialization ---
    
    # V is a 2D array to store option values at each node. 
    # V[i, j] is the option value at time step j, with i up-moves.
    V = np.zeros((steps + 1, steps + 1))
    
    # Calculate stock prices at maturity (j = steps)
    # ST[i] = S * u^i * d^(steps - i)
    ST = S * (u ** np.arange(steps + 1)) * (d ** np.arange(steps, -1, -1))
    
    # Calculate Payoff at maturity (j = steps)
    V[:, steps] = np.maximum(0, z * (ST - K))
    
    # --- 3. Backward Induction ---
    
    # Loop backwards from the last time step to t=0
    for j in range(steps - 1, -1, -1):
        
        # Calculate Continuation Value (Expected discounted value of holding the option)
        # Use vectorized operations on the V-values from the next time step (j+1)
        continuation_value = disc * (p * V[1:j+2, j+1] + (1 - p) * V[0:j+1, j+1])
        
        # Calculate Stock Prices at current time step (j)
        Sj = S * (u ** np.arange(j + 1)) * (d ** np.arange(j, -1, -1))
        
        # Calculate Immediate Exercise Value (Intrinsic Value)
        intrinsic_value = np.maximum(0, z * (Sj - K))
        
        # American Option Check: V[j] = max(Intrinsic Value, Continuation Value)
        V[0:j+1, j] = np.maximum(continuation_value, intrinsic_value)

    # Return the option price at t=0 (V[0, 0])
    return V[0, 0]


def american_binomial_with_greeks(S, K, T, r, q, sigma, steps=200, option_type='call'):
    """
    Computes American option price and Greeks (Delta, Gamma, Vega, Rho, Theta)
    using the bt_american_continuous_div function and central finite-difference 
    approximations.

    Parameters
    ----------
    S : float
        Current underlying price.
    ... (other parameters similar to the pricing function)

    Returns
    -------
    dict : { 'Price', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho' }
    """

    # --- Small perturbations for finite difference ---
    dS = 0.01 * S       # Perturbation for underlying price (Delta, Gamma)
    dSigma = 0.01 * sigma  # Perturbation for volatility (Vega)
    dR = 0.0001         # Perturbation for interest rate (Rho)
    dT = 1 / 365.0      # Perturbation for time (Theta), typically one day

    # --- Base price calculation ---
    base = bt_american_continuous_div(S, K, T, r, q, sigma, steps, option_type)

    # --- Price w.r.t. S (Delta & Gamma) ---
    up = bt_american_continuous_div(S + dS, K, T, r, q, sigma, steps, option_type)
    down = bt_american_continuous_div(S - dS, K, T, r, q, sigma, steps, option_type)
    
    # Central Difference for Delta
    delta = (up - down) / (2 * dS)
    # Second-order Central Difference for Gamma
    gamma = (up - 2 * base + down) / (dS ** 2)

    # --- Price w.r.t. volatility (Vega) ---
    vega_price = bt_american_continuous_div(S, K, T, r, q, sigma + dSigma, steps, option_type)
    # Forward Difference for Vega
    vega = (vega_price - base) / dSigma

    # --- Price w.r.t. interest rate (Rho) ---
    rho_price = bt_american_continuous_div(S, K, T, r + dR, q, sigma, steps, option_type)
    # Forward Difference for Rho
    rho = (rho_price - base) / dR

    # --- Price w.r.t. time (Theta) ---
    # Time decay is V(t+dt) - V(t) or V(T - dT) - V(T) 
    theta_price = bt_american_continuous_div(S, K, T - dT, r, q, sigma, steps, option_type)
    # Note: Theta is defined as -dV/dt, so we use a negative sign in the denominator: (V(T-dT) - V(T)) / (-dT)
    theta = (theta_price - base) / (-dT)

    return {
        'Price': base,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Rho': rho,
        'Theta': theta
    }


def bt_american_discrete_div(S, K, T, r, divAmts, divTimes, sigma, steps, option_type='call'):
    """
    American Option Binomial Tree Model with Discrete Cash Dividends (D).
    Uses a recursive approach to adjust the stock price at each dividend date.

    The model assumes the stock price drops by the dividend amount D at the ex-dividend date.

    Parameters
    ----------
    S : float                 # Current underlying asset price
    K : float                 # Strike price
    T : float                 # Time to maturity (in years)
    r : float                 # Risk-free interest rate
    divAmts : list/array      # Dividend amounts [D1, D2, ...]
    divTimes : list/array     # Dividend dates (as time steps, e.g., [50, 100, ...])
    sigma : float             # Annual volatility
    steps : int               # Total number of time steps (N)
    option_type : str         # 'call' or 'put'

    Returns
    -------
    float : American option price at t=0
    """
    
    # Convert inputs to numpy arrays for consistency
    divAmts = np.array(divAmts)
    divTimes = np.array(divTimes)
    
    # --- 1. Base Case / Boundary Check ---
    
    # If no more dividends or next dividend is outside the current grid, 
    # revert to the standard American option price (with continuous dividend q=0, so b=r).
    if divAmts.size == 0 or (divTimes.size > 0 and divTimes[0] > steps):
        # We use q=0.0 because the discrete dividends are already accounted for.
        return bt_american_continuous_div(S, K, T, r, q=0.0, sigma=sigma, steps=steps, option_type=option_type)
        
    # --- 2. Parameter Setup for the current segment (up to the first dividend) ---
    
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Risk-neutral probability uses only 'r' for drift, as dividends are handled separately
    p = (np.exp(r * dt) - d) / (u - d)     
    disc = np.exp(-r * dt)
    
    z = 1 if option_type.lower() == 'call' else -1
    
    # --- 3. Grid Setup up to the First Dividend Date ---
    
    N_div = divTimes[0] # Time step of the first dividend (j)
    
    # V array stores values from t=0 up to the first dividend date N_div
    V = np.zeros((N_div + 1, N_div + 1))
    
    # --- 4. Backward Induction (from N_div back to t=0) ---

    for j in range(N_div, -1, -1):
        # Sj: Stock prices at current time step j
        Sj = S * (u ** np.arange(j + 1)) * (d ** np.arange(j, -1, -1))
        
        if j < N_div:
            # --- a) Normal Time Step (Before Dividend Date) ---
            
            # Continuation Value
            continuation_value = disc * (p * V[1:j+2, j+1] + (1 - p) * V[0:j+1, j+1])
            
            # Intrinsic Value
            intrinsic_value = np.maximum(0, z * (Sj - K))
            
            # American Option Check: max(Continuation, Intrinsic)
            V[0:j+1, j] = np.maximum(continuation_value, intrinsic_value)
            
        else: # j == N_div (First Dividend Date)
            # --- b) Dividend Date Node (Recursive Call) ---
            
            # Iterate through all nodes 'i' at the dividend time j
            for i in range(j + 1):
                price = Sj[i]
                
                # 1. Exercise Value: Option exercised *before* the dividend is paid
                val_exercise = np.maximum(0, z * (price - K))
                
                # 2. Continuation Value: Option held *through* the dividend payment
                
                # Stock price adjusts down by the dividend amount D1 (price - D1)
                S_adj = price - divAmts[0]
                
                # Remaining parameters for the recursive call
                T_rem = T - N_div * dt
                N_rem = steps - N_div
                
                # Recursive call to price the remaining option (S_adj is the new S)
                # Subsequent dividend times are adjusted relative to the new start time
                val_no_exercise = bt_american_discrete_div(
                    S_adj, K, T_rem, r, 
                    divAmts[1:], divTimes[1:] - N_div, # Slice the arrays and adjust times
                    sigma, N_rem, option_type
                )
                
                # American Check: max(Exercise before dividend, Continuation after dividend)
                V[i, j] = np.maximum(val_no_exercise, val_exercise)

    # Return the option price at t=0
    return V[0, 0]


def risk_parity_csd(cov: np.ndarray, budget: np.ndarray | None = None) -> np.ndarray:
    """
    Compute Risk Parity portfolio weights using the CSD-based objective.

    Objective:
        min SSE = Σ_i (CSD_i* - mean(CSD*))²
        where CSD_i* = (w_i * (Σw)_i / σ_p) / b_i
        and σ_p = sqrt(wᵀΣw)

    Constraints:
        Σ_i w_i = 1, w_i ≥ 0

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of asset returns, shape (n, n).
    budget : np.ndarray | None
        Target risk budget vector (b_i), shape (n,).
        If None, uses equal budgets (1/n).

    Returns
    -------
    w : np.ndarray
        Risk parity weights satisfying the CSD-based equal risk contribution condition.
    """
    n = cov.shape[0]
    if budget is None:
        budget = np.ones(n) / n
    else:
        budget = budget / np.sum(budget)

    # --- Helper functions ---
    def portfolio_vol(w):
        return np.sqrt(np.dot(w, cov @ w))

    def csd_star(w):
        sigma_p = portfolio_vol(w)
        csd = w * (cov @ w) / sigma_p      # element-wise
        csd_star = csd / budget            # normalize by risk budget
        return csd_star

    # --- Objective: SSE of normalized CSDs ---
    def objective(w):
        csd_s = csd_star(w)
        return np.sum((csd_s - csd_s.mean())**2)

    # --- Constraints ---
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    res = minimize(objective, w0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol': 1e-12, 'maxiter': 5000})

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return res.x


def max_sharpe_ratio(mu, cov, rf=0.0, bounds=None, long_only=True):
    """
    Maximize Sharpe ratio under normal assumption.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns (n, )
    cov : np.ndarray
        Covariance matrix (n, n)
    rf : float
        Risk-free rate
    bounds : list of (float, float)
        Bounds for weights. Example: [(0.1, 0.5)] * n
    long_only : bool
        If True and no bounds given, default bounds (0, 1)
        If False, allow shorting with bounds (-1, 1)

    Returns
    -------
    w : np.ndarray
        Portfolio weights that maximize Sharpe ratio.
    sharpe : float
        Maximum Sharpe ratio achieved.
    """

    n = len(mu)
    if bounds is None:
        if long_only:
            bounds = [(0, 1)] * n
        else:
            bounds = [(-1, 1)] * n

    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_return - rf) / port_vol

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    w0 = np.ones(n) / n

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons, method='SLSQP')
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w = res.x
    sharpe = -(res.fun)
    return w, sharpe


def asset_attribution(returns: pd.DataFrame, init_weights: np.ndarray) -> pd.DataFrame:
    """
    Geometric return attribution (with dynamic weights) + risk attribution via regression.

    Math (per your slides)
    ----------------------
    Weight evolution (no rebalancing):
        w*_i,t = w_i,t * (1 + r_i,t)
        R_t    = sum_i w*_i,t - 1 = w_t · r_t
        w_i,t+1 = w*_i,t / (1 + R_t)

    Geometric return attribution:
        R      = prod_t (1 + R_t) - 1
        GR     = ln(1 + R)
        K      = GR / R          (if R != 0 else 1)
        k_t    = ln(1 + R_t) / (K * R_t)  (safe for R_t≈0)
        A_i    = sum_t k_t * w_i,t * r_i,t

    Risk attribution via regression (OLS):
        a_i,t  = w_i,t * r_i,t           (weighted asset return series)
        p_t    = R_t                     (portfolio return series)
        beta_i = Cov(a_i, p) / Var(p)
        RA_i   = sigma_p * beta_i = Cov(a_i, p) / sigma_p
        where sigma_p = std(p_t)

    Parameters
    ----------
    returns : pd.DataFrame, shape (T, N)
        Asset return time series (columns = assets).
    init_weights : np.ndarray, shape (N,)
        Initial portfolio weights.

    Returns
    -------
    pd.DataFrame
        Three rows:
          1) 'TotalReturn'                per-asset arithmetic totals + portfolio total
          2) 'Geometric Attribution'      per-asset geometric attribution + portfolio geometric total
          3) 'Risk Attribution (beta·σp)' per-asset regression-based risk attribution; portfolio cell = σ_p
    """
    # ---- prep ----
    n_assets = returns.shape[1]
    T = returns.shape[0]
    cols = list(returns.columns)

    w = np.asarray(init_weights, dtype=float).copy()
    w /= w.sum()

    # store evolving weights and returns
    Wt = np.zeros((T, n_assets))      # weights at start of each period
    Rt = np.zeros(T)                  # portfolio return each period

    # ---- evolve weights through time (no rebalancing) ----
    for t in range(T):
        r = returns.iloc[t].values
        Wt[t] = w
        Rt[t] = float(w @ r)
        w_star = w * (1 + r)
        w = w_star / w_star.sum()

    # ---- portfolio totals (arithmetic & geometric) ----
    R_total = float(np.prod(1 + Rt) - 1)
    GR_total = float(np.log1p(R_total))
    K = GR_total / R_total if R_total != 0 else 1.0

    # k_t scaling (handle Rt≈0 safely)
    kt = np.zeros(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        kt = np.log1p(Rt) / (K * Rt)
    kt[~np.isfinite(kt)] = 0.0

    # ---- geometric attribution per asset ----
    A = np.zeros(n_assets)
    for t in range(T):
        A += kt[t] * Wt[t] * returns.iloc[t].values
    geo_port_total = float(A.sum())   # ~ GR_total

    # ---- arithmetic total return per asset (for row 1) ----
    total_ret_assets = (1 + returns).prod().values - 1.0

    # ---- risk attribution via regression ----
    # a_i,t = w_i,t * r_i,t ; p_t = Rt
    sigma_p = float(np.std(Rt, ddof=1)) if T > 1 else 0.0
    RA = np.zeros(n_assets)
    if sigma_p > 0:
        p = Rt - Rt.mean()
        var_p = float(np.var(Rt, ddof=1))
        for i in range(n_assets):
            a_i = Wt[:, i] * returns.iloc[:, i].values
            a_i_c = a_i - a_i.mean()
            cov_ip = float(np.dot(a_i_c, p) / (T - 1)) if T > 1 else 0.0
            beta_ip = cov_ip / var_p if var_p > 0 else 0.0
            RA[i] = sigma_p * beta_ip
    else:
        RA[:] = 0.0

    # ---- assemble output ----
    out = pd.DataFrame({"Value": [
        "TotalReturn",
        "Return Attribution",
        "Vol Attribution (beta·σp)"
    ]})

    for j, name in enumerate(cols):
        out[name] = [total_ret_assets[j], A[j], RA[j]]

    out["Portfolio"] = [R_total, geo_port_total, sigma_p]
    return out


import numpy as np
import pandas as pd

def factor_attribution(
    asset_returns: pd.DataFrame,     # (T × N) Asset return time series
    betas: pd.DataFrame,             # (N × K) Asset–factor exposures
    init_weights: np.ndarray,        # (N,) Initial portfolio weights
    factor_returns: pd.DataFrame,    # (T × K) Factor return time series
    include_alpha_risk: bool = True  # Whether to include alpha in risk attribution
) -> pd.DataFrame:
    """
    Perform dynamic-weight geometric factor return attribution and risk attribution.

    The method combines:
        1) Dynamic weight updates (buy-and-hold drift, no rebalancing)
        2) Geometric return attribution for each factor and alpha
        3) Regression-based risk attribution (Cov/σ_p)

    References
    ----------
    Portfolio weight evolution:
        w*_i,t = w_i,t * (1 + r_i,t)
        R_t    = Σ_i w*_i,t - 1 = w_t · r_t
        w_i,t+1 = w*_i,t / (1 + R_t)

    Portfolio factor exposure:
        B_t = Σ_i w_i,t * β_i,·   (1×K vector per period)

    Geometric attribution scaling:
        R  = Π_t (1 + R_t) - 1
        GR = ln(1 + R)
        K  = GR / R
        k_t = ln(1 + R_t) / (K * R_t)

    Factor and alpha return attribution:
        A_f = Σ_t k_t * B_{t,f} * F_{t,f}
        α_t = R_t − Σ_f B_{t,f} F_{t,f}
        A_α = Σ_t k_t * α_t

    Risk attribution (OLS-based):
        a_{f,t} = B_{t,f} * F_{t,f}
        RA_f = Cov(a_f, R) / σ_p
        σ_p = std(R_t)
        RA_α = Cov(α, R) / σ_p  (optional)
    """

    # -------------------------------
    # 1. Align data shapes and columns
    # -------------------------------
    assets = list(asset_returns.columns)
    betas = betas.reindex(index=assets)          # align rows by asset order
    factors = list(factor_returns.columns)
    betas = betas[factors]                       # ensure column order matches factors

    T, N = asset_returns.shape
    K = len(factors)

    # -------------------------------
    # 2. Initialize weights
    # -------------------------------
    w = np.asarray(init_weights, dtype=float).reshape(-1)
    if w.shape[0] != N:
        raise ValueError("init_weights length must match number of assets.")
    w /= w.sum()                                 # normalize to sum to 1

    # Arrays to store per-period weights and portfolio returns
    Wt = np.zeros((T, N))
    Rt = np.zeros(T)

    # -------------------------------
    # 3. Simulate dynamic weight drift
    # -------------------------------
    for t in range(T):
        r_t = asset_returns.iloc[t].values
        Wt[t] = w                                # record starting weights
        Rt[t] = float(w @ r_t)                   # portfolio return for period t
        w_star = w * (1.0 + r_t)                 # grow by asset return
        denom = w_star.sum()
        w = np.ones_like(w) / N if denom <= 0 else w_star / denom

    # -------------------------------
    # 4. Compute geometric scaling factors
    # -------------------------------
    R_total = float(np.prod(1.0 + Rt) - 1.0)
    GR_total = float(np.log1p(R_total))
    Kscale = GR_total / R_total if R_total != 0 else 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        kt = np.log1p(Rt) / (Kscale * Rt)
    kt[~np.isfinite(kt)] = 0.0                   # handle divide-by-zero

    # -------------------------------
    # 5. Compute portfolio factor exposures
    # -------------------------------
    # B_t = Σ_i w_i,t * β_i,·   (shape: T×K)
    beta_mat = betas.values
    Bts = np.zeros((T, K))
    for t in range(T):
        Bts[t] = Wt[t] @ beta_mat

    # -------------------------------
    # 6. Compute factor contributions and alpha
    # -------------------------------
    F = factor_returns.values                    # (T×K)
    a_ft = Bts * F                               # (T×K), factor contributions per period
    factor_part_t = a_ft.sum(axis=1)             # (T,), total factor part
    alpha_t = Rt - factor_part_t                 # (T,), residual (alpha)

    # -------------------------------
    # 7. Geometric return attribution
    # -------------------------------
    A_factors = (kt[:, None] * a_ft).sum(axis=0) # (K,), each factor’s contribution
    A_alpha = float((kt * alpha_t).sum())        # alpha contribution
    A_port_geo = float(A_factors.sum() + A_alpha)# portfolio geometric return

    # -------------------------------
    # 8. Regression-based risk attribution
    # -------------------------------
    sigma_p = float(np.std(Rt, ddof=1)) if T > 1 else 0.0
    RA_factors = np.zeros(K)
    RA_alpha = 0.0

    if sigma_p > 0:
        p_c = Rt - Rt.mean()
        var_p = float(np.var(Rt, ddof=1))

        # Factors
        for j in range(K):
            x = a_ft[:, j] - a_ft[:, j].mean()
            cov = float((x @ p_c) / (T - 1)) if T > 1 else 0.0
            beta = cov / var_p if var_p > 0 else 0.0
            RA_factors[j] = sigma_p * beta

        # Alpha
        if include_alpha_risk:
            x = alpha_t - alpha_t.mean()
            cov = float((x @ p_c) / (T - 1)) if T > 1 else 0.0
            beta = cov / var_p if var_p > 0 else 0.0
            RA_alpha = sigma_p * beta

    # -------------------------------
    # 9. Aggregate total returns
    # -------------------------------
    total_ret_factors = (1.0 + factor_returns).prod().values - 1.0
    total_ret_alpha = float((1.0 + alpha_t).prod() - 1.0)

    # -------------------------------
    # 10. Construct output table
    # -------------------------------
    out = pd.DataFrame({"Value": ["TotalReturn", "Return Attribution", "Vol Attribution"]})

    # Factor columns
    for j, f in enumerate(factors):
        out[f] = [total_ret_factors[j], A_factors[j], RA_factors[j]]

    # Alpha column
    out["Alpha"] = [total_ret_alpha, A_alpha, RA_alpha]

    # Portfolio column
    out["Portfolio"] = [R_total, A_port_geo, sigma_p]

    return out