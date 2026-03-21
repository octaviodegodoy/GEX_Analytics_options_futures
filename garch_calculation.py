import numpy as np
import pandas as pd
from arch import arch_model

def calculate_garch_volatility(series, p=1, q=1, o=0, dist='normal', annualize=True, trading_days=252):
    """
    Fit a GARCH-family model to a return series and return the conditional volatility.

    Parameters
    ----------
    series : array-like
        Time series of returns (e.g., log returns).
    p : int
        Order of the GARCH terms (default=1).
    q : int
        Order of the ARCH terms (default=1).
    o : int
        Order of the asymmetric term (default=0, for GARCH use 0, for GJR-GARCH/AGARCH use 1).
    dist : str
        Distribution for innovations ('normal', 't', etc.).
    annualize : bool
        Whether to annualize the volatility (default=True).
    trading_days : int
        Number of trading days in a year (default=252).
    model_type : str
        Type of GARCH model: 'GARCH', 'GJR-GARCH' (AGARCH), or 'EGARCH'.

    Returns
    -------
    pd.Series
        Conditional volatility (annualized if specified).
    arch_model.Result
        Fitted model result object.
    """
    def _get_model(series, p, q, o, dist, model_type):
        if model_type.upper() == 'GARCH':
            return arch_model(series, p=p, q=q, o=0, dist=dist, rescale=False)
        elif model_type.upper() in ['GJR-GARCH', 'AGARCH']:
            return arch_model(series, p=p, q=q, o=1, dist=dist, rescale=False)
        elif model_type.upper() == 'EGARCH':
            return arch_model(series, p=p, q=q, o=0, dist=dist, rescale=False, vol='EGARCH')
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    series = pd.Series(series).dropna()
    model_type = locals().get('model_type', 'GARCH')
    am = _get_model(series, p, q, o, dist, model_type)
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility
    if annualize:
        cond_vol = cond_vol * np.sqrt(trading_days)
    return cond_vol, res
