import numpy as np
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
import aida_cliport


def sag(
    u,
    r,
    k,
    s_des=0.9,
    p_rand=0.0,
    u_normalization=True,
    impute=True,
    n_min=1,
    n_rep=10,
):
    """
    Get the threshold for the uncertainty measure that achieves a desired sensitivity.
    :param u: list of uncertainties
    :param r: list of teacher rewards
    :param k: list of model update counts
    :param s_des: desired sensitivity
    :param p_rand: rate of random queries
    :param u_normalization: normalize for shifting uncertainty because of model updates
    :param impute: impute missing labels
    :param n_min: minimum number of negative samples
    :param n_rep: number of repetitions for imputation
    """
    window = 0
    u_array = np.array(u, dtype=float)
    r_array = np.asarray(r, dtype=int)
    k_array = np.asarray(k, dtype=int)

    window_idx = np.logical_and(r_array > -2, k_array >= k_array[-1] - window)
    u_window = u_array[window_idx]
    r_window = r_array[window_idx]
    k_window = k_array[window_idx]
    known = np.logical_not(r_window == 0)

    while np.sum(r_window == aida_cliport.KNOWN_FAILURE) < n_min or np.sum(r_window == aida_cliport.KNOWN_SUCCESS) < 1:
        if np.sum(window_idx) < window:
            # we should have at least n_min positive and n_min negative samples
            return np.nanmin(u_array)
        else:
            window += 1
            window_idx = np.logical_and(r_array > -2, k_array >= k_array[-1] - window)
            u_window = u_array[window_idx]
            r_window = r_array[window_idx]
            k_window = k_array[window_idx]
            known = np.logical_not(r_window == 0)
    if u_normalization:
        # compensate for decreasing uncertainty
        # fit a linear model to the uncertainty
        x = k_window
        y = u_window
        uncertainty_mean = LinearRegression().fit(x.reshape(-1, 1), y)
        u_window = (
            u_window
            - uncertainty_mean.predict(x.reshape(-1, 1))
            + uncertainty_mean.predict(np.array([k_window[-1]]).reshape(-1, 1))
        )

    if impute:
        failures = -r_window.copy()
        unknown = np.logical_not(known)
        y = failures[known]  #  We negate the labels to match the sensitivity convention
        X = u_window[known].reshape(-1, 1)
        clf = LogisticRegression(penalty=None).fit(X, y)
        probas = clf.predict_proba(u_window[unknown].reshape(-1, 1))
        gammas = []
        for _ in range(n_rep):
            failures[unknown] = np.asarray(probas[:, 1] > np.random.rand(probas.shape[0]), dtype="int") * 2 - 1
            _, tpr, threshs = roc_curve(failures, u_window, pos_label=1)
            fnr = 1 - tpr
            gamma = np.interp(s_des, tpr + p_rand * fnr, threshs)
            gammas.append(gamma)
        gamma = np.median(gammas)
    else:
        _, tpr, threshs = roc_curve(-r_window, u_window, pos_label=1)
        fnr = 1 - tpr
        gamma = np.interp(s_des, tpr + p_rand * fnr, threshs)
    if np.isnan(gamma):
        gamma = np.nanmin(u_window)
    return gamma
