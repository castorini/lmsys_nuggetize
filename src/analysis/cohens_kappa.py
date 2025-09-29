import numpy as np


def weight_matrix(
    k: int, scheme: str = "linear", convention: str = "disagreement"
) -> np.ndarray:
    """
    Create a k×k weight matrix.
      scheme: 'linear'   -> w_ij = |i - j| / (k - 1)
              'quadratic'-> w_ij = (i - j)^2 / (k - 1)^2
      convention: 'disagreement' (DATAtab) -> 0 on diagonal
                  'agreement'              -> 1 on diagonal
    """
    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if scheme == "linear":
                w = abs(i - j) / (k - 1)
            elif scheme == "quadratic":
                w = (i - j) ** 2 / ((k - 1) ** 2)
            else:
                raise ValueError("scheme must be 'linear' or 'quadratic'")
            W[i, j] = w
    if convention == "agreement":
        W = 1.0 - W
    elif convention != "disagreement":
        raise ValueError("convention must be 'disagreement' or 'agreement'")
    return W


def weighted_kappa(
    conf_mat: np.ndarray, scheme: str = "linear", convention: str = "disagreement"
):
    """
    Compute weighted Cohen's kappa for a multiclass confusion matrix.
    conf_mat: k×k np.array where conf_mat[i, j] is count of Human=i, Nugget=j.
    Returns dict with kappa and intermediate terms.
    """
    C = np.asarray(conf_mat, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("conf_mat must be a square (k×k) matrix")
    N = C.sum()
    k = C.shape[0]
    W = weight_matrix(k, scheme=scheme, convention=convention)

    # Observed weighted disagreement/agreement
    Qo = (W * C).sum() / N
    # Expected counts under independence
    row = C.sum(axis=1)
    col = C.sum(axis=0)
    E = np.outer(row, col) / N
    Qe = (W * E).sum() / N

    if convention == "disagreement":  # DATAtab
        kappa = 1.0 - Qo / Qe
        Po = 1.0 - Qo  # observed weighted agreement
        Pe = 1.0 - Qe  # expected weighted agreement
    else:  # agreement
        Po = Qo
        Pe = Qe
        kappa = (Po - Pe) / (1.0 - Pe)

    return {
        "kappa": float(kappa),
        "Po": float(Po),
        "Pe": float(Pe),
        "Qo": float(Qo),
        "Qe": float(Qe),
        "scheme": scheme,
        "convention": convention,
        "N": int(N),
        "weight_matrix": W,
    }


if __name__ == "__main__":
    configs = [
        {
            "cm": np.array([[982, 296, 436], [387, 292, 366], [485, 312, 964]]),
            "threshold": 0.05,
            "metric": "all_score",
        },
        {
            "cm": np.array([[958, 335, 421], [372, 320, 353], [458, 364, 939]]),
            "threshold": 0.06,
            "metric": "all_score",
        },
        {
            "cm": np.array([[938, 381, 395], [360, 342, 343], [443, 394, 924]]),
            "threshold": 0.07,
            "metric": "all_score",
        },
        {
            "cm": np.array([[907, 431, 376], [340, 373, 332], [422, 442, 897]]),
            "threshold": 0.08,
            "metric": "all_score",
        },
        {
            "cm": np.array([[883, 473, 358], [323, 406, 316], [394, 495, 872]]),
            "threshold": 0.09,
            "metric": "all_score",
        },
        {
            "cm": np.array([[845, 536, 333], [302, 447, 296], [365, 559, 837]]),
            "threshold": 0.1,
            "metric": "all_score",
        },
        {
            "cm": np.array([[830, 566, 318], [291, 467, 287], [354, 595, 812]]),
            "threshold": 0.11,
            "metric": "all_score",
        },
        {
            "cm": np.array([[804, 602, 308], [278, 491, 276], [335, 630, 796]]),
            "threshold": 0.12,
            "metric": "all_score",
        },
        {
            "cm": np.array([[779, 646, 289], [264, 519, 262], [320, 667, 774]]),
            "threshold": 0.13,
            "metric": "all_score",
        },
        {
            "cm": np.array([[763, 680, 271], [254, 538, 253], [307, 713, 741]]),
            "threshold": 0.14,
            "metric": "all_score",
        },
        {
            "cm": np.array([[744, 710, 260], [244, 558, 243], [294, 745, 722]]),
            "threshold": 0.15,
            "metric": "all_score",
        },
    ]

    for config in configs:
        M = config["cm"]
        print("Confusion matrix (Human × Nugget):\n", M.astype(int), "\n")
        for scheme in ("linear", "quadratic"):
            res = weighted_kappa(M, scheme=scheme, convention="disagreement")
            config["ck"] = res
            print(config)
            # print(f"{scheme.title()} weights (DATAtab disagreement convention):")
            # print(f"  N = {res['N']}")
            # print(f"  Qo (obs. disagreement)  = {res['Qo']:.4f}")
            # print(f"  Qe (exp. disagreement)  = {res['Qe']:.4f}")
            # print(f"  ==> Weighted kappa      = {res['kappa']:.3f}")
            # print(f"  (equivalently Po={res['Po']:.4f}, Pe={res['Pe']:.4f} with agreement convention)\n")
