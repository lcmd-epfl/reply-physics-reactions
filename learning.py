import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, KFold

def predict_KRR(X_train, X_test, y_train, y_test, sigma=100, l2reg=1e-6):
    g_gauss = 1.0 / (2 * sigma ** 2)

    K = rbf_kernel(X_train, X_train, gamma=g_gauss)
    K[np.diag_indices_from(K)] += l2reg
    alpha = np.dot(np.linalg.inv(K), y_train)
    K_test = rbf_kernel(X_test, X_train, gamma=g_gauss)

    y_pred = np.dot(K_test, alpha)
    mae = np.mean(np.abs(y_test - y_pred))
    return mae, y_pred


def opt_hyperparams(
    X_train, X_val, y_train, y_val,
    sigmas=[1, 10, 100, 1000],
    l2regs=[1e-10, 1e-7, 1e-4],
):
    maes = np.zeros((len(sigmas), len(l2regs)))

    for j, sigma in enumerate(sigmas):
        for k, l2reg in enumerate(l2regs):
            mae, y_pred = predict_KRR(
                X_train, X_val, y_train, y_val, sigma=sigma, l2reg=l2reg)
            print(f"MAE {mae} for sigma {sigma} and l2reg {l2reg}")
            maes[j, k] = mae

    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]

    print(
        "min mae",
        maes[min_j, min_k],
        "for sigma=",
        min_sigma,
        "and l2reg=",
        min_l2reg,
    )
    return min_sigma, min_l2reg

def KFold_MAE(X, y, folds=10):
    kf = KFold(n_splits=int(folds/2), random_state=2, shuffle=True)
    maes = []
    for i, (tr_idx, val_te_idx) in enumerate(kf.split(y)):
        print(f"CV split {i + 1} / {int(folds/2)}")
        X_tr = X[tr_idx]
        y_tr = y[tr_idx]

        print("First te/val split")
        # further split into val indices
        len_te = int(len(val_te_idx) / 2)
        te_idx = val_te_idx[:len_te]
        val_idx = val_te_idx[len_te:]
        print(f"{len(y)} total dataset, {len(tr_idx)} train, {len(te_idx)} test, {len(val_idx)} val")

        X_te = X[te_idx]
        X_val = X[val_idx]

        y_te = y[te_idx]
        y_val = y[val_idx]

        print("Optimising hypers...")
        sigma, l2reg = opt_hyperparams(X_tr, X_val, y_tr, y_val, l2regs=[1e-10, 1e-7, 1e-4], sigmas=[10, 100, 1e3])
        print(f"Opt hypers sigma={sigma} and l2reg={l2reg}")
        mae, y_pred = predict_KRR(X_tr, X_te, y_tr, y_te, sigma=sigma, l2reg=l2reg)
        maes.append(mae)

        print("MAE", mae)

        print("Second te/val split")
        # further split into val indices
        te_idx = val_te_idx[len_te:]
        val_idx = val_te_idx[:len_te]
        print(f"{len(tr_idx)} training points, {len(te_idx)} test, {len(val_idx)} val")

        X_te = X[te_idx]
        X_val = X[val_idx]

        y_te = y[te_idx]
        y_val = y[val_idx]

        sigma, l2reg = opt_hyperparams(X_tr, X_val, y_tr, y_val, l2regs=[1e-10, 1e-7, 1e-4], sigmas=[10, 100, 1e3])
        print(f"Opt hypers sigma={sigma} and l2reg={l2reg}")

        mae, y_pred = predict_KRR(X_tr, X_te, y_tr, y_te, sigma=sigma, l2reg=l2reg)
        maes.append(mae)

        print("MAE", mae)
    return maes