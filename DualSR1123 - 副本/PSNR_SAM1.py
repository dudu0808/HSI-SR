import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


# =========================
# Helpers
# =========================
def img_2d_mat(x_true, x_pred):
    """
    HWC -> (C, H*W)
    """
    h, w, c = x_true.shape
    x_true = x_true.astype(np.float32)
    x_pred = x_pred.astype(np.float32)

    x_mat = np.reshape(np.transpose(x_true, (2, 0, 1)), (c, -1))
    y_mat = np.reshape(np.transpose(x_pred, (2, 0, 1)), (c, -1))
    return x_mat, y_mat


# =========================
# Metrics
# =========================
def compare_rmse(x_true, x_pred):
    x_true = x_true.astype(np.float32)
    x_pred = x_pred.astype(np.float32)
    return np.linalg.norm(x_true - x_pred) / (np.sqrt(x_true.size) + 1e-8)


def compare_mpsnr(x_true, x_pred, data_range):
    x_true = x_true.astype(np.float32)
    x_pred = x_pred.astype(np.float32)
    channels = x_true.shape[2]
    total_psnr = [
        compare_psnr(image_true=x_true[:, :, k], image_test=x_pred[:, :, k], data_range=data_range)
        for k in range(channels)
    ]
    return float(np.mean(total_psnr))


def compare_mssim(x_true, x_pred, data_range, multidimension=False):
    # 注意：skimage 的参数名是 channel_axis (新版本)；你这里按 band 逐通道算最稳
    mssim = [
        compare_ssim(im1=x_true[:, :, i], im2=x_pred[:, :, i], data_range=data_range)
        for i in range(x_true.shape[2])
    ]
    return float(np.mean(mssim))


def compare_ergas(x_true, x_pred, ratio):
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)  # (C, HW)
    sum_ergas = 0.0
    eps = 1e-12
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(err * err)
        mean_x = np.mean(vec_x)
        sum_ergas += r_mse / (mean_x * mean_x + eps)
    return float((100.0 / ratio) * np.sqrt(sum_ergas / x_true.shape[0]))


def compare_sam(x_true, x_pred):
    """
    Vectorized SAM (degrees).
    x_true/x_pred: HWC
    - ignore pixels with near-zero norm to avoid divide-by-zero
    """
    x_true = x_true.astype(np.float32)
    x_pred = x_pred.astype(np.float32)

    # (H*W, C)
    t = x_true.reshape(-1, x_true.shape[2])
    p = x_pred.reshape(-1, x_pred.shape[2])

    # norms
    nt = np.linalg.norm(t, axis=1)
    npred = np.linalg.norm(p, axis=1)
    mask = (nt > 1e-12) & (npred > 1e-12)

    if mask.sum() == 0:
        return 0.0  # 全 0 图，直接返回 0，避免除 0

    t = t[mask]
    p = p[mask]
    nt = nt[mask]
    npred = npred[mask]

    cos = np.sum(t * p, axis=1) / (nt * npred)
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.arccos(cos)  # rad
    sam_deg = float(np.mean(ang) * 180.0 / np.pi)
    return sam_deg


def compare_corr(x_true, x_pred):
    """
    CrossCorrelation across bands (mean over valid bands).
    Fix: avoid NaN by ignoring zero-variance bands.
    """
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)  # (C, HW)

    x_true = x_true - np.mean(x_true, axis=1, keepdims=True)
    x_pred = x_pred - np.mean(x_pred, axis=1, keepdims=True)

    numerator = np.sum(x_true * x_pred, axis=1)  # (C,)
    denom = np.sqrt(np.sum(x_true * x_true, axis=1) * np.sum(x_pred * x_pred, axis=1))  # (C,)

    mask = denom > 1e-12
    if mask.sum() == 0:
        return 0.0

    corr = numerator[mask] / denom[mask]
    return float(np.mean(corr))


def compare_sid(x_true, x_pred):
    x_true = x_true.astype(np.float32)
    x_pred = x_pred.astype(np.float32)
    N = x_true.shape[2]
    err = np.zeros(N, dtype=np.float32)
    for i in range(N):
        xi = x_true[:, :, i]
        yi = x_pred[:, :, i]
        err[i] = abs(
            np.sum(yi * np.log10((yi + 1e-3) / (xi + 1e-3))) +
            np.sum(xi * np.log10((xi + 1e-3) / (yi + 1e-3)))
        )
    return float(np.mean(err / (x_true.shape[0] * x_true.shape[1] + 1e-8)))


def quality_assessment(x_true, x_pred, data_range, ratio, multi_dimension=False):
    """
    x_true, x_pred: HWC float (0~1 recommended)
    """
    result = {
        "MPSNR": compare_mpsnr(x_true=x_true, x_pred=x_pred, data_range=data_range),
        "MSSIM": compare_mssim(x_true=x_true, x_pred=x_pred, data_range=data_range, multidimension=multi_dimension),
        "ERGAS": compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
        "SAM": compare_sam(x_true=x_true, x_pred=x_pred),
        "CrossCorrelation": compare_corr(x_true=x_true, x_pred=x_pred),
        "RMSE": compare_rmse(x_true=x_true, x_pred=x_pred),
    }
    return result
