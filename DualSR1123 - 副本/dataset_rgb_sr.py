import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image


# =========================
# Utilities
# =========================
def create_F():
    F = np.array([
        [2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
        [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
        [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float32)
    return F / F.sum(axis=1, keepdims=True)


def hwc_to_chw_float255(x):
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(x)


def modcrop_hwc(x, scale):
    h, w = x.shape[:2]
    return x[:h - (h % scale), :w - (w % scale)]


def bicubic_downsample_uint8(hr_uint8, scale):
    h, w = hr_uint8.shape[:2]
    lr_img = Image.fromarray(hr_uint8, mode="RGB").resize((w // scale, h // scale), resample=Image.BICUBIC)
    return np.array(lr_img, dtype=np.uint8)


def percentile_normalize_01(rgb_float, p_lo=0.1, p_hi=99.9, eps=1e-8):
    lo = np.percentile(rgb_float, p_lo)
    hi = np.percentile(rgb_float, p_hi)
    rgb = (rgb_float - lo) / (hi - lo + eps)
    return np.clip(rgb, 0.0, 1.0)


def hsi31_to_rgb_uint8(hsi, F, use_percentile_norm=True):
    hsi = hsi.astype(np.float32)

    # 兜底：31xHxW -> HxWx31
    if hsi.ndim == 3 and hsi.shape[0] == 31 and hsi.shape[-1] != 31:
        hsi = np.transpose(hsi, (1, 2, 0))

    assert hsi.ndim == 3 and hsi.shape[-1] == 31, f"Expected HxWx31, got {hsi.shape}"

    rgb = hsi @ F.T  # (H,W,31) x (31,3) -> (H,W,3)

    if use_percentile_norm:
        rgb = percentile_normalize_01(rgb)

    hr = (rgb * 255.0).round().astype(np.uint8)
    return hr


def random_crop_pair(lr_hwc, hr_hwc, lr_patch, scale):
    h, w = lr_hwc.shape[:2]
    if h < lr_patch or w < lr_patch:
        raise ValueError(f"LR too small: {lr_hwc.shape}, patch={lr_patch}")
    x = random.randint(0, w - lr_patch)
    y = random.randint(0, h - lr_patch)
    lr = lr_hwc[y:y + lr_patch, x:x + lr_patch, :]
    hr = hr_hwc[y * scale:(y + lr_patch) * scale, x * scale:(x + lr_patch) * scale, :]
    return lr, hr


def center_crop_pair(lr_hwc, hr_hwc, lr_patch, scale):
    h, w = lr_hwc.shape[:2]
    if h < lr_patch or w < lr_patch:
        raise ValueError(f"LR too small: {lr_hwc.shape}, patch={lr_patch}")
    x = (w - lr_patch) // 2
    y = (h - lr_patch) // 2
    lr = lr_hwc[y:y + lr_patch, x:x + lr_patch, :]
    hr = hr_hwc[y * scale:(y + lr_patch) * scale, x * scale:(x + lr_patch) * scale, :]
    return lr, hr


def to_uint8_255(x):
    """把 DIV2K mat 里的 lr/hr 转成 HWC uint8 0~255（兼容 float 0~1 或 0~255）"""
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:  # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))
    if x.dtype == np.uint8:
        return x
    x = x.astype(np.float32)
    if x.max() <= 1.5:
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return np.round(x).astype(np.uint8)


# =========================
# Final Dataset
# =========================
class MixedDIV2KMatAndHSI31RGBSR(Dataset):
    """
    训练(train)：DIV2K(train) 全部 + HSI(CAVE/ICVL train) 全部（不按比例抽样）
    验证(val) ：只用 HSI(CAVE validate + ICVL val)

    输出：
      lr: torch.float32 [3, lr_patch, lr_patch] in 0~255
      hr: torch.float32 [3, lr_patch*scale, lr_patch*scale] in 0~255
    """

    def __init__(self, mode="train", scale=4, lr_patch=48, use_percentile_norm=True):
        assert mode in ["train", "val"], "mode must be 'train' or 'val'"

        self.mode = mode
        self.scale = scale
        self.lr_patch = lr_patch
        self.use_percentile_norm = use_percentile_norm
        self.F = create_F()

        # ===== DIV2K (only used in train) =====
        self.div2k_lr_key = "lr"
        self.div2k_hr_key = "hr"

        if mode == "train":
            div2k_mat_dir = "/home/shiyanshi/dbq/DIV2K/train"
            self.div2k_paths = sorted(glob.glob(os.path.join(div2k_mat_dir, "*.mat")))
            if not self.div2k_paths:
                raise RuntimeError(f"No DIV2K .mat found in {div2k_mat_dir}")
        else:
            self.div2k_paths = []

        # ===== HSI (train or val) =====
        if mode == "train":
            hsi_dirs = ["/home/shiyanshi/dbq/CAVE/train", "/home/shiyanshi/dbq/ICVL/train"]
            hsi_key_map = {"/home/shiyanshi/dbq/CAVE/train": "Z", "/home/shiyanshi/dbq/ICVL/train": "gt"}
        else:
            hsi_dirs = ["/home/shiyanshi/dbq/CAVE/validate", "/home/shiyanshi/dbq/ICVL/val"]
            hsi_key_map = {"/home/shiyanshi/dbq/CAVE/validate": "Z", "/home/shiyanshi/dbq/ICVL/val": "gt"}

        self.hsi_items = []
        for d in hsi_dirs:
            mats = sorted(glob.glob(os.path.join(d, "*.mat")))
            self.hsi_items += [(m, hsi_key_map[d]) for m in mats]

        if not self.hsi_items:
            raise RuntimeError(f"No HSI .mat found for mode={mode} in {hsi_dirs}")

        print(f"[MixedDataset-FULL] mode={mode} DIV2K mats={len(self.div2k_paths)} HSI31 mats={len(self.hsi_items)}")

    def __len__(self):
        if self.mode == "train":
            return len(self.div2k_paths) + len(self.hsi_items)
        return len(self.hsi_items)

    def _get_div2k_pair_by_index(self, idx):
        p = self.div2k_paths[idx]
        d = loadmat(p)
        lr = to_uint8_255(d[self.div2k_lr_key])
        hr = to_uint8_255(d[self.div2k_hr_key])

        if self.mode == "train":
            lr, hr = random_crop_pair(lr, hr, self.lr_patch, self.scale)
        else:
            lr, hr = center_crop_pair(lr, hr, self.lr_patch, self.scale)
        return lr, hr

    def _get_hsi_pair_by_index(self, idx):
        mat_path, key = self.hsi_items[idx]
        d = loadmat(mat_path)
        hsi = d[key]

        # HSI -> RGB(HR uint8)
        hr = hsi31_to_rgb_uint8(hsi, self.F, use_percentile_norm=self.use_percentile_norm)

        # 保证可整除 scale
        hr = modcrop_hwc(hr, self.scale)

        # bicubic downsample 得到 LR
        lr = bicubic_downsample_uint8(hr, self.scale)

        if self.mode == "train":
            # 训练仍然随机 patch
            lr, hr = random_crop_pair(lr, hr, self.lr_patch, self.scale)
        else:
            # ===== val：整图评估，不裁剪 =====
            # 直接用整张 lr/hr
            pass

        return lr, hr

    def __getitem__(self, idx):
        if self.mode == "train":
            if idx < len(self.div2k_paths):
                lr, hr = self._get_div2k_pair_by_index(idx)
            else:
                lr, hr = self._get_hsi_pair_by_index(idx - len(self.div2k_paths))
        else:
            lr, hr = self._get_hsi_pair_by_index(idx)

        return hwc_to_chw_float255(lr), hwc_to_chw_float255(hr)
