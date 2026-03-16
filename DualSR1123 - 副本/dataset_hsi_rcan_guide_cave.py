import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image


# =========================
# SPF (same as your RCAN training)
# =========================
def create_F():
    # CAVE 和 Harvard 都是 31 波段，这里保持不变
    F = np.array([
        [2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
        [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
        [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float32)
    return F / F.sum(axis=1, keepdims=True)


# =========================
# Utils
# =========================
def modcrop_hwc(x, scale: int):
    h, w = x.shape[:2]
    return x[:h - (h % scale), :w - (w % scale)]


def bicubic_downsample_hsi(hsi_hwc_01: np.ndarray, scale: int):
    """HSI HxWxC float32 0~1 -> (H/scale)x(W/scale)xC float32 0~1"""
    h, w, c = hsi_hwc_01.shape
    out_h, out_w = h // scale, w // scale
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    for i in range(c):
        band = (np.clip(hsi_hwc_01[..., i], 0, 1) * 255.0).astype(np.uint8)
        # Image.BICUBIC
        band_lr = Image.fromarray(band, mode="L").resize((out_w, out_h), resample=Image.BICUBIC)
        out[..., i] = np.asarray(band_lr, dtype=np.float32) / 255.0
    return out


def hsi_to_rgb_lr_255(hsi_lr_hwc_01: np.ndarray, F: np.ndarray):
    """LR_HSI(0~1) -> LR_RGB(0~255 float32) by SPF"""
    rgb = hsi_lr_hwc_01 @ F.T
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.float32)


def hwc01_to_chw_tensor(x: np.ndarray):
    """HxWxC 0~1 -> torch [C,H,W] float32"""
    if x.ndim == 2:
        x = x[..., None]
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def hwc255_to_chw_tensor(x: np.ndarray):
    """HxWxC 0~255 -> torch [C,H,W] float32"""
    if x.ndim == 2:
        x = x[..., None]
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def _load_hsi_as_hwc01(mat_path: str, key: str, scale: int):
    """load mat -> HWC float32 in 0~1, and modcrop to divisible by scale"""
    try:
        d = loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

    if key not in d:
        # 尝试自动寻找最可能的 key (数据量最大的变量)
        keys = [k for k in d.keys() if not k.startswith('__')]
        # 简单策略：找 shape 维度为 3 的
        for k in keys:
            if d[k].ndim == 3:
                key = k
                break

    hsi = d[key]

    # to HWC
    # CAVE 通常已经是 HWC 或 CHW，如果 31 在第一维且最后一维不是 31，则转置
    if hsi.ndim == 3 and hsi.shape[0] == 31 and hsi.shape[-1] != 31:
        hsi = np.transpose(hsi, (1, 2, 0))
    hsi = hsi.astype(np.float32)

    # normalize fallback
    mx = float(hsi.max())
    if mx > 1.5:
        hsi = hsi / (mx + 1e-8)
    hsi = np.clip(hsi, 0.0, 1.0)

    # modcrop
    hsi = modcrop_hwc(hsi, scale)
    return hsi


# =========================
# Dataset (CAVE Only)
# =========================
class HSIWithRCANGuideDataset(Dataset):
    """
    CAVE 数据集专用加载器
    """

    def __init__(
            self,
            mode="train",
            scale=4,
            hr_patch=128,
            stride=64,  # CAVE数据少，stride设小一点可以多切点片
            harvard_root="/home/shiyanshi/dbq/CAVE",  # 参数名保留兼容性，实际传CAVE路径
            dataset="cave",  # 保留接口兼容性
            repeat=2,  # ✅ 新增：重复倍数
    ):
        assert mode in ["train", "val"], "mode must be 'train' or 'val'"
        self.mode = mode
        self.scale = int(scale)
        self.repeat = repeat  # ✅ 记录 repeat
        self.F = create_F()

        # CAVE 路径配置 (假设 train 和 validate 文件夹)
        # CAVE 数据集的 Key 通常是 "Z" 或者文件名本身，这里假设统一为 "Z"
        # 如果你的 .mat 文件里 key 不一样，_load_hsi_as_hwc01 里有简单的自动寻找逻辑
        cave_root = harvard_root

        if mode == "train":
            img_dir = os.path.join(cave_root, "train")
            self.mat_key = "Z"
        else:
            img_dir = os.path.join(cave_root, "val")  # 或者 val
            self.mat_key = "Z"

        if not os.path.exists(img_dir):
            # 尝试不带子文件夹直接读 (有的数据集结构没有分 train/val 文件夹)
            if os.path.exists(cave_root):
                print(f"[Warning] {img_dir} not found, trying root {cave_root}")
                img_dir = cave_root
            else:
                raise ValueError(f"Path not found: {img_dir}")

        # 收集所有 .mat
        self.items = sorted(glob.glob(os.path.join(img_dir, "*.mat")))

        if not self.items:
            raise RuntimeError(f"No HSI mats found in {img_dir}")

        # ---- train patch config (ONLY for train) ----
        self.patch_index = []

        if self.mode == "train":
            self.hr_patch = int(hr_patch)
            self.stride = int(stride)
            self.lr_patch = self.hr_patch // self.scale
            self.lr_stride = self.stride // self.scale

            # 预计算 Patch 索引
            for mat_path in self.items:
                # 预读一次获取尺寸
                hsi_hr_full = _load_hsi_as_hwc01(mat_path, self.mat_key, self.scale)
                if hsi_hr_full is None: continue

                H, W, _ = hsi_hr_full.shape

                # 滑动窗口
                for y in range(0, H - self.hr_patch + 1, self.stride):
                    for x in range(0, W - self.hr_patch + 1, self.stride):
                        self.patch_index.append((mat_path, self.mat_key, y, x))

            print(
                f"[CAVE Dataset] mode=train imgs={len(self.items)} "
                f"patches={len(self.patch_index)} repeat={self.repeat} "
                f"total_len={len(self.patch_index) * self.repeat}"
            )
        else:
            print(f"[CAVE Dataset] mode=val imgs={len(self.items)} (full image eval)")

    def __len__(self):
        if self.mode == "train":
            # ✅ 返回长度乘以 repeat
            return len(self.patch_index) * self.repeat
        else:
            return len(self.items)

    def __getitem__(self, idx):
        if self.mode == "train":
            # ✅ 取模，实现循环读取
            idx = idx % len(self.patch_index)
            mat_path, key, y, x = self.patch_index[idx]
        else:
            mat_path = self.items[idx]
            key = self.mat_key
            y = x = None

        # HR full (HWC 0~1, modcrop)
        hsi_hr_full = _load_hsi_as_hwc01(mat_path, key, self.scale)

        # build LR full (HWC 0~1) + LR_RGB full (0~255)
        hsi_lr_full = bicubic_downsample_hsi(hsi_hr_full, self.scale)
        lr_rgb_full = hsi_to_rgb_lr_255(hsi_lr_full, self.F)

        if self.mode == "train":
            # crop HR patch
            hr = hsi_hr_full[y:y + self.hr_patch, x:x + self.hr_patch, :]

            # aligned LR crop
            ly, lx = y // self.scale, x // self.scale
            lr = hsi_lr_full[ly:ly + self.lr_patch, lx:lx + self.lr_patch, :]
            lr_rgb = lr_rgb_full[ly:ly + self.lr_patch, lx:lx + self.lr_patch, :]
        else:
            # full image
            hr = hsi_hr_full
            lr = hsi_lr_full
            lr_rgb = lr_rgb_full

        # to tensor
        hsi_hr_t = hwc01_to_chw_tensor(hr)  # [31,H,W] 0~1
        hsi_lr_t = hwc01_to_chw_tensor(lr)  # [31,h,w] 0~1
        lr_rgb_t = hwc255_to_chw_tensor(lr_rgb)  # [3,h,w]  0~255

        name = os.path.basename(mat_path)
        return hsi_hr_t, hsi_lr_t, lr_rgb_t, name