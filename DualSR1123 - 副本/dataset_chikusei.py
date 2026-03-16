import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image


# =========================
# SPF Helper (不变)
# =========================
def load_spf_from_mat(mat_path):
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"SPF mat file not found at: {mat_path}")
    data = loadmat(mat_path)
    if 'R' not in data:  # 请确认你的 key 是 'R' 还是 'hsi'
        # 如果不知道key，尝试自动获取
        keys = [k for k in data.keys() if not k.startswith('__')]
        key = keys[0]
    else:
        key = 'R'

    F = data[key].astype(np.float32)
    # (128, 4) -> (4, 128)
    if F.shape[0] == 128:
        F = F.T
    # 取前3通道
    if F.shape[0] > 3:
        F = F[:3, :]
    # 归一化
    F = F / (F.sum(axis=1, keepdims=True) + 1e-8)
    return F


# =========================
# Utils (不变)
# =========================
def bicubic_downsample_hsi(hsi_hwc_01: np.ndarray, scale: int):
    h, w, c = hsi_hwc_01.shape
    out_h, out_w = h // scale, w // scale
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    for i in range(c):
        band = (np.clip(hsi_hwc_01[..., i], 0, 1) * 255.0).astype(np.uint8)
        band_lr = Image.fromarray(band, mode="L").resize((out_w, out_h), resample=Image.BICUBIC)
        out[..., i] = np.asarray(band_lr, dtype=np.float32) / 255.0
    return out


def hsi_to_rgb_lr_255(hsi_lr_hwc_01: np.ndarray, F: np.ndarray):
    rgb = hsi_lr_hwc_01 @ F.T
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.float32)


def hwc01_to_chw_tensor(x: np.ndarray):
    if x.ndim == 2: x = x[..., None]
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def hwc255_to_chw_tensor(x: np.ndarray):
    if x.ndim == 2: x = x[..., None]
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def _load_hsi_as_hwc01(mat_path: str):
    try:
        d = loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None
    keys = [k for k in d.keys() if not k.startswith('__')]
    # 简单的寻找 key 逻辑：找维度为3的变量
    key = next((k for k in keys if d[k].ndim == 3), None)
    if key is None: return None

    hsi = d[key].astype(np.float32)
    hsi = np.clip(hsi, 0.0, 1.0)  # 确保是 0-1
    return hsi


# =========================
# Dataset Class (核心修改)
# =========================
class ChikuseiDataset(Dataset):
    def __init__(
            self,
            mode="train",
            scale=4,
            data_root="/home/shiyanshi/dbq/chikusei1",
            spf_path="/home/shiyanshi/dbq/chikusei_128_4.mat",
            hr_patch=128,  # LR=32 -> HR=128
            stride=64,  # 你的要求：步长 64
            repeat=1,  # 因为切片已经扩充了49倍，这里repeat设1即可
    ):
        self.mode = mode
        self.scale = int(scale)
        self.repeat = repeat
        self.hr_patch = int(hr_patch)
        self.stride = int(stride)

        # Load SPF
        self.F = load_spf_from_mat(spf_path)

        # 确定文件夹
        if mode == "train":
            img_dir = os.path.join(data_root, "train")
        elif mode == "val":
            img_dir = os.path.join(data_root, "val")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 获取所有图片路径
        self.mat_paths = sorted(glob.glob(os.path.join(img_dir, "*.mat")))
        if not self.mat_paths:
            raise RuntimeError(f"No .mat files found in {img_dir}")

        # ✅ 核心：构建滑动窗口索引
        # 我们不把图读进来，只存 (mat_path, y, x) 的坐标
        self.patch_indices = []

        if self.mode == "train":
            print(f"[Dataset] Indexing sliding windows (HR={self.hr_patch}, Stride={self.stride})...")
            # 假设所有图都是 512x512，我们直接生成坐标，不用读图以节省初始化时间
            # 如果图片尺寸不统一，需要在下面循环里读图获取 H, W
            H_img, W_img = 512, 512

            for mat_path in self.mat_paths:
                # 滑动窗口生成坐标
                for y in range(0, H_img - self.hr_patch + 1, self.stride):
                    for x in range(0, W_img - self.hr_patch + 1, self.stride):
                        self.patch_indices.append((mat_path, y, x))

            print(f"[Dataset] Train: Found {len(self.mat_paths)} images -> {len(self.patch_indices)} patches.")
        else:
            # 验证集通常跑全图，不切片
            print(f"[Dataset] Val: {len(self.mat_paths)} full images.")

    def __len__(self):
        if self.mode == "train":
            return len(self.patch_indices) * self.repeat
        else:
            return len(self.mat_paths)

    def __getitem__(self, idx):
        if self.mode == "train":
            idx = idx % len(self.patch_indices)
            mat_path, y, x = self.patch_indices[idx]
        else:
            mat_path = self.mat_paths[idx]
            y, x = 0, 0  # Val 模式下即使 crop 也是从头开始，但下面会有逻辑区分

        # 1. Load Full HR Image
        hsi_hr = _load_hsi_as_hwc01(mat_path)

        # 2. Crop (Training Only)
        if self.mode == "train":
            hsi_hr = hsi_hr[y: y + self.hr_patch, x: x + self.hr_patch, :]
            # 此时 hsi_hr 形状变为 (128, 128, 128)

        # 3. Downsample to LR
        # Train: 128 -> 32
        # Val: 512 -> 128
        hsi_lr = bicubic_downsample_hsi(hsi_hr, self.scale)

        # 4. Generate RGB Guide
        lr_rgb = hsi_to_rgb_lr_255(hsi_lr, self.F)

        # 5. To Tensor
        hsi_hr_t = hwc01_to_chw_tensor(hsi_hr)
        hsi_lr_t = hwc01_to_chw_tensor(hsi_lr)
        lr_rgb_t = hwc255_to_chw_tensor(lr_rgb)

        name = os.path.basename(mat_path)
        return hsi_hr_t, hsi_lr_t, lr_rgb_t, name