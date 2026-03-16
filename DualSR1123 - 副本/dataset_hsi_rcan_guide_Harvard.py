import os
import glob
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from PIL import Image


# =========================
# SPF (保持不变)
# =========================
def create_F():
    F = np.array([
        [2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
        [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
        [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float32)
    return F / F.sum(axis=1, keepdims=True)


# =========================
# Utils (保持不变)
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
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def hwc255_to_chw_tensor(x: np.ndarray):
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


# =========================
# Dataset with Repeat Strategy
# =========================
class HSIWithRCANGuideDataset(Dataset):
    def __init__(
            self,
            mode="train",
            scale=4,
            hr_patch=128,
            harvard_root="/home/shiyanshi/dbq/Harvard1",
            dataset="harvard",
            repeat=50  # <--- ✅ 新增参数：重复采样次数
    ):
        assert mode in ["train", "val", "test"], "mode must be 'train', 'val' or 'test'"
        self.mode = mode
        self.scale = int(scale)
        self.hr_patch = int(hr_patch)
        self.F = create_F()

        # 训练集通常需要重复采样以增加每个Epoch的迭代次数
        # 验证集/测试集只需要跑一次全图，不需要重复
        if mode == 'train':
            self.repeat = repeat
        else:
            self.repeat = 1

        if mode == 'train':
            data_dir = os.path.join(harvard_root, 'train')
        elif mode == 'val':
            data_dir = os.path.join(harvard_root, 'val')
        else:
            data_dir = os.path.join(harvard_root, 'test')

        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        if len(self.file_paths) == 0:
            raise RuntimeError(f"No .mat files found in {data_dir}")

        print(
            f"[Dataset] mode={mode} files={len(self.file_paths)} repeat={self.repeat} total_len={len(self.file_paths) * self.repeat}")

    def __len__(self):
        # ✅ 骗过 DataLoader，让它以为数据量是原来的 repeat 倍
        return len(self.file_paths) * self.repeat

    def __getitem__(self, idx):
        # ✅ 映射回真实的文件索引
        file_idx = idx % len(self.file_paths)

        path = self.file_paths[file_idx]
        try:
            mat = scipy.io.loadmat(path)
            hsi_hr = mat['ref'].astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            hsi_hr = np.zeros((512, 512, 31), dtype=np.float32)

        h, w, c = hsi_hr.shape

        if self.mode == "train":
            # 随机裁剪：每次 epoch 调用到这张图时，都会随机切一个新位置
            if h >= self.hr_patch and w >= self.hr_patch:
                y = np.random.randint(0, h - self.hr_patch + 1)
                x = np.random.randint(0, w - self.hr_patch + 1)
                hr_img = hsi_hr[y: y + self.hr_patch, x: x + self.hr_patch, :]
            else:
                hr_img = hsi_hr
        else:
            # 验证/测试：中心裁剪或整图
            h_new = h - (h % self.scale)
            w_new = w - (w % self.scale)
            hr_img = hsi_hr[:h_new, :w_new, :]

        # 实时下采样生成 LR
        lr_img = bicubic_downsample_hsi(hr_img, self.scale)
        lr_rgb = hsi_to_rgb_lr_255(lr_img, self.F)

        img_hr_t = hwc01_to_chw_tensor(hr_img)
        img_lr_t = hwc01_to_chw_tensor(lr_img)
        img_rgb_t = hwc255_to_chw_tensor(lr_rgb)

        name = os.path.splitext(os.path.basename(path))[0]
        return img_hr_t, img_lr_t, img_rgb_t, name