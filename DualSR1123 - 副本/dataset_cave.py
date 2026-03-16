import os
import random
import torch
import torch.utils.data as data
import numpy as np
import scipy.io as sio
import torch.nn.functional as F


class CAVEImplicitDataset(data.Dataset):
    def __init__(self, root, mode='train', scale=4, lr_patch_size=48, samples=2304, repeat=50):
        """
        Args:
            root: CAVE数据集根目录 (包含 .mat 文件)
            mode: 'train' 或 'val'
            repeat: 虚拟扩充倍数 (默认50)。CAVE只有32张图，不扩充训练会很差。
        """
        self.root = root
        self.mode = mode
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.samples = samples
        self.repeat = repeat

        # === 1. 文件扫描 ===
        # CAVE 数据集通常较小，有时大家不分文件夹。
        # 这里逻辑是：如果 root 下有 train/val 子文件夹，就用子文件夹。
        # 如果没有，就读取 root 下所有文件，然后按 26:6 (约8:2) 自动划分。

        target_dir = os.path.join(root, mode)
        if os.path.exists(target_dir):
            # 存在子文件夹，直接读取
            self.files = sorted(
                [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith('.mat')])
        else:
            # 不存在子文件夹，读取根目录并自动划分
            all_files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith('.mat')])
            split_idx = int(0.8 * len(all_files))  # 80% 训练
            if mode == 'train':
                self.files = all_files[:split_idx]
            else:
                self.files = all_files[split_idx:]

        if len(self.files) == 0:
            raise ValueError(f"在 {root} 未找到 .mat 文件！")

        # === 2. 打印信息 ===
        if self.mode == 'train':
            print(
                f"[CAVE-TRAIN] 物理文件: {len(self.files)} 张 | 虚拟扩充: {len(self.files)} x {repeat} = {len(self.files) * repeat} Iterations")
        else:
            print(f"[CAVE-VAL] 加载成功: {len(self.files)} 张")

    def __len__(self):
        if self.mode == 'train':
            return len(self.files) * self.repeat
        else:
            return len(self.files)

    def __getitem__(self, index):
        # 虚拟索引映射到物理文件
        file_idx = index % len(self.files)
        fpath = self.files[file_idx]

        try:
            mat = sio.loadmat(fpath)
        except Exception as e:
            print(f"Error reading: {fpath}")
            raise e

        # === 3. 自动寻找 Key ===
        # CAVE 数据集的 Key 可能千奇百怪
        valid_keys = [k for k in mat.keys() if not k.startswith('__')]
        img_hr = None

        # 优先匹配常见名称
        for k in ['hsi', 'pixel_response', 'reflectance', 'data', 'img']:
            if k in valid_keys:
                img_hr = mat[k]
                break
        # 如果没找到，尝试找维度是 3 的变量
        if img_hr is None:
            for k in valid_keys:
                if isinstance(mat[k], np.ndarray) and mat[k].ndim == 3:
                    img_hr = mat[k]
                    break

        if img_hr is None:
            raise ValueError(f"无法在 {fpath} 中找到 HSI 数据，Keys: {valid_keys}")

        # 数据类型转换与归一化
        img_hr = img_hr.astype(np.float32)
        # CAVE 有时是 0-1，有时是 0-65535，有时是 0-255
        if img_hr.max() > 1.0:
            img_hr /= img_hr.max()

        H, W, C = img_hr.shape

        # === 4. 训练模式 (LIIF 随机采样) ===
        if self.mode == 'train':
            lr_h, lr_w = self.lr_patch_size, self.lr_patch_size
            hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

            # 随机裁剪
            x = random.randint(0, W - hr_w)
            y = random.randint(0, H - hr_h)
            crop_hr = img_hr[y:y + hr_h, x:x + hr_w, :]

            # 生成 Tensor
            hr_tensor = torch.from_numpy(crop_hr).permute(2, 0, 1).unsqueeze(0)
            lr_tensor = F.interpolate(hr_tensor, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)
            lr_hsi = lr_tensor.squeeze(0).clamp(0, 1)

            # 生成坐标 Grid
            h_range = torch.linspace(-1, 1, hr_h)
            w_range = torch.linspace(-1, 1, hr_w)
            grid_h, grid_w = torch.meshgrid(h_range, w_range, indexing='ij')
            grid = torch.stack([grid_w, grid_h], dim=-1).reshape(-1, 2)

            # 采样 GT
            hr_flat = torch.from_numpy(crop_hr).reshape(-1, C)
            indices = torch.randperm(grid.shape[0])[:self.samples]

            coords = grid[indices]
            hr_sample = hr_flat[indices]

            cell = torch.ones_like(coords)
            cell[:, 0] *= 2 / hr_w
            cell[:, 1] *= 2 / hr_h

            return lr_hsi, coords, cell, hr_sample

        # === 5. 验证模式 (全图) ===
        else:
            # 裁剪为 scale 整数倍
            H_tgt = (H // self.scale) * self.scale
            W_tgt = (W // self.scale) * self.scale
            img_hr = img_hr[:H_tgt, :W_tgt, :]

            hr_tensor = torch.from_numpy(img_hr).permute(2, 0, 1).unsqueeze(0)
            lr_tensor = F.interpolate(hr_tensor, scale_factor=1 / self.scale, mode='bicubic', align_corners=False)
            lr_hsi = lr_tensor.squeeze(0).clamp(0, 1)

            h_range = torch.linspace(-1, 1, H_tgt)
            w_range = torch.linspace(-1, 1, W_tgt)
            grid_h, grid_w = torch.meshgrid(h_range, w_range, indexing='ij')
            coords = torch.stack([grid_w, grid_h], dim=-1).reshape(-1, 2)

            cell = torch.ones_like(coords)
            cell[:, 0] *= 2 / W_tgt
            cell[:, 1] *= 2 / H_tgt

            hr_sample = torch.from_numpy(img_hr).reshape(-1, C)

            return lr_hsi, coords, cell, hr_sample