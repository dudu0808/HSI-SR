import os
import random
import torch
import torch.utils.data as data
import numpy as np
import scipy.io as sio
import torch.nn.functional as F


class HarvardImplicitDataset(data.Dataset):
    def __init__(self, root, mode='train', scale=4, lr_patch_size=48, samples=2304, repeat=50):
        """
        Args:
            repeat: [新增参数] 虚拟重复次数。
                    如果 train 有 40 张图，repeat=50，则 len=2000。
                    这样每个 Epoch 会随机采样 2000 个 patch。
        """
        self.root = root
        self.mode = mode
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.samples = samples
        self.repeat = repeat  # 保存重复次数

        # 1. 路径拼接
        self.split_path = os.path.join(root, mode)
        if not os.path.exists(self.split_path):
            self.split_path = os.path.join(root, mode.capitalize())

        if not os.path.exists(self.split_path):
            raise FileNotFoundError(f"找不到数据集子文件夹: {self.split_path}")

        # 2. 读取文件列表
        self.files = sorted([
            os.path.join(self.split_path, f)
            for f in os.listdir(self.split_path)
            if f.lower().endswith('.mat')
        ])

        if len(self.files) == 0:
            raise ValueError(f"在 {self.split_path} 下没有找到 .mat 文件")

        # 打印日志时区分物理数量和虚拟数量
        if self.mode == 'train':
            print(
                f"[{mode.upper()}] 物理文件: {len(self.files)} 张 | 虚拟扩充: {len(self.files)} x {self.repeat} = {len(self.files) * self.repeat} Iterations")
        else:
            print(f"[{mode.upper()}] 加载成功: {len(self.files)} 张 (验证集不扩充)")

    def __len__(self):
        # === 修改重点 ===
        if self.mode == 'train':
            return len(self.files) * self.repeat
        else:
            return len(self.files)

    def __getitem__(self, index):
        # === 修改重点：取余数以循环使用物理文件 ===
        file_idx = index % len(self.files)

        try:
            mat = sio.loadmat(self.files[file_idx])
        except Exception as e:
            print(f"Error reading: {self.files[file_idx]}")
            raise e

        # ... (以下读取 Image 逻辑保持不变，为了节省篇幅我简化了 key 的判断，你可以直接复制原本的 key 判断逻辑) ...
        # 建议直接复制你上一版能跑通的 key 读取逻辑放在这里
        if 'ref' in mat:
            img_hr = mat['ref']
        elif 'img' in mat:
            img_hr = mat['img']
        elif 'cube' in mat:
            img_hr = mat['cube']
        else:
            img_hr = mat[list(mat.keys())[-1]]

        img_hr = img_hr.astype(np.float32)
        if img_hr.max() > 1: img_hr /= img_hr.max()

        H, W, C = img_hr.shape

        if self.mode == 'train':
            # === 这里的逻辑完全不用变，因为每次调用都会 random.randint ===
            # 即使同一个文件被访问多次，每次裁剪的位置也是不同的
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

            # 生成 Grid
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

        else:
            # 验证集不用随机裁剪，逻辑不变
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