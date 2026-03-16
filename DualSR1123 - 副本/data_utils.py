import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


class CustomMatDataset(Dataset):
    def __init__(self, root_dir, subset='train', stride=128):
        """
        简化版高光谱图像数据集
        Args:
            root_dir: 数据集根目录
            subset: 子集 ('train', 'test', 'validate')
            stride: 滑动窗口步长
        """
        self.data_path = os.path.join(root_dir, subset)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据集目录不存在: {self.data_path}")

        self.mat_files = [f for f in os.listdir(self.data_path) if f.endswith('.mat')]
        self.mat_files.sort()
        if not self.mat_files:
            raise FileNotFoundError(f"在目录 '{self.data_path}' 中没有找到.mat文件")

        self.stride = stride
        self.hr_size = 128  # 高分辨率切片大小
        self.lr_size = 32  # 低分辨率切片大小
        self.num_bands = 31

        # 预先选择固定的三个波段用于可视化（例如选择前三个波段，或者有代表性的波段）
        self.rgb_bands = [0, 10, 20]  # 固定使用第0、10、20个波段作为RGB通道

        # 计算每张图像的切片数量
        self.num_crops_per_image = self._calculate_num_crops()
        print(f"数据集: {subset}, 每张图像切片数: {self.num_crops_per_image}")
        print(f"总样本数: {len(self.mat_files)} × {self.num_crops_per_image} = {len(self)}")
        print(f"可视化使用波段: {self.rgb_bands}")

        # 创建可视化输出目录
        self.viz_dir = os.path.join(root_dir, f"visualization_{subset}")
        os.makedirs(self.viz_dir, exist_ok=True)
        print(f"可视化结果将保存到: {self.viz_dir}")

    def _calculate_num_crops(self):
        """计算每张512x512图像可以生成的128x128切片数量"""
        img_size = 512
        crop_size = self.hr_size

        # 计算在宽度和高度方向上的切片数量
        num_h = (img_size - crop_size) // self.stride + 1
        num_w = (img_size - crop_size) // self.stride + 1

        # 检查是否需要添加边缘切片
        if (img_size - crop_size) % self.stride != 0:
            num_h += 1
            num_w += 1

        return num_h * num_w

    def _create_pseudo_rgb(self, hyperspectral_data):
        """
        从31维高光谱数据创建伪RGB图像
        使用固定的三个波段作为RGB通道
        """
        # 使用预先固定的三个波段
        rgb_image = np.zeros((hyperspectral_data.shape[1], hyperspectral_data.shape[2], 3))

        for i, band in enumerate(self.rgb_bands):
            band_data = hyperspectral_data[band]
            # 归一化到0-1
            band_normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)
            # 转换为0-255
            rgb_image[:, :, i] = band_normalized * 255

        return rgb_image.astype(np.uint8)

    def __len__(self):
        return len(self.mat_files) * self.num_crops_per_image

    def __getitem__(self, idx):
        file_idx = idx // self.num_crops_per_image
        crop_idx = idx % self.num_crops_per_image

        mat_file_path = os.path.join(self.data_path, self.mat_files[file_idx])
        mat_data = sio.loadmat(mat_file_path)

        if 'Z' not in mat_data:
            raise KeyError(f"在 {self.mat_files[file_idx]} 中未找到变量 'Z'")

        hr_original_np = mat_data['Z'].astype(np.float32)  # Shape (512, 512, 31)

        # 归一化到 [0, 1]
        if hr_original_np.max() > 1.0:
            hr_original_np /= hr_original_np.max()

        # 转换为 (31, 512, 512)
        hr_original_tensor = torch.from_numpy(hr_original_np).permute(2, 0, 1)

        # 计算切片位置
        img_size = 512
        crop_size = self.hr_size
        num_w = (img_size - crop_size) // self.stride + 1

        # 计算当前切片的位置
        grid_row = crop_idx // num_w
        grid_col = crop_idx % num_w

        # 计算起始坐标，处理边缘情况
        start_h = min(grid_row * self.stride, img_size - crop_size)
        start_w = min(grid_col * self.stride, img_size - crop_size)

        # 提取HR切片 (31, 128, 128)
        hr_slice = hr_original_tensor[:, start_h:start_h + crop_size, start_w:start_w + crop_size]

        # 生成LR输入：下采样到 (31, 32, 32)
        lr_input = F.interpolate(hr_slice.unsqueeze(0),
                                 size=(self.lr_size, self.lr_size),
                                 mode='bicubic',
                                 align_corners=False).squeeze(0)

        # 可视化前10个样本
        # if idx < 10:
        #     self._visualize_sample(idx, file_idx, crop_idx, start_h, start_w,
        #                            hr_original_np, hr_slice.numpy(), lr_input.numpy())

        # 确保维度正确
        assert lr_input.shape == (31, 32, 32), f"LR输入形状错误: {lr_input.shape}"
        assert hr_slice.shape == (31, 128, 128), f"HR目标形状错误: {hr_slice.shape}"

        return lr_input, hr_slice

    def _visualize_sample(self, idx, file_idx, crop_idx, start_h, start_w,
                          original_hr, hr_slice, lr_input):
        """可视化样本"""
        # 创建伪RGB图像（使用相同的三个波段）
        original_rgb = self._create_pseudo_rgb(original_hr.transpose(2, 0, 1))
        hr_rgb = self._create_pseudo_rgb(hr_slice)
        lr_rgb = self._create_pseudo_rgb(lr_input)

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示原始图像中的位置
        axes[0].imshow(original_rgb)
        # 在原始图像上标记切片位置
        rect = plt.Rectangle((start_w, start_h), self.hr_size, self.hr_size,
                             linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title(f'Original 512x512\nSlice Position: ({start_h},{start_w})')
        axes[0].axis('off')

        # 显示HR切片
        axes[1].imshow(hr_rgb)
        axes[1].set_title(f'HR Slice 128x128\nBands: {self.rgb_bands}')
        axes[1].axis('off')

        # 显示LR输入（上采样以便可视化）
        lr_upsampled = np.array(Image.fromarray(lr_rgb).resize((128, 128), Image.BICUBIC))
        axes[2].imshow(lr_upsampled)
        axes[2].set_title(f'LR Input 32x32 (upsampled)\nBands: {self.rgb_bands}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f'sample_{idx:02d}_file{file_idx}_crop{crop_idx}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 单独保存HR和LR图像
        Image.fromarray(hr_rgb).save(os.path.join(self.viz_dir, f'sample_{idx:02d}_hr.png'))
        Image.fromarray(lr_rgb).save(os.path.join(self.viz_dir, f'sample_{idx:02d}_lr.png'))

        print(f"保存可视化结果: sample_{idx:02d} (文件: {self.mat_files[file_idx]}, 切片: {crop_idx})")
        print(f"  位置: ({start_h}, {start_w}), 使用波段: {self.rgb_bands}")


# 测试代码
if __name__ == "__main__":
    # 创建数据集实例（会自动可视化前10个样本）
    dataset = CustomMatDataset(root_dir="/home/shiyanshi/dbq/CAVE", subset='train', stride=64)

    # 获取前10个样本
    for i in range(min(10, len(dataset))):
        lr, hr = dataset[i]
        print(f"样本 {i}: LR形状 {lr.shape}, HR形状 {hr.shape}")

    print(f"\n可视化结果已保存到: {dataset.viz_dir}")