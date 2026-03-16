import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  # 用于读取 .mat 文件

# 从 rcan_model.py 中导入 RCAN 模型及所有必要的子模块
from .rcan_model import RCAN, MeanShift, ResidualGroup, RCAB, CALayer, default_conv, Upsampler

# 导入您的数据处理模块
from data_utils import CustomMatDataset
from torch.utils.data import DataLoader

# --- 1. 定义辅助模块 ---

# 空间像素融合 (SPF) 模块 - 修改为固定 SRF 矩阵
class SPF(nn.Module):
    def __init__(self, in_channels=31, out_channels=3, device='cuda'):
        super(SPF, self).__init__()
        wavelengths = np.linspace(400, 700, in_channels)  # 假设 31 波段，400-700nm
        srf_matrix = np.zeros((in_channels, out_channels))
        srf_matrix[:, 0] = np.exp(-((wavelengths - 650) ** 2) / (2 * 25 ** 2))  # Red
        srf_matrix[:, 1] = np.exp(-((wavelengths - 550) ** 2) / (2 * 25 ** 2))  # Green
        srf_matrix[:, 2] = np.exp(-((wavelengths - 450) ** 2) / (2 * 25 ** 2))  # Blue
        self.srf_matrix = nn.Parameter(torch.from_numpy(srf_matrix).float().to(device), requires_grad=False)

    def forward(self, x):
        # x: (B, 31, H, W)
        device = x.device  # 获取输入张量的设备
        # 确保 srf_matrix 在同一设备上
        srf_matrix = self.srf_matrix.to(device)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # (B, 31, H*W)
        rgb_flat = torch.matmul(srf_matrix.T, x_flat)  # (3, H*W) 注意转置以匹配 (3,31) @ (31, H*W)
        rgb = rgb_flat.view(B, 3, H, W)
        # 归一化
        max_val = rgb.max()
        if max_val > 0:
            rgb = (rgb / max_val) * 255
        rgb = torch.clamp(rgb, 0, 255) / 255  # 归一化到 [0,1]
        return rgb

# 高光谱重建模块
class HSIReconstruction(nn.Module):
    def __init__(self, in_channels_rgb=3, out_channels_hsi=64, num_features=64):
        super(HSIReconstruction, self).__init__()
        self.conv_initial = nn.Conv2d(in_channels_rgb, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_mid = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(num_features, out_channels_hsi, kernel_size=3, padding=1)

    def forward(self, hr_rgb):
        x = self.relu(self.conv_initial(hr_rgb))
        x = self.relu(self.conv_mid(x))
        hsi_output = self.conv_final(x)
        return hsi_output

# --- 2. VGG 特征提取器 ---
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer='relu3_3'):
        super(VGGFeatureExtractor, self).__init__()
        vgg19_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for param in vgg19_features.parameters():
            param.requires_grad = False
        self.features = nn.Sequential()
        layer_names = {
            'relu1_1': '0', 'relu1_2': '2',
            'relu2_1': '5', 'relu2_2': '7',
            'relu3_1': '10', 'relu3_2': '12', 'relu3_3': '14', 'relu3_4': '16',
            'relu4_1': '19', 'relu4_2': '21', 'relu4_3': '23', 'relu4_4': '25',
            'relu5_1': '28', 'relu5_2': '30', 'relu5_3': '32', 'relu5_4': '34',
        }
        target_layer_idx = -1
        if feature_layer in layer_names:
            target_layer_idx = int(layer_names[feature_layer])
        else:
            raise ValueError(f"Unsupported VGG feature layer: {feature_layer}")
        for i, layer in enumerate(vgg19_features):
            self.features.add_module(str(i), layer)
            if i == target_layer_idx:
                break

    def forward(self, x):
        return self.features(x)

# --- 3. 辅助函数：图像加载/保存 ---
def save_hsi_data(hsi_tensor, output_path):
    hsi_numpy = hsi_tensor.squeeze(0).cpu().numpy()
    np.save(output_path, hsi_numpy)
    print(f"高分辨率 HSI 保存至 {output_path}")

def save_rgb_image(tensor, output_path):
    transform = ToPILImage()
    img = transform(tensor.squeeze(0).cpu().clamp(0, 1))
    img.save(output_path, quality=95)  # 设置 JPG 压缩质量为 95

def visualize_srf(wavelengths, srf_matrix, title="Simplified Spectral Response Functions"):
    """
    可视化光谱响应函数。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, srf_matrix[:, 0], label='Red Channel SRF', color='red')
    plt.plot(wavelengths, srf_matrix[:, 1], label='Green Channel SRF', color='green')
    plt.plot(wavelengths, srf_matrix[:, 2], label='Blue Channel SRF', color='blue')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Response')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def convert_spectral_to_rgb_and_save_jpg(spectral_data, wavelengths, srf_matrix, output_jpg_path):
    """
    将光谱数据通过光谱响应函数转换为3D RGB图像，并保存为JPG。

    参数:
    spectral_data (np.ndarray): 输入的多光谱数据，形状为 (高度, 宽度, 波段数)。
    wavelengths (np.ndarray): 与 spectral_data 波段对应的波长数组（nm）。
    srf_matrix (np.ndarray): 光谱响应函数矩阵，形状为 (波段数, 3)。
    output_jpg_path (str): 输出JPG文件的保存路径。
    """
    if spectral_data.ndim != 3:
        raise ValueError(f"输入光谱数据必须是3维的 (高度, 宽度, 波段数)，但当前形状是 {spectral_data.shape}")

    height, width, num_bands = spectral_data.shape

    if num_bands != srf_matrix.shape[0]:
        raise ValueError(
            f"光谱数据波段数 ({num_bands}) "
            f"与 SRF 矩阵波段数 ({srf_matrix.shape[0]}) 不匹配。"
        )
    if wavelengths.size != num_bands:
        raise ValueError(
            f"波长数组大小 ({wavelengths.size}) "
            f"与光谱数据波段数 ({num_bands}) 不匹配。"
        )

    print(f"输入光谱数据形状: {spectral_data.shape} (高度, 宽度, 波段数)")
    print(f"光谱响应函数 SRF 形状: {srf_matrix.shape} (波段数, RGB通道)")

    rgb_float = np.einsum('ijk,kl->ijl', spectral_data.astype(float), srf_matrix)

    max_val = np.max(rgb_float)
    if max_val > 0:
        rgb_normalized = (rgb_float / max_val) * 255
    else:
        rgb_normalized = rgb_float

    rgb_image = np.clip(rgb_normalized, 0, 255).astype(np.uint8)

    print(f"转换后的 RGB 图像形状: {rgb_image.shape}")

    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_jpg_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        output_img = Image.fromarray(rgb_image, 'RGB')
        output_img.save(output_jpg_path, format='JPEG')
        print(f"成功将光谱数据转换为 RGB 图片并保存为: {output_jpg_path}")
    except Exception as e:
        print(f"保存 JPG 文件时发生错误: {e}")

# --- 4. 主流程函数 ---
def main():
    # --- 配置参数 ---
    data_root_dir = r"/home/shiyanshi/dbq/CAVE"
    subset = 'train'
    rcan_weights_path = r"/home/shiyanshi/dbq/models_ECCV2018RCAN/RCAN_BIX2.pt"
    sr_scale = 2
    output_dir = 'output_data'
    os.makedirs(output_dir, exist_ok=True)
    lr_rgb_output_path = r"/home/shiyanshi/dbq/DualSR-master/dubingqingnet/model/output_data/lr_rgb_32.jpg"
    hr_rgb_output_path = os.path.join(output_dir, "hr_rgb_64.jpg")

    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # --- 实例化并加载 RCAN 模型 ---
    rcan_args = {
        'n_resgroups': 10,
        'n_resblocks': 20,
        'n_feats': 64,
        'rgb_range': 255,
        'n_colors': 3,
        'res_scale': 1,
        'reduction': 16,
        'scale': sr_scale,
        'no_upsampling': False,
        'act': nn.ReLU(True),
    }
    rcan_model = RCAN(rcan_args).to(device)
    print(f"从 {rcan_weights_path} 加载 RCAN 权重...")
    rcan_state_dict = torch.load(rcan_weights_path, map_location=device)
    new_rcan_state_dict = {}
    for k, v in rcan_state_dict.items():
        if k.startswith('model.'):
            new_rcan_state_dict[k[len('model.'):]] = v
        else:
            new_rcan_state_dict[k] = v
    try:
        rcan_model.load_state_dict(new_rcan_state_dict)
        print("RCAN 权重加载成功。")
    except RuntimeError as e:
        print(f"加载 RCAN 权重时出错: {e}")
        print("尝试以 strict=False 加载。")
        rcan_model.load_state_dict(new_rcan_state_dict, strict=False)
        print("RCAN 权重以 strict=False 加载成功。")
    rcan_model.eval()
    print("RCAN 模型设置为评估模式。")

    # --- 实例化 VGG 特征提取器 ---
    vgg_feature_extractor = VGGFeatureExtractor(feature_layer='relu3_3').to(device)
    vgg_feature_extractor.eval()
    print("VGG 特征提取器已实例化并设置为评估模式。")

    # --- 实例化其他模块 ---
    spf_model = SPF(in_channels=31, out_channels=3).to(device)
    hsi_reconstruction_model = HSIReconstruction(
        in_channels_rgb=3,
        out_channels_hsi=64,
        num_features=64
    ).to(device)

    # --- 使用 CustomMatDataset 加载数据 ---
    print(f"从 {data_root_dir}/{subset} 使用 CustomMatDataset 加载 HSI 数据...")
    dataset = CustomMatDataset(root_dir=data_root_dir, subset=subset)
    if len(dataset) == 0:
        raise ValueError(f"在 {os.path.join(data_root_dir, subset)} 中未找到 .mat 文件。请检查数据路径和文件名。")
    lr_hsi_tensor, hr_hsi_gt_tensor, texture_map = dataset[0]
    lr_hsi_tensor = lr_hsi_tensor.unsqueeze(0).to(device)
    hr_hsi_gt_tensor = hr_hsi_gt_tensor.unsqueeze(0).to(device)
    texture_map = texture_map.unsqueeze(0).to(device)
    _, _, actual_lr_h, actual_lr_w = lr_hsi_tensor.shape
    _, _, hr_h, hr_w = hr_hsi_gt_tensor.shape
    print(f"低分辨率 HSI 形状 (来自 CustomMatDataset): {lr_hsi_tensor.shape}")
    print(f"高分辨率 HSI 真值形状 (来自 CustomMatDataset): {hr_hsi_gt_tensor.shape}")
    print(f"纹理图形状 (来自 CustomMatDataset): {texture_map.shape}")

    # --- 整个流程的推理 ---
    with torch.no_grad():
        print("执行空间像素融合 (SPF)...")
        lr_rgb_tensor = spf_model(lr_hsi_tensor)
        print(f"SPF 后的低分辨率 RGB 形状: {lr_rgb_tensor.shape}")
        save_rgb_image(lr_rgb_tensor, lr_rgb_output_path)
        print(f"低分辨率 RGB 图像保存至 {lr_rgb_output_path}")

        # 可视化模拟的 SRF
        wavelengths = np.linspace(400, 700, 31)  # 假设 31 波段，400-700nm
        srf_matrix = np.zeros((31, 3))
        srf_matrix[:, 0] = np.exp(-((wavelengths - 650) ** 2) / (2 * 25 ** 2))  # Red
        srf_matrix[:, 1] = np.exp(-((wavelengths - 550) ** 2) / (2 * 25 ** 2))  # Green
        srf_matrix[:, 2] = np.exp(-((wavelengths - 450) ** 2) / (2 * 25 ** 2))  # Blue
        visualize_srf(wavelengths, srf_matrix, "Simplified Spectral Response Functions for LR HSI")

        print("从低分辨率 RGB 提取 VGG 特征...")
        normalize_vgg_input = lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(
            device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        vgg_features = vgg_feature_extractor(normalize_vgg_input(lr_rgb_tensor))
        print(f"VGG 特征形状: {vgg_features.shape}")

        print("使用 RCAN 执行 RGB 超分辨率...")
        lr_rgb_for_rcan = lr_rgb_tensor * rcan_args['rgb_range'] if rcan_args['rgb_range'] == 255 else lr_rgb_tensor
        hr_rgb_tensor = rcan_model(lr_rgb_for_rcan)
        hr_rgb_tensor_clamped = hr_rgb_tensor / rcan_args['rgb_range'] if rcan_args['rgb_range'] == 255 else hr_rgb_tensor
        hr_rgb_tensor_clamped = hr_rgb_tensor_clamped.clamp(0, 1)
        print(f"RCAN 后的高分辨率 RGB 形状: {hr_rgb_tensor.shape}")
        save_rgb_image(hr_rgb_tensor_clamped, hr_rgb_output_path)
        print(f"高分辨率 RGB 图像保存至 {hr_rgb_output_path}")

        # 显示高分辨率 RGB 图像
        print("显示高分辨率 RGB 图像...")
        hr_rgb_np = hr_rgb_tensor_clamped.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(hr_rgb_np)
        plt.title('高分辨率 RGB 图像')
        plt.axis('off')
        plt.show()

        print("执行高光谱重建 (输出 64 通道)...")
        hr_hsi_output = hsi_reconstruction_model(hr_rgb_tensor_clamped)
        print(f"重建后的高分辨率 HSI 形状: {hr_hsi_output.shape}")
        save_hsi_data(hr_hsi_output, os.path.join(output_dir, "hr_hsi_64.npy"))

    print("\n--- 主流程成功完成！ ---")

# --- 5. Demo 部分 ---
def demo():
    print("开始 RGB 提取和高光谱重建演示...")
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n演示中止: {e}")
        print("请确保指定的数据根目录和 RCAN 权重文件存在且可访问。")
    except KeyError as e:
        print(f"\n演示中止: {e}")
        print("请检查 .mat 文件的结构。CustomMatDataset 期望 'X' 变量具有特定形状。")
    except ValueError as e:
        print(f"\n演示中止: {e}")
        print("发生值错误，可能是由于数据加载或处理过程中出现意外的数据形状或维度。")
    except Exception as e:
        print(f"\n演示过程中发生意外错误: {e}")

if __name__ == "__main__":
    demo()