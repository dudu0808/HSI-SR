import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 方案一: 使用卷积模拟小波变换的低频提取 (低通滤波 + 下采样) ---
class ConvWaveletLowPass(nn.Module):
    def __init__(self, in_channels):
        super(ConvWaveletLowPass, self).__init__()
        # 简单卷积模拟低通滤波：使用大核并下采样
        # 这里的卷积核可以看作是小波变换中的低通滤波器
        # stride=2 会将尺寸减半 (例如 64x64 -> 32x32)
        # padding=1 保持输出尺寸计算正确 (output = (input - kernel + 2*padding)/stride + 1)
        # groups=in_channels 实现深度可分离卷积，每个通道独立处理，模拟小波变换对每个通道独立操作
        self.conv_lp = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels)

    def forward(self, x):
        # 卷积操作同时进行低通滤波和 2 倍下采样
        return self.conv_lp(x)


# --- 通道注意力模块 (Channel Attention) 用于优化融合 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --- 主神经网络定义 ---
class WaveletFusionNet(nn.Module):
    def __init__(self,
                 low_res_channels: int = 31,  # 用于 hr_output_from_v5 的通道数 (31)
                 high_res_channels: int = 31,  # 用于 hr_output_from_v6 的通道数 (31)
                 texture_channels: int = 1,  # 纹理信息图通道数
                 output_channels: int = 31):  # 最终输出 HSI 的通道数
        super(WaveletFusionNet, self).__init__()

        # --- 新增/修改模块 ---

        # 1. 对 hr_output_from_v6 进行下采样，使其与 hr_output_from_v5 尺寸一致 (128x128 -> 64x64)
        # 这一步是为了让两者在进入低频提取模块前具有相同的空间分辨率，
        # 从而确保提取出的低频特征尺寸相同，便于计算低频损失。
        self.downsample_v6_for_wavelet = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

        # 2. 小波变换低频提取模块
        # 对 hr_output_from_v5 (31x64x64) 提取低频 -> 31x32x32
        self.wavelet_low_pass_v5 = ConvWaveletLowPass(low_res_channels)
        # 对 hr_output_from_v6 (下采样到 31x64x64 后) 提取低频 -> 31x32x32
        self.wavelet_low_pass_v6 = ConvWaveletLowPass(high_res_channels)

        # 3. 最终融合层：结合 hr_output_from_v6 和纹理信息图
        # 融合后的总通道数 = hr_output_from_v6 通道 (31) + 纹理图通道 (1) = 32
        fused_input_channels = high_res_channels + texture_channels

        self.channel_attention = ChannelAttention(fused_input_channels)  # 新增通道注意力模块

        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(fused_input_channels, 128, kernel_size=3, padding=1),  # 增加深度
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=1)  # 点卷积调整到最终输出通道
        )

    def forward(self, hr_output_from_v5: torch.Tensor, hr_output_from_v6: torch.Tensor, texture_map: torch.Tensor):
        # 输入维度:
        # hr_output_from_v5: (N, 31, 64, 64)  # 您的第一个 Transformer 模块的输出
        # hr_output_from_v6: (N, 31, 128, 128) # 您的第二个 Transformer 模块的输出
        # texture_map:     (N, 1, 128, 128) # 原始纹理信息图

        # --- 1. 小波变换提取低频信息 ---
        # 提取 hr_output_from_v5 的低频 (N, 31, 32, 32)
        v5_low_freq = self.wavelet_low_pass_v5(hr_output_from_v5)

        # 将 hr_output_from_v6 下采样到 64x64，以便提取可比较的低频信息
        # hr_output_from_v6_down: (N, 31, 64, 64)
        hr_output_from_v6_down = self.downsample_v6_for_wavelet(hr_output_from_v6)

        # 提取 hr_output_from_v6 (下采样后) 的低频 (N, 31, 32, 32)
        v6_low_freq = self.wavelet_low_pass_v6(hr_output_from_v6_down)

        # --- 2. 最终融合 ---
        # 沿着通道维度拼接 hr_output_from_v6 和 texture_map
        # 拼接顺序：(N, 31, 128, 128) + (N, 1, 128, 128) = (N, 32, 128, 128)
        fused_features = torch.cat([hr_output_from_v6, texture_map], dim=1)

        # 应用通道注意力模块，优化融合权重
        fused_features = self.channel_attention(fused_features)

        # 通过最终卷积层调整通道数到 31
        final_output_hsi = self.final_fusion_conv(fused_features) + hr_output_from_v6  # 添加残差连接，提升细节保留
        # 最终 output_hsi 形状为 (N, 31, 128, 128)

        # 返回最终的高光谱图像和用于低频损失的两个低频分量
        return final_output_hsi, v5_low_freq, v6_low_freq


# --- Demo 运行函数 ---
if __name__ == "__main__":
    # 定义模型参数
    # 第一个 Transformer 模块输出 (V5)
    V5_OUT_CHANNELS = 31
    V5_OUT_SIZE = 64
    # 第二个 Transformer 模块输出 (V6)
    V6_OUT_CHANNELS = 31
    V6_OUT_SIZE = 128
    # 纹理图
    TEXTURE_CHANNELS = 1
    # 最终输出 HSI
    FINAL_OUTPUT_CHANNELS = 31
    FINAL_OUTPUT_SIZE = 128

    # 设置设备 (优先使用 GPU，否则使用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 实例化模型
    model = WaveletFusionNet(
        low_res_channels=V5_OUT_CHANNELS,  # V5 的通道数
        high_res_channels=V6_OUT_CHANNELS,  # V6 的通道数
        texture_channels=TEXTURE_CHANNELS,
        output_channels=FINAL_OUTPUT_CHANNELS
    ).to(device)

    # 将模型设置为评估模式
    model.eval()

    # 创建模拟输入张量
    dummy_v5_output = torch.randn(1, V5_OUT_CHANNELS, V5_OUT_SIZE, V5_OUT_SIZE).to(device)
    print(f"模拟第一个 Transformer 输出 (31x64x64) 尺寸: {dummy_v5_output.shape}")

    dummy_v6_output = torch.randn(1, V6_OUT_CHANNELS, V6_OUT_SIZE, V6_OUT_SIZE).to(device)
    print(f"模拟第二个 Transformer 输出 (31x128x128) 尺寸: {dummy_v6_output.shape}")

    dummy_texture_map = torch.randn(1, TEXTURE_CHANNELS, FINAL_OUTPUT_SIZE, FINAL_OUTPUT_SIZE).to(device)
    print(f"模拟原始纹理信息图 (1x128x128) 尺寸: {dummy_texture_map.shape}")

    # 关闭梯度计算，节省内存并加速推理
    with torch.no_grad():
        final_hsi_output, low_freq_v5, low_freq_v6 = model(dummy_v5_output, dummy_v6_output, dummy_texture_map)

    print(f"\n最终输出 HSI 尺寸: {final_hsi_output.shape}")
    print(f"V5 低频分量尺寸 (用于损失): {low_freq_v5.shape}")
    print(f"V6 低频分量尺寸 (用于损失): {low_freq_v6.shape}")

    # 验证输出尺寸是否符合要求
    expected_final_hsi_shape = (1, FINAL_OUTPUT_CHANNELS, FINAL_OUTPUT_SIZE, FINAL_OUTPUT_SIZE)
    # 64x64 的图像经过 ConvWaveletLowPass (stride=2) 变为 32x32
    expected_low_freq_shape = (1, V5_OUT_CHANNELS, V5_OUT_SIZE // 2, V5_OUT_SIZE // 2)

    assert final_hsi_output.shape == expected_final_hsi_shape, "最终 HSI 输出尺寸不匹配预期！"
    assert low_freq_v5.shape == expected_low_freq_shape, "V5 低频分量尺寸不匹配预期！"
    assert low_freq_v6.shape == expected_low_freq_shape, "V6 低频分量尺寸不匹配预期！"

    print("\n所有模型输出尺寸均符合预期！")