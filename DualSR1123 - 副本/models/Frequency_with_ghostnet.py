import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGhostBlock(nn.Module):
    """简化的Ghost模块"""

    def __init__(self, in_channels, out_channels):
        super(SimpleGhostBlock, self).__init__()
        self.main_conv = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.ghost_conv = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1, groups=out_channels // 2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.main_conv(x)
        x2 = self.ghost_conv(x1)
        x_out = torch.cat([x1, x2], dim=1)
        return F.relu(self.bn(x_out))


class MSIFeatureExtractor(nn.Module):
    """MSI特征提取器"""

    def __init__(self, in_channels=3, out_channels=32):
        super(MSIFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, msi):
        return self.conv_layers(msi)


class MSIToHSIResidual(nn.Module):
    """将MSI转换为HSI的残差模块"""

    def __init__(self, in_channels=3, out_channels=31):
        super(MSIToHSIResidual, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, msi):
        return self.conv_layers(msi)


class SimpleFeaturePyramid(nn.Module):
    """简化的特征金字塔"""

    def __init__(self, in_channels, base_channels=32):
        super(SimpleFeaturePyramid, self).__init__()

        # 编码器
        self.enc1 = SimpleGhostBlock(in_channels, base_channels)
        self.enc2 = SimpleGhostBlock(base_channels, base_channels * 2)
        self.enc3 = SimpleGhostBlock(base_channels * 2, base_channels * 4)

        # 解码器
        self.dec2 = SimpleGhostBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = SimpleGhostBlock(base_channels * 2 + base_channels, base_channels)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))

        # 解码路径
        dec2_input = torch.cat([self.upsample(enc3), enc2], dim=1)
        dec2 = self.dec2(dec2_input)

        dec1_input = torch.cat([self.upsample(dec2), enc1], dim=1)
        dec1 = self.dec1(dec1_input)

        return dec1


class FrequencyAttention(nn.Module):
    """频域注意力机制"""

    def __init__(self, channels):
        super(FrequencyAttention, self).__init__()
        self.freq_conv = nn.Conv2d(channels, channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // 8, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 8, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        freq_feat = self.freq_conv(x)
        y = self.avg_pool(freq_feat).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LightweightSuperResolutionNet(nn.Module):
    """轻量化超分辨率网络 - 简化版本"""

    def __init__(self, in_channels=31, msi_channels=3, base_channels=32):
        super(LightweightSuperResolutionNet, self).__init__()

        # MSI处理模块
        self.msi_extractor = MSIFeatureExtractor(msi_channels, base_channels)
        self.msi_to_hsi = MSIToHSIResidual(msi_channels, in_channels)

        # 初始上采样
        self.initial_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # 主特征提取分支
        self.feature_pyramid = SimpleFeaturePyramid(in_channels, base_channels)

        # MSI特征融合
        self.msi_fusion_conv = nn.Conv2d(base_channels * 2, base_channels, 1)

        # 频域注意力
        self.freq_attention = FrequencyAttention(base_channels)

        # 最终重建
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, msi):
        """
        Args:
            x: 低分辨率HSI输入 [B, 31, 32, 32]
            msi: 多光谱图像输入 [B, 3, 128, 128]
        Returns:
            output: 超分结果 [B, 31, 128, 128]
        """
        # 初始上采样
        identity = self.initial_upsample(x)

        # 提取MSI特征
        msi_feat = self.msi_extractor(msi)

        # 主特征提取
        hsi_feat = self.feature_pyramid(identity)

        # MSI特征融合
        fused_feat = torch.cat([hsi_feat, msi_feat], dim=1)
        fused_feat = self.msi_fusion_conv(fused_feat)

        # 频域注意力
        attended_feat = self.freq_attention(fused_feat)

        # 最终重建
        output = self.final_conv(attended_feat)

        # 残差连接
        msi_residual = self.msi_to_hsi(msi)
        output = output + identity + msi_residual

        return output.clamp(0, 1)


class MultiScaleSuperResolutionNet(nn.Module):
    """多尺度超分辨率网络 - 最终版本"""

    def __init__(self, in_channels=31, msi_channels=3, base_channels=32):
        super(MultiScaleSuperResolutionNet, self).__init__()

        # 主网络
        self.main_net = LightweightSuperResolutionNet(in_channels, msi_channels, base_channels)

    def forward(self, x, msi, target_scale='128'):
        """
        Args:
            x: 低分辨率HSI输入 [B, 31, 32, 32]
            msi: 多光谱图像输入 [B, 3, 128, 128]
            target_scale: 目标尺度，目前只支持'128'
        """
        if target_scale == '128':
            return self.main_net(x, msi)
        else:
            raise ValueError("目前只支持target_scale='128'")


# 测试代码
def test_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建网络
    model = MultiScaleSuperResolutionNet(in_channels=31, msi_channels=3, base_channels=32).to(device)

    print("=== 维度转换测试 ===")

    # 测试输入
    dummy_input_lr = torch.randn(2, 31, 32, 32).to(device)
    dummy_input_msi = torch.randn(2, 3, 128, 128).to(device)

    print(f"LR HSI输入形状: {dummy_input_lr.shape}")
    print(f"MSI输入形状: {dummy_input_msi.shape}")

    # 测试前向传播
    with torch.no_grad():
        output = model(dummy_input_lr, dummy_input_msi, target_scale='128')
        print(f"输出形状: {output.shape}")

    # 验证形状
    assert output.shape == (2, 31, 128, 128), f"输出形状错误，期望(2, 31, 128, 128)，得到{output.shape}"

    print("\n✓ 网络测试通过!")
    print("✓ 所有维度转换正确!")

    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 总参数量: {total_params:,}")


if __name__ == "__main__":
    test_network()