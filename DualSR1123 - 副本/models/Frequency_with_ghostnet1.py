import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGhostBlock(nn.Module):
    """简化的 Ghost 模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels must be even for SimpleGhostBlock"
        self.main_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.ghost_conv = nn.Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            padding=1,
            groups=out_channels // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.main_conv(x)
        x2 = self.ghost_conv(x1)
        x_out = torch.cat([x1, x2], dim=1)
        return F.relu(self.bn(x_out))


class RGBGuideExtractor(nn.Module):
    """RGB 引导特征提取器（原 MSIFeatureExtractor 改名）"""

    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, guide_rgb):
        return self.conv_layers(guide_rgb)


class RGBToHSIResidual(nn.Module):
    """RGB -> HSI 的残差模块（原 MSIToHSIResidual 改名）"""

    def __init__(self, in_channels=3, out_channels=31):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, guide_rgb):
        return self.conv_layers(guide_rgb)


class SimpleFeaturePyramid(nn.Module):
    """简化的特征金字塔（U-Net 风格）"""

    def __init__(self, in_channels, base_channels=32):
        super().__init__()

        # 编码器
        self.enc1 = SimpleGhostBlock(in_channels, base_channels)
        self.enc2 = SimpleGhostBlock(base_channels, base_channels * 2)
        self.enc3 = SimpleGhostBlock(base_channels * 2, base_channels * 4)

        # 解码器
        self.dec2 = SimpleGhostBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = SimpleGhostBlock(base_channels * 2 + base_channels, base_channels)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

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


class ChannelAttention(nn.Module):
    """通道注意力（你原 FrequencyAttention 本质是 SE，这里改名更准确）"""

    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LightweightSuperResolutionNet(nn.Module):
    """轻量化超分辨率网络（改名 + 去 clamp 的最小改动版）"""

    def __init__(self, in_channels=31, guide_channels=3, base_channels=32):
        super().__init__()

        # RGB guide 处理
        self.guide_extractor = RGBGuideExtractor(guide_channels, base_channels)
        self.rgb_to_hsi = RGBToHSIResidual(guide_channels, in_channels)

        # 初始上采样（HSI LR -> HSI HR 尺度）
        self.initial_upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        # 主特征提取分支（在 HR 尺度上做）
        self.feature_pyramid = SimpleFeaturePyramid(in_channels, base_channels)

        # guide 特征融合（concat 后 1x1 压回 base_channels）
        self.guide_fusion_conv = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

        # 注意力
        self.attn = ChannelAttention(base_channels)

        # 最终重建
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

    def forward(self, hsi_lr, guide_rgb):
        """
        Args:
            hsi_lr   : [B,31,h,w] 0~1
            guide_rgb: [B,3,H,W]  0~1  (RCAN 输出 / 255 得到)
        Returns:
            pred_hsi_hr (NOT clamped): [B,31,H,W]
        """
        # 1) 基线：HSI 插值上采样
        identity = self.initial_upsample(hsi_lr)

        # 2) guide 特征（HR）
        guide_feat = self.guide_extractor(guide_rgb)

        # 3) HSI 主干特征（HR）
        hsi_feat = self.feature_pyramid(identity)

        # 4) 融合 + 注意力
        fused = torch.cat([hsi_feat, guide_feat], dim=1)
        fused = self.guide_fusion_conv(fused)
        fused = self.attn(fused)

        # 5) 重建残差
        out = self.final_conv(fused)

        # 6) 三路残差相加（不 clamp）
        out = out + identity + self.rgb_to_hsi(guide_rgb)
        return out


class MultiScaleSuperResolutionNet(nn.Module):
    """多尺度封装（你当前只用 scale=4 / 128 输出，所以保持最简接口）"""

    def __init__(self, in_channels=31, guide_channels=3, base_channels=32):
        super().__init__()
        self.main_net = LightweightSuperResolutionNet(in_channels, guide_channels, base_channels)

    def forward(self, hsi_lr, guide_rgb, target_scale="128"):
        if target_scale == "128":
            return self.main_net(hsi_lr, guide_rgb)
        raise ValueError("Currently only supports target_scale='128'")


# -------------------------
# Quick shape test
# -------------------------
def test_network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = MultiScaleSuperResolutionNet(in_channels=31, guide_channels=3, base_channels=32).to(device)

    hsi_lr = torch.randn(2, 31, 32, 32, device=device)
    guide_rgb = torch.randn(2, 3, 128, 128, device=device)

    with torch.no_grad():
        out = model(hsi_lr, guide_rgb, target_scale="128")

    print("hsi_lr:", tuple(hsi_lr.shape))
    print("guide_rgb:", tuple(guide_rgb.shape))
    print("out:", tuple(out.shape))
    assert out.shape == (2, 31, 128, 128)

    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)


if __name__ == "__main__":
    test_network()
