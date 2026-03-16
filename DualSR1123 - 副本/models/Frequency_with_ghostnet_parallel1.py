import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Ghost Block (Spectrally-Grouped, auto-divisible, g=4)
# =========================
class SimpleGhostBlock(nn.Module):
    """简化的 Ghost 模块（spectrally-grouped ghost conv, groups auto-divisible）"""
    def __init__(self, in_channels, out_channels, ghost_groups: int = 4):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels must be even for SimpleGhostBlock"

        ghost_ch = out_channels // 2
        # auto find a valid groups <= ghost_groups that divides ghost_ch
        g = int(ghost_groups)
        g = max(1, min(g, ghost_ch))
        while ghost_ch % g != 0 and g > 1:
            g -= 1

        self.main_conv = nn.Conv2d(in_channels, ghost_ch, kernel_size=1)

        # ✅ grouped ghost conv (instead of depthwise)
        self.ghost_conv = nn.Conv2d(
            ghost_ch,
            ghost_ch,
            kernel_size=3,
            padding=1,
            groups=g
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self._ghost_groups_used = g  # optional: for debug/print

    def forward(self, x):
        x1 = self.main_conv(x)
        x2 = self.ghost_conv(x1)
        x_out = torch.cat([x1, x2], dim=1)
        return F.relu(self.bn(x_out))


class RGBGuideExtractor(nn.Module):
    """RGB 引导特征提取器"""
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
        return self.conv_layers(guide_rgb)  # B×32×H×W


class RGBToHSIResidual(nn.Module):
    """RGB -> HSI 残差模块（共享参数，适配任意分辨率）"""
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
        return self.conv_layers(guide_rgb)  # B×31×H×W


class SimpleFeaturePyramid(nn.Module):
    """简化的特征金字塔（U-Net 风格），支持返回 64×64 和 128×128 两个尺度特征"""
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        # 编码器
        self.enc1 = SimpleGhostBlock(in_channels, base_channels)            # -> 32 @128
        self.enc2 = SimpleGhostBlock(base_channels, base_channels * 2)      # -> 64 @64
        self.enc3 = SimpleGhostBlock(base_channels * 2, base_channels * 4)  # -> 128 @32

        # 解码器
        self.dec2 = SimpleGhostBlock(base_channels * 4 + base_channels * 2, base_channels * 2)  # -> 64 @64
        self.dec1 = SimpleGhostBlock(base_channels * 2 + base_channels, base_channels)          # -> 32 @128

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, return_multi=False):
        """
        x: B×31×128×128
        return_multi=False -> return dec1 only: B×32×128×128
        return_multi=True  -> return (dec1, dec2): dec1 B×32×128×128, dec2 B×64×64×64
        """
        # 编码
        enc1 = self.enc1(x)               # B×32×128×128
        enc2 = self.enc2(self.pool(enc1)) # B×64×64×64
        enc3 = self.enc3(self.pool(enc2)) # B×128×32×32

        # 解码
        dec2_input = torch.cat([self.upsample(enc3), enc2], dim=1)  # B×(128+64)=192×64×64
        dec2 = self.dec2(dec2_input)                                # B×64×64×64

        dec1_input = torch.cat([self.upsample(dec2), enc1], dim=1)  # B×(64+32)=96×128×128
        dec1 = self.dec1(dec1_input)                                # B×32×128×128

        if return_multi:
            return dec1, dec2
        return dec1


class ChannelAttention(nn.Module):
    """通道注意力（SE）"""
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


class FusionHead(nn.Module):
    """
    concat -> 1x1(64->32) -> SE -> conv(32->31)
    输入：
      hsi_feat:  B×32×H×W
      guide_feat:B×32×H×W
    输出：
      out_res:   B×31×H×W
    """
    def __init__(self, base_channels=32, out_channels=31):
        super().__init__()
        self.fuse_1x1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.attn = ChannelAttention(base_channels)
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, hsi_feat, guide_feat):
        fused = torch.cat([hsi_feat, guide_feat], dim=1)  # B×64×H×W
        fused = self.fuse_1x1(fused)                      # B×32×H×W
        fused = self.attn(fused)                          # B×32×H×W
        out_res = self.final(fused)                       # B×31×H×W
        return out_res


class SpectralMixing(nn.Module):
    """
    很轻量的 spectral mixing：
    - 只在光谱通道维做显式交互（1×1 conv bottleneck）
    - residual: y = x + W2(ReLU(W1(x)))
    """
    def __init__(self, channels: int = 31, spectral_rank: int = 8):
        super().__init__()
        spectral_rank = int(spectral_rank)
        spectral_rank = max(1, min(spectral_rank, channels))
        self.proj1 = nn.Conv2d(channels, spectral_rank, kernel_size=1, bias=True)
        self.proj2 = nn.Conv2d(spectral_rank, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.proj2(self.act(self.proj1(x)))


class LightweightSuperResolutionNet(nn.Module):
    """
    并行式（×2 + ×4）：
    - 金字塔输出 hsi_feat_128(32×128×128) 与 hsi_feat_64_raw(64×64×64)
    - guide_feat_128(32×128×128) 下采样 -> guide_feat_64(32×64×64)
    - 两个 head 输出 pred_x2 / pred_x4
    - 融合 refine 得 final_x4
    - 插入 spectral mixing：对 pred_x2 / pred_x4 做轻量 band-to-band 交互
    """
    def __init__(self, in_channels=31, guide_channels=3, base_channels=32, spectral_rank=8):
        super().__init__()

        # guide 特征
        self.guide_extractor = RGBGuideExtractor(guide_channels, base_channels)

        # RGB->HSI 残差（共享，用于 64/128 两个尺度）
        self.rgb_to_hsi = RGBToHSIResidual(guide_channels, in_channels)

        # identity 分支（×2 / ×4）
        self.up_x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        # HSI 主干（输入 identity_x4，输出多尺度特征）
        self.feature_pyramid = SimpleFeaturePyramid(in_channels, base_channels)

        # 把 dec2 的 64 通道压到 32（匹配 guide_feat）
        self.dec2_reduce = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)  # 64->32

        # 两个尺度的 head
        self.head_x2 = FusionHead(base_channels=base_channels, out_channels=in_channels)  # 31×64×64
        self.head_x4 = FusionHead(base_channels=base_channels, out_channels=in_channels)  # 31×128×128

        # ⭐ 轻量 spectral mixing（共享）
        self.spectral_mix = SpectralMixing(channels=in_channels, spectral_rank=spectral_rank)

        # 最终融合 refine
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, hsi_lr, guide_rgb, return_x2=False):
        """
        Inputs:
          hsi_lr   : B×31×32×32 (0~1)
          guide_rgb: B×3 ×128×128 (0~1)
        Outputs:
          final_x4 : B×31×128×128
          (optional) pred_x2 : B×31×64×64
        """
        # ----- identities -----
        identity_x2 = self.up_x2(hsi_lr)  # B×31×64×64
        identity_x4 = self.up_x4(hsi_lr)  # B×31×128×128

        # ----- guide feats -----
        guide_feat_128 = self.guide_extractor(guide_rgb)  # B×32×128×128
        guide_feat_64 = F.interpolate(
            guide_feat_128, scale_factor=0.5, mode="bilinear", align_corners=False
        )  # B×32×64×64

        # ----- HSI feats (multi-scale) -----
        hsi_feat_128, hsi_feat_64_raw = self.feature_pyramid(identity_x4, return_multi=True)
        hsi_feat_64 = self.dec2_reduce(hsi_feat_64_raw)  # B×32×64×64

        # ----- ×2 branch -----
        out_res_x2 = self.head_x2(hsi_feat_64, guide_feat_64)  # B×31×64×64
        guide_rgb_64 = F.interpolate(guide_rgb, scale_factor=0.5, mode="bilinear", align_corners=False)
        rgb_res_x2 = self.rgb_to_hsi(guide_rgb_64)             # B×31×64×64
        pred_x2 = out_res_x2 + identity_x2 + rgb_res_x2        # B×31×64×64
        pred_x2 = self.spectral_mix(pred_x2)

        # ----- ×4 branch -----
        out_res_x4 = self.head_x4(hsi_feat_128, guide_feat_128)  # B×31×128×128
        rgb_res_x4 = self.rgb_to_hsi(guide_rgb)                  # B×31×128×128
        pred_x4 = out_res_x4 + identity_x4 + rgb_res_x4          # B×31×128×128
        pred_x4 = self.spectral_mix(pred_x4)

        # ----- fusion (final) -----
        pred_x2_up = F.interpolate(pred_x2, scale_factor=2, mode="bilinear", align_corners=False)  # B×31×128×128
        refine = self.fuse_refine(torch.cat([pred_x4, pred_x2_up], dim=1))                          # B×31×128×128
        final_x4 = pred_x4 + refine

        if return_x2:
            return final_x4, pred_x2
        return final_x4


class MultiScaleSuperResolutionNet(nn.Module):
    """多尺度封装（保持接口不变）"""
    def __init__(self, in_channels=31, guide_channels=3, base_channels=32, spectral_rank=8):
        super().__init__()
        self.main_net = LightweightSuperResolutionNet(
            in_channels=in_channels,
            guide_channels=guide_channels,
            base_channels=base_channels,
            spectral_rank=spectral_rank,
        )

    def forward(self, hsi_lr, guide_rgb, target_scale="128", return_x2=False):
        if target_scale == "128":
            return self.main_net(hsi_lr, guide_rgb, return_x2=return_x2)
        raise ValueError("Currently only supports target_scale='128'")


def test_network():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = MultiScaleSuperResolutionNet(
        in_channels=31, guide_channels=3, base_channels=32, spectral_rank=8
    ).to(device)

    hsi_lr = torch.randn(2, 31, 32, 32, device=device)
    guide_rgb = torch.randn(2, 3, 128, 128, device=device)

    with torch.no_grad():
        out = model(hsi_lr, guide_rgb, target_scale="128")
        out2 = model(hsi_lr, guide_rgb, target_scale="128", return_x2=True)

    print("hsi_lr:", tuple(hsi_lr.shape))
    print("guide_rgb:", tuple(guide_rgb.shape))
    print("out(final_x4):", tuple(out.shape))
    assert out.shape == (2, 31, 128, 128)

    final_x4, pred_x2 = out2
    print("out(final_x4, pred_x2):", tuple(final_x4.shape), tuple(pred_x2.shape))
    assert final_x4.shape == (2, 31, 128, 128)
    assert pred_x2.shape == (2, 31, 64, 64)

    total_params = sum(p.numel() for p in model.parameters())
    print("total params:", total_params)


if __name__ == "__main__":
    test_network()
