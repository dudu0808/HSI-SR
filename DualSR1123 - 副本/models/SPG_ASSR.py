import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 基础组件 (Basic Blocks)
# ==============================================================================

class PriorGuideFusion(nn.Module):
    """
    [Section III-C] Prior Guide Fusion (Fig. 2)
    公式 (4): F_f = F_hsi * M + F_rgb * (1 - M)
    """

    def __init__(self, dim):
        super(PriorGuideFusion, self).__init__()
        # 论文提到使用 Conv 和 Softmax (或 Sigmoid) 生成 Mask
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, f_hsi, f_rgb):
        # 如果尺寸不匹配 (RGB特征通常分辨率更高)，需要对齐
        if f_hsi.shape[-2:] != f_rgb.shape[-2:]:
            f_rgb_aligned = F.interpolate(f_rgb, size=f_hsi.shape[-2:], mode='bilinear', align_corners=False)
        else:
            f_rgb_aligned = f_rgb

        # Cat 并在通道维度生成 Mask
        mask = self.conv(torch.cat([f_hsi, f_rgb_aligned], dim=1))

        # 融合
        f_f = f_hsi * mask + f_rgb_aligned * (1 - mask)
        return f_f


class MCM(nn.Module):
    """
    [Section III-C] Multiconvolution Module (MCM) - Eq. (3)
    用于提取 RGB 的多尺度空间特征
    """

    def __init__(self, dim):
        super(MCM, self).__init__()
        self.split_dim = dim // 2

        # 不同核大小的 Depth-wise 卷积
        self.dwc_k3 = nn.Conv2d(self.split_dim, self.split_dim, 3, 1, 1, groups=self.split_dim)
        self.dwc_k5 = nn.Conv2d(self.split_dim, self.split_dim, 5, 1, 2, groups=self.split_dim)

        self.merge = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # Split
        x1, x2 = torch.split(x, self.split_dim, dim=1)
        # Convs
        x1 = self.dwc_k3(x1)
        x2 = self.dwc_k5(x2)
        # Cat & Merge
        out = torch.cat([x1, x2], dim=1)
        return self.merge(out)


class SpectralTransformerBlock(nn.Module):
    """
    [Section III-C] Spectral Transformer Block
    用于捕获 HSI 的光谱相关性
    """

    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.66):
        super(SpectralTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_expansion_factor)),
            nn.GELU(),
            nn.Linear(int(dim * ffn_expansion_factor), dim)
        )

    def forward(self, x):
        # x: [B, C, H, W] -> flatten spatial -> [B, HW, C]
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        # Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # FFN
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        # Reshape back
        out = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return out


class MLP(nn.Module):
    """用于 implicit function 的多层感知机"""

    def __init__(self, in_dim, out_dim, hidden_list=[256, 256]):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# ==============================================================================
# 2. 空间纹理先验生成器 (STPG) [cite: 48, 153]
# ==============================================================================

class STPG(nn.Module):
    """
    论文中使用预训练的 SwinIR 或 CiaoSR。
    为了提供完整可运行代码，这里实现了一个等效功能的 ResNet 提取器。
    在实际使用中，应加载预训练权重并冻结参数。
    """

    def __init__(self, embed_dim=128):
        super(STPG, self).__init__()
        # 简单的特征提取器，模拟预训练网络的功能
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU()
        )
        # 注意：STPG 需要输出 High-Resolution 的 RGB 特征
        # 如果输入是 LR RGB，这里通常包含上采样层。
        # 论文提到这是一个 Arbitrary Scale Upsampling module。
        # 这里简化为直接输出特征，假设输入已被插值或网络包含上采样。

    def forward(self, x_rgb):
        # 返回深层特征 F_rgb_hr
        return self.body(x_rgb)


# ==============================================================================
# 3. 特征融合增强模块 (FFEM) [cite: 188]
# ==============================================================================

class FFEM(nn.Module):
    """
    Feature Fusion Enhance Module
    结构：Input -> Head -> Fusion -> [Spectral Trans || MCM] -> Fusion -> Output
    """

    def __init__(self, hsi_bands, embed_dim=128):
        super(FFEM, self).__init__()

        # 浅层特征提取
        self.head_hsi = nn.Conv2d(hsi_bands, embed_dim, 3, 1, 1)
        self.head_rgb = nn.Conv2d(3, embed_dim, 3, 1, 1)

        # 第一次融合
        self.fusion1 = PriorGuideFusion(embed_dim)

        # 中间处理层 (根据论文描述的 L1, L2 层数)
        # HSI 分支
        self.hsi_blocks = nn.Sequential(
            SpectralTransformerBlock(embed_dim),
            SpectralTransformerBlock(embed_dim)
        )
        # RGB 分支
        self.rgb_blocks = nn.Sequential(
            MCM(embed_dim),
            MCM(embed_dim)
        )

        # 第二次融合 (输出前)
        self.fusion2 = PriorGuideFusion(embed_dim)

    def forward(self, x_hsi, x_rgb):
        # x_hsi: LR HSI, x_rgb: LR RGB

        f0_hsi = self.head_hsi(x_hsi)
        f0_rgb = self.head_rgb(x_rgb)

        # 初始融合
        f_fused_1 = self.fusion1(f0_hsi, f0_rgb)

        # 并行分支处理
        # 论文暗示 HSI 分支利用融合后的特征，RGB 分支利用 RGB 特征
        f_hsi_deep = self.hsi_blocks(f_fused_1)
        f_rgb_deep = self.rgb_blocks(f0_rgb)

        # 最终融合
        f_final = self.fusion2(f_hsi_deep, f_rgb_deep)

        return f_final


# ==============================================================================
# 4. 先验引导任意尺度上采样模块 (PGASUM) [cite: 205]
# ==============================================================================

class PGASUM(nn.Module):
    """
    Prior Guided Arbitrary-Scale Upsampling Module
    核心机制：Local Ensemble Attention + Implicit Function
    """

    def __init__(self, dim, out_channels):
        super(PGASUM, self).__init__()

        # Key 和 Value 的投影层 (处理 RGB Prior)
        # Input: feature(dim) + relative_coord(2)
        self.k_proj = nn.Linear(dim + 2, dim)
        self.v_proj = nn.Linear(dim + 2, dim)

        # 最终预测像素的 MLP
        self.mlp = MLP(in_dim=dim, out_dim=out_channels, hidden_list=[256, 256])

        self.dim = dim

    def query_prior(self, feat_hr, coords):
        """
        从 HR RGB 特征图中根据坐标采样特征
        feat_hr: [B, C, H_hr, W_hr]
        coords: [B, N, 2] (范围 -1 到 1)
        """
        # grid_sample 需要 [B, H, W, 2] 格式
        # 伪造维度以利用 grid_sample 进行并行采样
        b, n, _ = coords.shape
        samples = F.grid_sample(
            feat_hr,
            coords.view(b, 1, n, 2),
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        )  # Output: [B, C, 1, N]

        return samples.view(b, -1, n).permute(0, 2, 1)  # [B, N, C]

    def forward(self, f_hsi_lr, f_rgb_hr, coords, cell):
        """
        f_hsi_lr: FFEM 输出的 LR HSI 特征 [B, C, h, w]
        f_rgb_hr: STPG 输出的 HR RGB 特征 [B, C, H, W]
        coords: 查询坐标 [B, N, 2]
        cell: 像素大小 [B, N, 2]
        """
        b, n, _ = coords.shape

        # --- 1. 获取 Query (从 LR HSI 特征中采样) ---
        # LIIF 使用最近邻采样作为 Query code
        q_feat = F.grid_sample(
            f_hsi_lr,
            coords.view(b, 1, n, 2),
            mode='nearest',
            align_corners=False,
            padding_mode='border'
        ).view(b, self.dim, n).permute(0, 2, 1)  # [B, N, C]

        # --- 2. 获取 Key 和 Value (从 HR RGB Prior 中采样) ---
        # 采样得到空间对齐的 RGB 特征
        rgb_feat = self.query_prior(f_rgb_hr, coords)  # [B, N, C]

        # 构建相对坐标信息 (简化实现：直接拼接 cell size)
        # 论文 Eq. 7 使用 x_q - x_k，在 implicit function 中 cell size 常作为相对信息的代理
        kv_input = torch.cat([rgb_feat, cell], dim=-1)  # [B, N, C+2]

        k = self.k_proj(kv_input)  # [B, N, C]
        v = self.v_proj(kv_input)  # [B, N, C]

        # --- 3. Cross Attention (Eq. 5) ---
        # Attention = Softmax(Q * K^T / sqrt(d)) * V
        attn_score = (q_feat * k).sum(dim=-1, keepdim=True) / math.sqrt(self.dim)
        # 注意：这里简化了 Local Ensemble。
        # 标准 LIIF 会对每个 Query 找周围 4 个特征做 Ensemble。
        # 论文 Fig. 3 展示了 Cross Attention 是在 Local Region 进行的。
        # 为了代码简洁，这里展示 Point-wise 的 Attention 逻辑，
        # 在实际训练中，Local Ensemble 通常通过 unfold 或者多次 grid_sample 实现。

        # 这里实现简化的 Attention 融合
        f_attn = torch.sigmoid(attn_score) * v + q_feat  # Residual style fusion

        # --- 4. 预测像素值 ---
        pred = self.mlp(f_attn)  # [B, N, Out_Bands]

        return pred


# ==============================================================================
# 5. SPG-ASSR 整体网络架构 [cite: 147, 152]
# ==============================================================================

class SPG_ASSR(nn.Module):
    def __init__(self, n_bands=102, embed_dim=128):
        super(SPG_ASSR, self).__init__()

        # 模拟光谱响应函数 R (用于从 HSI 生成 LR RGB)
        self.spectral_response = nn.Conv2d(n_bands, 3, 1, bias=False)

        # 模块 1: STPG (Spatial Texture Prior Generator)
        self.stpg = STPG(embed_dim=embed_dim)

        # 模块 2: FFEM (Feature Fusion Enhance Module)
        self.ffem = FFEM(hsi_bands=n_bands, embed_dim=embed_dim)

        # 模块 3: PGASUM (Upsampling)
        self.pgasum = PGASUM(dim=embed_dim, out_channels=n_bands)

    def forward(self, x_lr_hsi, coords, cell):
        """
        x_lr_hsi: 输入的低分辨率 HSI [B, Bands, h, w]
        coords: 需要查询的高分辨率坐标点 [B, N, 2]
        cell: 坐标点的像素尺寸信息 [B, N, 2]
        """

        # 1. 生成 LR RGB [cite: 154]
        x_lr_rgb = self.spectral_response(x_lr_hsi)

        # 2. STPG 提取先验 [cite: 155]
        # 假设 STPG 包含上采样能力，输出的是 HR 尺度的特征图
        # 如果是推理阶段，x_lr_rgb 可能需要先被上采样到 HR 尺寸再输入 STPG，
        # 或者 STPG 内部包含 Super-Resolution 网络。
        # 为匹配 PGASUM 逻辑，这里假设输出为 HR 特征。
        # 在代码测试中，我们简单地对 LR RGB 进行插值模拟 STPG 的 SR 过程。
        h_lr, w_lr = x_lr_hsi.shape[-2:]
        # 这里的 scale 仅用于模拟，实际由 coords 决定
        # 假设 HR 特征足够大以覆盖 coords 范围
        f_rgb_hr = F.interpolate(
            self.stpg(x_lr_rgb),
            scale_factor=4,  # 假设 STPG 默认做 x4 SR 提取特征
            mode='bilinear',
            align_corners=False
        )

        # 3. FFEM 特征融合 [cite: 169]
        f_fused = self.ffem(x_lr_hsi, x_lr_rgb)

        # 4. PGASUM 任意尺度重建 [cite: 170]
        # 预测 coords 位置的 HSI 像素值
        pred_hsi = self.pgasum(f_fused, f_rgb_hr, coords, cell)

        # 5. 残差连接 (论文 Eq. 8: I_pred + I_nearest)
        # 需要获取 LR HSI 在 coords 处的双线性/最近邻插值结果作为基准
        res_hsi = F.grid_sample(
            x_lr_hsi,
            coords.view(x_lr_hsi.shape[0], 1, -1, 2),
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        ).view(x_lr_hsi.shape[0], -1, coords.shape[1]).permute(0, 2, 1)

        return pred_hsi + res_hsi


# ==============================================================================
# 测试运行块 (Demo)
# ==============================================================================
if __name__ == '__main__':
    # 设置参数
    B = 2  # Batch size
    C_hsi = 102  # HSI 波段数
    H_lr, W_lr = 32, 32
    N_pixels = 1000  # 随机查询的像素点数量

    # 实例化模型
    model = SPG_ASSR(n_bands=C_hsi, embed_dim=64)

    # 模拟输入数据
    lr_hsi = torch.randn(B, C_hsi, H_lr, W_lr)

    # 模拟坐标 (LIIF 格式: B x N x 2, 值域 [-1, 1])
    coords = torch.rand(B, N_pixels, 2) * 2 - 1

    # 模拟 Cell (像素大小信息)
    cell = torch.rand(B, N_pixels, 2)

    # 前向传播
    output = model(lr_hsi, coords, cell)

    print(f"输入 HSI 尺寸: {lr_hsi.shape}")
    print(f"查询坐标数量: {N_pixels}")
    print(f"输出 SR 结果尺寸: {output.shape}")  # 预期: [2, 1000, 102]