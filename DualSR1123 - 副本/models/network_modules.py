import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torchvision.models import VGG19_Weights


class HSIResAttentionBlock(nn.Module):
    """光谱 + 空间注意力残差块，适用于 HSI/特征图。

    结构：Conv3x3 -> ReLU -> Conv3x3 -> (Channel + Spatial Attention) -> 残差相加。
    不改变输入/输出的通道数和空间尺寸。
    """

    def __init__(self, channels, reduction=8):
        super(HSIResAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 光谱注意力（通道注意力，SE 样式）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # 空间注意力（2D 注意力）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        # 通道注意力
        ca = self.channel_attn(out)
        out = out * ca

        # 空间注意力
        avg_pool = torch.mean(out, dim=1, keepdim=True)
        max_pool, _ = torch.max(out, dim=1, keepdim=True)
        sa_input = torch.cat([avg_pool, max_pool], dim=1)
        sa = self.spatial_attn(sa_input)
        out = out * sa

        out = out + identity
        out = self.relu(out)
        return out


# --- 1. PatchEmbedding 类 ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int, img_height: int, img_width: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        num_patches_h = img_height // patch_size
        num_patches_w = img_width // patch_size
        self.num_patches = num_patches_h * num_patches_w

        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C_in, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions ({H},{W}) must be divisible by patch_size ({self.patch_size}).")

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_encoding
        return x


# --- 2. RGBFeatureExtractor128 (模块一) ---
class RGBFeatureExtractor128(nn.Module):
    def __init__(self):
        super(RGBFeatureExtractor128, self).__init__()
        vgg19_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # 冻结全部 VGG 参数，后续仅作为固定特征提取器使用
        for p in vgg19_features.parameters():
            p.requires_grad = False

        self.slice1 = nn.Sequential()
        for x in range(0, 8):  # 到 conv2_2 之后 (输出 128 通道)
            self.slice1.add_module(str(x), vgg19_features[x])

        self.slice2 = nn.Sequential()
        for x in range(8, 17):  # 从 conv3_1 到 conv3_4 之后 (输出 256 通道)
            self.slice2.add_module(str(x), vgg19_features[x])

        self.slice3 = nn.Sequential()
        for x in range(17, 26):  # 从 conv4_1 到 conv4_4 之后 (输出 512 通道)
            self.slice3.add_module(str(x), vgg19_features[x])

        self.spectral_downsample = nn.Conv2d(31, 3, kernel_size=3, padding=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 256 + 512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_lrhsi=None, x_rgb=None):
        # 支持直接传入 RCAN 输出的 RGB，或传入 LR-HSI 走原路径
        if x_rgb is not None:
            input_rgb = x_rgb
        else:
            x_upsampled = self.upsample(x_lrhsi)
            input_rgb = self.spectral_downsample(x_upsampled)

        # 对 VGG 输入做 ImageNet 归一化，避免数值过大
        mean = torch.tensor([0.485, 0.456, 0.406], device=input_rgb.device, dtype=input_rgb.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input_rgb.device, dtype=input_rgb.dtype).view(1, 3, 1, 1)
        input_rgb_norm = (input_rgb - mean) / std
        # VGG 特征在无梯度模式下提取，避免反向传播经过 VGG
        with torch.no_grad():
            f1 = self.slice1(input_rgb_norm)
            f2 = self.slice2(f1)
            f3 = self.slice3(f2)

        f1_resized = F.interpolate(f1, size=(128, 128), mode='bilinear', align_corners=False)
        f2_resized = F.interpolate(f2, size=(128, 128), mode='bilinear', align_corners=False)
        f3_resized = F.interpolate(f3, size=(128, 128), mode='bilinear', align_corners=False)

        fused_features = torch.cat((f1_resized, f2_resized, f3_resized), dim=1)
        output = self.fusion(fused_features)
        return output


# --- 3. RGBFeatureExtractor64 (模块二) ---
class RGBFeatureExtractor64(nn.Module):
    def __init__(self):
        super(RGBFeatureExtractor64, self).__init__()
        vgg19_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # 冻结全部 VGG 参数，后续仅作为固定特征提取器使用
        for p in vgg19_features.parameters():
            p.requires_grad = False

        self.slice1 = nn.Sequential()
        for x in range(0, 8):
            self.slice1.add_module(str(x), vgg19_features[x])

        self.slice2 = nn.Sequential()
        for x in range(8, 17):
            self.slice2.add_module(str(x), vgg19_features[x])

        self.slice3 = nn.Sequential()
        for x in range(17, 26):
            self.slice3.add_module(str(x), vgg19_features[x])

        self.spectral_downsample = nn.Conv2d(31, 3, kernel_size=3, padding=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 256 + 512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_lrhsi=None, x_rgb=None):
        # 支持直接传入 RCAN 输出的 RGB，或传入 LR-HSI 走原路径
        if x_rgb is not None:

            input_rgb = x_rgb
        else:

            x_upsampled = self.upsample(x_lrhsi)
            input_rgb = self.spectral_downsample(x_upsampled)

        # 对 VGG 输入做 ImageNet 归一化，避免数值过大
        mean = torch.tensor([0.485, 0.456, 0.406], device=input_rgb.device, dtype=input_rgb.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=input_rgb.device, dtype=input_rgb.dtype).view(1, 3, 1, 1)
        input_rgb_norm = (input_rgb - mean) / std
        # VGG 特征在无梯度模式下提取，避免反向传播经过 VGG
        with torch.no_grad():
            f1 = self.slice1(input_rgb_norm)
            f2 = self.slice2(f1)
            f3 = self.slice3(f2)

        f1_resized = F.interpolate(f1, size=(64, 64), mode='bilinear', align_corners=False)
        f2_resized = F.interpolate(f2, size=(64, 64), mode='bilinear', align_corners=False)
        f3_resized = F.interpolate(f3, size=(64, 64), mode='bilinear', align_corners=False)

        fused_features = torch.cat((f1_resized, f2_resized, f3_resized), dim=1)
        output = self.fusion(fused_features)
        return output


# --- 4. LrHsiFeatureExtractor64 (模块三) ---
class LrHsiFeatureExtractor64(nn.Module):
    def __init__(self):
        super(LrHsiFeatureExtractor64, self).__init__()
        self.conv1 = nn.Conv2d(31, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 使用两个光谱+空间注意力残差块增强 HSI 低分辨率特征
        self.attention_blocks = nn.Sequential(
            HSIResAttentionBlock(64),
            HSIResAttentionBlock(64)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_lrhsi):


        x = self.relu(self.conv1(x_lrhsi))
        x = self.attention_blocks(x)
        x_upsampled = self.upsample(x)  # (N, 64, 64, 64)
        return x_upsampled


# --- 5. LrHsiFeatureFusion128 (模块四) ---
class LrHsiFeatureFusion128(nn.Module):
    def __init__(self):
        super(LrHsiFeatureFusion128, self).__init__()
        self.upsample_lrhsi = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample_lrhsi_feature = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 先做一次基础卷积融合，再通过注意力残差块增强
        self.fusion_head = nn.Sequential(
            nn.Conv2d(31 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.fusion_body = nn.Sequential(
            HSIResAttentionBlock(64),
            HSIResAttentionBlock(64)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, lrhsi_input, lrhsi_feature_64):


        lrhsi_upsampled = self.upsample_lrhsi(lrhsi_input)  # (N, 31, 128, 128)
        lrhsi_feature_128 = self.upsample_lrhsi_feature(lrhsi_feature_64)  # (N, 64, 128, 128)

        combined_features = torch.cat((lrhsi_upsampled, lrhsi_feature_128), dim=1)
        x = self.fusion_head(combined_features)
        output = self.fusion_body(x)  # (N, 64, 128, 128)
        return output


# --- 6. CrossAttentionImageNetV5 (模块五) ---
class CrossAttentionImageNetV5(nn.Module):
    def __init__(self,
                 query_feature_height: int = 64,
                 query_feature_width: int = 64,
                 context_feature_height: int = 64,
                 context_feature_width: int = 64,
                 patch_size: int = 4,
                 query_channels: int = 64,
                 context_channels: int = 64,
                 output_channels: int = 31,
                 d_model: int = 128,
                 num_heads: int = 2,
                 num_transformer_blocks: int = 1):
        super(CrossAttentionImageNetV5, self).__init__()

        self.query_feature_height = query_feature_height
        self.query_feature_width = query_feature_width
        self.context_feature_height = context_feature_height
        self.context_feature_width = context_feature_width
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_transformer_blocks = num_transformer_blocks

        if query_feature_height % patch_size != 0 or query_feature_width % patch_size != 0:
            raise ValueError("Query 特征图尺寸必须能被 patch_size 整除。")
        if context_feature_height % patch_size != 0 or context_feature_width % patch_size != 0:
            raise ValueError("Context 特征图尺寸必须能被 patch_size 整除。")

        self.num_patches_query = (query_feature_height // patch_size) * (query_feature_width // patch_size)
        self.num_patches_context = (context_feature_height // patch_size) * (context_feature_width // patch_size)

        self.query_embedding = PatchEmbedding(query_channels, patch_size, d_model, query_feature_height,
                                              query_feature_width)
        self.context_embedding = PatchEmbedding(context_channels, patch_size, d_model, context_feature_height,
                                                context_feature_width)

        self.cross_attention_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model)
                ),
                nn.LayerNorm(d_model)
            ]) for _ in range(num_transformer_blocks)
        ])

        self.linear_to_pixelshuffle_channels = nn.Linear(d_model, output_channels * (patch_size ** 2))
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

        self.final_upsample = nn.Identity()  # 64x64 输入 -> 64x64 输出

        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.normal_(m.in_proj_weight, mean=0.0, std=math.sqrt(2.0 / (m.embed_dim + m.embed_dim)))
                if m.out_proj.weight is not None:
                    nn.init.normal_(m.out_proj.weight, mean=0.0, std=math.sqrt(2.0 / (m.embed_dim + m.embed_dim)))

    def forward(self, query_feature: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:


        N = query_feature.shape[0]

        Q = torch.nan_to_num(self.query_embedding(query_feature))
        K = torch.nan_to_num(self.context_embedding(context_feature))
        V = K

        # 在 FP32 下执行注意力，避免 AMP(FP16) 下的数值溢出导致 NaN
        orig_dtype = Q.dtype
        # 修正新版 API：仅在注意力段禁用 AMP（保持 FP32）
        with torch.amp.autocast('cuda:1', enabled=False):
            processed_seq_fp32 = Q.float()
            K_fp32 = K.float()
            V_fp32 = V.float()
            for i in range(self.num_transformer_blocks):
                attn_block = self.cross_attention_blocks[i][0]
                norm1_layer = self.cross_attention_blocks[i][1]
                ffn_layer = self.cross_attention_blocks[i][2]
                norm2_layer = self.cross_attention_blocks[i][3]

                attn_output, _ = attn_block(query=processed_seq_fp32, key=K_fp32, value=V_fp32)
                attn_output_res = norm1_layer(processed_seq_fp32 + attn_output)
                ff_output = ffn_layer(attn_output_res)
                processed_seq_fp32 = norm2_layer(attn_output_res + ff_output)
            processed_seq = processed_seq_fp32.to(orig_dtype)

        # --- 最终投影和图像重建 ---
        # 1. 投影到适合 PixelShuffle 的通道数
        # (N, num_patches_lr, d_model) -> (N, num_patches_lr, output_channels * (patch_size^2))
        pixelshuffle_input_seq = torch.nan_to_num(self.linear_to_pixelshuffle_channels(processed_seq))

        # 2. 将序列重塑为低分辨率的特征图，以便进行 PixelShuffle
        # (N, num_patches_lr, output_channels * (patch_size^2))
        # -> (N, output_channels * (patch_size^2), num_patches_h, num_patches_w)
        pixelshuffle_input_flat = pixelshuffle_input_seq.permute(0, 2, 1)

        H_lr_patches = self.query_feature_height // self.patch_size
        W_lr_patches = self.query_feature_width // self.patch_size

        intermediate_map_for_pixelshuffle = pixelshuffle_input_flat.view(
            N,
            self.linear_to_pixelshuffle_channels.out_features,
            H_lr_patches,
            W_lr_patches
        )

        # 3. 使用 PixelShuffle 将每个 patch 内部的通道展开为空间信息
        # 此时图像尺寸应该是 (N, output_channels, query_feature_height, query_feature_width) = (N, 31, 64, 64)
        output_after_pixelshuffle = self.pixel_shuffle(intermediate_map_for_pixelshuffle)

        # 4. 后续卷积层进行精炼
        output_image = self.final_upsample(output_after_pixelshuffle)
        output_image = torch.nan_to_num(self.final_conv(output_image))

        return output_image


# --- 7. CrossAttentionImageNetV6 (模块六) ---
class CrossAttentionImageNetV6(nn.Module):
    def __init__(self,
                 img_height: int = 128,
                 img_width: int = 128,
                 patch_size: int = 8,
                 query_channels: int = 64,
                 context_channels: int = 64,
                 output_channels: int = 31,
                 d_model: int = 128,
                 num_heads: int = 2,
                 num_transformer_blocks: int = 1):
        super(CrossAttentionImageNetV6, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_transformer_blocks = num_transformer_blocks

        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch_size 整除。")

        self.num_patches = (img_height // patch_size) * (img_width // patch_size)

        self.query_embedding = PatchEmbedding(query_channels, patch_size, d_model, img_height, img_width)
        self.context_embedding = PatchEmbedding(context_channels, patch_size, d_model, img_height, img_width)

        self.cross_attention_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model)
                ),
                nn.LayerNorm(d_model)
            ]) for _ in range(num_transformer_blocks)
        ])

        self.linear_to_pixelshuffle_channels = nn.Linear(d_model, output_channels * (patch_size ** 2))
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

        self.final_upsample = nn.Identity()  # 128x128 输入 -> 128x128 输出

        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.normal_(m.in_proj_weight, mean=0.0, std=math.sqrt(2.0 / (m.embed_dim + m.embed_dim)))
                if m.out_proj.weight is not None:
                    nn.init.normal_(m.out_proj.weight, mean=0.0, std=math.sqrt(2.0 / (m.embed_dim + m.embed_dim)))

    def forward(self, context_image: torch.Tensor, query_image: torch.Tensor) -> torch.Tensor:


        N = query_image.shape[0]

        Q = torch.nan_to_num(self.query_embedding(query_image))
        K = torch.nan_to_num(self.context_embedding(context_image))
        V = K

        # 在 FP32 下执行注意力，避免 AMP(FP16) 下的数值溢出导致 NaN
        orig_dtype = Q.dtype
        # 修正新版 API：仅在注意力段禁用 AMP（保持 FP32）
        with torch.amp.autocast('cuda:1', enabled=False):
            processed_seq_fp32 = Q.float()
            K_fp32 = K.float()
            V_fp32 = V.float()
            for i in range(self.num_transformer_blocks):
                attn_block = self.cross_attention_blocks[i][0]
                norm1_layer = self.cross_attention_blocks[i][1]
                ffn_layer = self.cross_attention_blocks[i][2]
                norm2_layer = self.cross_attention_blocks[i][3]

                attn_output, _ = attn_block(query=processed_seq_fp32, key=K_fp32, value=V_fp32)
                attn_output_res = norm1_layer(processed_seq_fp32 + attn_output)
                ff_output = ffn_layer(attn_output_res)
                processed_seq_fp32 = norm2_layer(attn_output_res + ff_output)
            processed_seq = processed_seq_fp32.to(orig_dtype)

        # --- 最终投影和图像重建 ---
        # 1. 投影到适合 PixelShuffle 的通道数
        # (N, num_patches_lr, d_model) -> (N, num_patches_lr, output_channels * (patch_size^2))
        pixelshuffle_input_seq = torch.nan_to_num(self.linear_to_pixelshuffle_channels(processed_seq))

        # 2. 将序列重塑为低分辨率的特征图，以便进行 PixelShuffle
        # (N, num_patches_lr, output_channels * (patch_size^2))
        # -> (N, output_channels * (patch_size^2), num_patches_h, num_patches_w)
        pixelshuffle_input_flat = pixelshuffle_input_seq.permute(0, 2, 1)

        H_lr_patches = self.img_height // self.patch_size
        W_lr_patches = self.img_width // self.patch_size

        intermediate_map_for_pixelshuffle = pixelshuffle_input_flat.view(
            N,
            self.linear_to_pixelshuffle_channels.out_features,
            H_lr_patches,
            W_lr_patches
        )

        # 3. 使用 PixelShuffle 将每个 patch 内部的通道展开为空间信息
        # 此时图像尺寸应该是 (N, output_channels, img_height, img_width) = (N, 31, 128, 128)
        output_after_pixelshuffle = self.pixel_shuffle(intermediate_map_for_pixelshuffle)

        # 4. 后续卷积层进行精炼
        output_image = self.final_upsample(output_after_pixelshuffle)
        output_image = torch.nan_to_num(self.final_conv(output_image))

        return output_image


# --- 8. FineStageNet (模块七) - 基于小波融合逻辑 ---
class FineStageNet(nn.Module):
    def __init__(self, low_res_channels=31, high_res_channels=31, texture_channels=1, output_channels=31):
        super(FineStageNet, self).__init__()

        # 低通滤波模拟小波变换的低频提取
        # kernel_size=3, stride=2, padding=1 会将 HxW 尺寸减半 (例如 64x64 -> 32x32)
        # groups=in_channels 实现深度可分离卷积，每个通道独立处理，模拟小波变换对每个通道独立操作
        self.conv_lp_lr = nn.Conv2d(low_res_channels, low_res_channels, kernel_size=3, stride=2, padding=1,
                                    groups=low_res_channels)
        self.conv_lp_hr = nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, stride=2, padding=1,
                                    groups=high_res_channels)

        # 低频特征上采样 (从 32x32 到 128x128) - 使用转置卷积获得更好的重建
        self.upsample_low_freq_64 = nn.ConvTranspose2d(low_res_channels, low_res_channels,
                                                       kernel_size=4, stride=4, padding=0)  # 32x32 -> 128x128
        self.upsample_low_freq_128 = nn.ConvTranspose2d(high_res_channels, high_res_channels,
                                                        kernel_size=4, stride=4, padding=0)  # 32x32 -> 128x128

        # �� 改进的注意力融合机制 - 修复维度匹配
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(low_res_channels + high_res_channels, low_res_channels, kernel_size=1),  # 输出31通道
            nn.Sigmoid()
        )

        # �� 简单多尺度分支：在 128x128 特征上再显式引入一个 64x64 尺度
        # hrhsi_128 -> AvgPool2d(2) 到 64x64 -> Conv -> 上采样回 128x128
        self.ms_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 128x128 -> 64x64
        self.ms_conv_64 = nn.Sequential(
            nn.Conv2d(high_res_channels, high_res_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.ms_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 64x64 -> 128x128

        # 最终融合层 - 更深的网络获得更好的表征
        # 融合后的总通道数 =
        #   增强低频特征通道 (low_res_channels)
        #   + hrhsi_128 (high_res_channels)
        #   + 多尺度分支 hrhsi_128_ms (high_res_channels)
        #   + texture_map (texture_channels)
        fused_input_channels = low_res_channels + high_res_channels + high_res_channels + texture_channels

        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(fused_input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hrhsi_64, hrhsi_128, texture_map):
        # 输入维度:
        # hrhsi_64: (N, 31, 64, 64)  # 第一个 Transformer 模块 (V5) 的输出
        # hrhsi_128: (N, 31, 128, 128)  # 第二个 Transformer 模块 (V6) 的输出
        # texture_map: (N, 1, 128, 128)  # 原始纹理信息图

        # --- 小波变换提取低频信息 ---
        # 提取 hrhsi_64 的低频 (N, 31, 32, 32)
        hrhsi_64_low_freq = self.conv_lp_lr(hrhsi_64)

        # 将 hrhsi_128 下采样到 64x64，以便提取可比较的低频信息
        hrhsi_128_down_64 = F.interpolate(hrhsi_128, size=(64, 64), mode='bilinear', align_corners=False)
        # 提取 hrhsi_128 (下采样后) 的低频 (N, 31, 32, 32)
        hrhsi_128_low_freq = self.conv_lp_hr(hrhsi_128_down_64)

        # 上采样低频特征到 128x128 以与最终输出对齐 - 使用改进的转置卷积
        hrhsi_64_low_freq_upsampled = self.upsample_low_freq_64(hrhsi_64_low_freq)  # 32x32 -> 128x128
        hrhsi_128_low_freq_upsampled = self.upsample_low_freq_128(hrhsi_128_low_freq)  # 32x32 -> 128x128

        # �� 简化的加权融合 - 确保维度稳定
        low_freq_combined = torch.cat([hrhsi_64_low_freq_upsampled, hrhsi_128_low_freq_upsampled], dim=1)
        attention_weights = self.attention_fusion(low_freq_combined)  # (N, 31, 128, 128)

        # 应用注意力权重到hrhsi_64的低频分量
        enhanced_low_freq_64 = hrhsi_64_low_freq_upsampled * attention_weights

        # --- 多尺度分支 ---
        # 从 hrhsi_128 再提取一个 64x64 尺度的特征，并上采样回 128x128
        hrhsi_128_ms_64 = self.ms_pool(hrhsi_128)                  # (N, 31, 64, 64)
        hrhsi_128_ms_64 = self.ms_conv_64(hrhsi_128_ms_64)         # (N, 31, 64, 64)
        hrhsi_128_ms_128 = self.ms_upsample(hrhsi_128_ms_64)       # (N, 31, 128, 128)

        # --- 最终融合 ---
        # 沿着通道维度拼接增强的低频特征、原始 hrhsi_128、多尺度分支和 texture_map
        fused_features = torch.cat([enhanced_low_freq_64, hrhsi_128, hrhsi_128_ms_128, texture_map], dim=1)
        # fused_features 形状为 (N, 31 + 31 + 31 + 1, 128, 128) = (N, 94, 128, 128)

        # 通过最终卷积层调整通道数到 31
        final_output_hsi = self.final_fusion_conv(fused_features)
        # 在 FineStageNet 内部加入一个简单的跳连接，将 Transformer 输出 hrhsi_128 直接加到最终重建结果上
        final_output_hsi = final_output_hsi + hrhsi_128
        # 最终 output_hsi 形状为 (N, 31, 128, 128)

        # 返回最终的高光谱图像和两个低频分量，用于计算损失
        return final_output_hsi, hrhsi_64_low_freq_upsampled, hrhsi_128_low_freq_upsampled