import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 1. PatchEmbedding 类 (无变化) ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int, img_height: int, img_width: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # 使用卷积层进行Patch提取和线性投影
        # kernel_size=patch_size, stride=patch_size 确保不重叠地提取每个patch
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # 计算Patch的数量
        num_patches_h = img_height // patch_size
        num_patches_w = img_width // patch_size
        self.num_patches = num_patches_h * num_patches_w

        # 可学习的位置编码
        # nn.Parameter 会使其成为模型参数，随训练更新
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, H, W)
        N, C_in, H, W = x.shape

        # 确保图像尺寸是patch_size的整数倍
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions ({H},{W}) must be divisible by patch_size ({self.patch_size}).")

        # 1. 卷积提取patch特征并投影到d_model维度
        # 输出形状: (N, d_model, H/patch_size, W/patch_size)
        x = self.proj(x)

        # 2. 展平空间维度，并转置以符合Transformer的(batch_size, sequence_length, embedding_dim)格式
        # x.flatten(2) 会将 (H/patch_size, W/patch_size) 展平
        # 形状: (N, d_model, num_patches)
        x = x.flatten(2)
        # 转置维度: (N, num_patches, d_model)
        x = x.transpose(1, 2)

        # 3. 添加位置编码
        x = x + self.positional_encoding
        return x


# --- 2. 修改 CrossAttentionImageNetV5 类 ---
class CrossAttentionImageNetV5(nn.Module):
    def __init__(self,
                 img_height: int = 64,  # 低分辨率特征图像高度 (Q 和 K, V)
                 img_width: int = 64,   # 低分辨率特征图像宽度 (Q 和 K, V)
                 patch_size: int = 4,   # Patch 的大小，例如 4x4
                 query_channels: int = 64,  # LR-Feature 的通道数 (Q)
                 context_channels: int = 64,  # RGB-Feature 的通道数 (K, V)
                 output_channels: int = 31,  # 最终输出的 HR-HSI 通道数
                 d_model: int = 512,        # Transformer 的嵌入维度
                 num_heads: int = 8,        # 多头注意力的头数
                 num_transformer_blocks: int = 2):  # 增加 Transformer 块数
        super(CrossAttentionImageNetV5, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_transformer_blocks = num_transformer_blocks

        # 验证输入图像尺寸是否能被 patch_size 整除
        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError("图像尺寸必须能被 patch_size 整除。")

        # 计算 Patch 数量作为 Transformer 的序列长度
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)

        # 1. Patch Embedding 层
        # Query (Q) 来自低分辨率特征 (LR-Feature)
        self.query_embedding = PatchEmbedding(query_channels, patch_size, d_model, img_height, img_width)
        # Key (K) 和 Value (V) 来自高分辨率 RGB 特征 (RGB-Feature)
        self.context_embedding = PatchEmbedding(context_channels, patch_size, d_model, img_height, img_width)

        # 2. Transformer 交叉注意力模块 (可以堆叠多个)
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

        # 3. 最终的投影层和图像重建
        # Transformer 的输出是 (N, num_patches, d_model)。
        # 我们要重建一个 (N, output_channels, img_height, img_width) 的图像。
        self.linear_to_pixelshuffle_channels = nn.Linear(d_model, output_channels * (patch_size ** 2))
        self.pixel_shuffle = nn.PixelShuffle(patch_size)  # 每次上采样 factor 是 patch_size

        # 额外的上采样层（如果需要）
        # 当前设计中，输入和输出尺寸相同 (64x64)，因此无需额外上采样
        self.final_upsample = nn.Identity()

        # 最后的卷积层用于细化
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, lr_feature: torch.Tensor, rgb_feature: torch.Tensor) -> torch.Tensor:
        # 输入维度:
        # lr_feature: (N, query_channels, img_height, img_width) -> (N, 64, 64, 64) (Query)
        # rgb_feature: (N, context_channels, img_height, img_width) -> (N, 64, 64, 64) (Key, Value)

        N = lr_feature.shape[0]

        # --- 1. Patch Embedding ---
        Q = self.query_embedding(lr_feature)  # 形状: (N, num_patches, d_model)
        K = self.context_embedding(rgb_feature)  # 形状: (N, num_patches, d_model)
        V = K  # 对于交叉注意力，K和V通常来自同一源

        # --- 2. 交叉注意力计算 (可以堆叠多个块) ---
        processed_seq = Q
        for i in range(self.num_transformer_blocks):
            attn_block = self.cross_attention_blocks[i][0]
            norm1_layer = self.cross_attention_blocks[i][1]
            ffn_layer = self.cross_attention_blocks[i][2]
            norm2_layer = self.cross_attention_blocks[i][3]

            attn_output, _ = attn_block(query=processed_seq, key=K, value=V)
            attn_output_res = norm1_layer(processed_seq + attn_output)
            ff_output = ffn_layer(attn_output_res)
            processed_seq = norm2_layer(attn_output_res + ff_output)

        # --- 3. 最终投影和图像重建 ---
        # 1. 投影到适合 PixelShuffle 的通道数
        # (N, num_patches, d_model) -> (N, num_patches, output_channels * (patch_size^2))
        pixelshuffle_input_seq = self.linear_to_pixelshuffle_channels(processed_seq)

        # 2. 将序列重塑为低分辨率的特征图，以便进行 PixelShuffle
        # (N, num_patches, output_channels * (patch_size^2))
        # -> (N, output_channels * (patch_size^2), num_patches_h, num_patches_w)
        pixelshuffle_input_flat = pixelshuffle_input_seq.permute(0, 2, 1)

        H_patches = self.img_height // self.patch_size
        W_patches = self.img_width // self.patch_size

        intermediate_map_for_pixelshuffle = pixelshuffle_input_flat.view(
            N,
            self.linear_to_pixelshuffle_channels.out_features,
            H_patches,
            W_patches
        )

        # 3. 使用 PixelShuffle 进行重建
        # 期望输出形状: (N, output_channels, img_height, img_width) 即 (N, 31, 64, 64)
        output_after_pixelshuffle = self.pixel_shuffle(intermediate_map_for_pixelshuffle)

        # 4. 最后的卷积层进行细化
        output_image = self.final_upsample(output_after_pixelshuffle)
        output_image = self.final_conv(output_image)

        return output_image


# --- 演示部分 ---
if __name__ == "__main__":
    # 定义模型参数
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    PATCH_SIZE = 4          # 例如 4x4 的 Patch
    QUERY_CHANNELS = 64     # LR-Feature 的通道数 (Q)
    CONTEXT_CHANNELS = 64   # RGB-Feature 的通道数 (K, V)
    OUTPUT_CHANNELS = 31    # 最终输出 HR-HSI 的通道数
    D_MODEL = 512           # Transformer 的嵌入维度
    NUM_HEADS = 8           # 注意力头数
    NUM_TRANSFORMER_BLOCKS = 2  # 增加 Transformer 块数

    # 实例化模型
    model = CrossAttentionImageNetV5(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        patch_size=PATCH_SIZE,
        query_channels=QUERY_CHANNELS,
        context_channels=CONTEXT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS
    )

    # 将模型移动到 GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已移动到: {device}")

    # 创建模拟输入张量
    dummy_lr_feature = torch.randn(1, QUERY_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)
    print(f"模拟高光谱图像 (Q) 输入尺寸: {dummy_lr_feature.shape}")

    dummy_rgb_feature = torch.randn(1, CONTEXT_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)
    print(f"模拟上下文图像 (K, V) 输入尺寸: {dummy_rgb_feature.shape}")

    # 将模型设置为评估模式
    model.eval()
    # 关闭梯度计算，节省内存并加速推理
    with torch.no_grad():
        output = model(dummy_lr_feature, dummy_rgb_feature)

    print(f"最终输出尺寸: {output.shape}")

    # 验证输出尺寸是否符合要求 31*64*64
    expected_output_shape = (1, OUTPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    assert output.shape == expected_output_shape, "输出尺寸不匹配预期！"
    print("模型输出尺寸符合预期！")

    # 可以打印模型结构，以便更好地理解
    # print(model)