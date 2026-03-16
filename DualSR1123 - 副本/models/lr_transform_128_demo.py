import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 1. 定义 PatchEmbedding 类 ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int, img_height: int, img_width: int):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # 使用卷积层进行 Patch 提取和线性投影
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # 计算 Patch 的数量
        num_patches_h = img_height // patch_size
        num_patches_w = img_width // patch_size
        self.num_patches = num_patches_h * num_patches_w

        # 可学习的位置编码
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, H, W)
        N, C_in, H, W = x.shape

        # 验证图像尺寸是否能被 patch_size 整除
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"图像尺寸 ({H},{W}) 必须能被 patch_size ({self.patch_size}) 整除。")

        # 1. 卷积提取 Patch 特征并投影到 d_model 维度
        x = self.proj(x)

        # 2. 展平空间维度，并转置以符合 Transformer 的 (batch_size, sequence_length, embedding_dim) 格式
        x = x.flatten(2).transpose(1, 2)

        # 3. 添加位置编码
        x = x + self.positional_encoding
        return x


# --- 2. 修改 CrossAttentionImageNetV6 类 ---
class CrossAttentionImageNetV6(nn.Module):
    def __init__(self,
                 img_height: int = 128,  # 输入/输出图像高度
                 img_width: int = 128,  # 输入/输出图像宽度
                 patch_size: int = 16,  # 每个 Patch 的大小，例如 16x16
                 query_channels: int = 64,  # Query 图像（高光谱）的原始通道数
                 context_channels: int = 64,  # Context 图像（上下文）的原始通道数
                 output_channels: int = 31,  # 最终输出图像的通道数
                 d_model: int = 512,        # Transformer 的嵌入维度
                 num_heads: int = 8,        # 多头注意力的头数
                 num_transformer_blocks: int = 2):  # 增加 Transformer 块数
        super(CrossAttentionImageNetV6, self).__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_transformer_blocks = num_transformer_blocks

        # 验证图像尺寸是否能被 patch_size 整除
        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError(f"图像尺寸 ({img_height},{img_width}) 必须能被 patch_size ({patch_size}) 整除。")

        # 计算 Patch 数量
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)

        # 1. Patch Embedding 层
        self.query_embedding = PatchEmbedding(query_channels, patch_size, d_model, img_height, img_width)
        self.context_embedding = PatchEmbedding(context_channels, patch_size, d_model, img_height, img_width)

        # 2. Transformer 交叉注意力模块
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

        # 3. 最终投影层和图像重建
        self.linear_to_pixelshuffle_channels = nn.Linear(d_model, 64 * (patch_size ** 2))  # 先投影到 64 通道 * patch_size^2
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

        # 新增：通道优化层，从 64 通道调整到 31 通道
        self.channel_transition = nn.Conv2d(64, output_channels, kernel_size=1)

        # 额外的上采样层（如果需要）
        self.final_upsample = nn.Identity()

        # 最后的卷积层用于细化
        self.final_conv = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, context_image: torch.Tensor, query_image: torch.Tensor) -> torch.Tensor:
        # 输入维度:
        # context_image: (N, context_channels, img_height, img_width) -> (N, 64, 128, 128) (K, V)
        # query_image: (N, query_channels, img_height, img_width) -> (N, 64, 128, 128) (Q)

        N = context_image.shape[0]

        # --- 1. 应用 Patch Embedding ---
        Q = self.query_embedding(query_image)  # 形状: (N, num_patches, d_model)
        K = self.context_embedding(context_image)  # 形状: (N, num_patches, d_model)
        V = K  # 对于交叉注意力，K 和 V 通常来自同一源

        # --- 2. Transformer 交叉注意力块 ---
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
        pixelshuffle_input_seq = self.linear_to_pixelshuffle_channels(processed_seq)
        pixelshuffle_input_flat = pixelshuffle_input_seq.permute(0, 2, 1)

        H_patches = self.img_height // self.patch_size
        W_patches = self.img_width // self.patch_size

        intermediate_map_for_pixelshuffle = pixelshuffle_input_flat.view(
            N,
            self.linear_to_pixelshuffle_channels.out_features,
            H_patches,
            W_patches
        )

        output_after_pixelshuffle = self.pixel_shuffle(intermediate_map_for_pixelshuffle)

        # 通道优化层，从 64 通道调整到 31 通道
        output_transition = self.channel_transition(output_after_pixelshuffle)

        output_image = self.final_upsample(output_transition)
        output_image = self.final_conv(output_image)

        return output_image


# --- 演示部分 ---
if __name__ == "__main__":
    # 定义模型参数
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    PATCH_SIZE = 16  # 每个 Patch 将是 16x16 像素
    QUERY_CHANNELS = 64  # Query 图像（高光谱）的通道数
    CONTEXT_CHANNELS = 64  # Context 图像（上下文）的通道数
    OUTPUT_CHANNELS = 31  # 最终输出图像的通道数
    D_MODEL = 512  # Transformer 嵌入维度
    NUM_HEADS = 8  # 注意力头数
    NUM_TRANSFORMER_BLOCKS = 2  # 增加 Transformer 块数

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 实例化模型并移动到设备
    model = CrossAttentionImageNetV6(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        patch_size=PATCH_SIZE,
        query_channels=QUERY_CHANNELS,
        context_channels=CONTEXT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS
    ).to(device)

    # 创建模拟输入张量
    batch_size = 1  # 测试批量处理
    dummy_context_image = torch.randn(batch_size, CONTEXT_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)
    print(f"模拟上下文图像 (K, V) 输入尺寸: {dummy_context_image.shape}")

    dummy_query_image = torch.randn(batch_size, QUERY_CHANNELS, IMG_HEIGHT, IMG_WIDTH).to(device)
    print(f"模拟高光谱图像 (Q) 输入尺寸: {dummy_query_image.shape}")

    # 将模型设置为评估模式
    model.eval()
    # 关闭梯度计算，节省内存并加速推理
    with torch.no_grad():
        output = model(dummy_context_image, dummy_query_image)

    print(f"最终输出尺寸: {output.shape}")

    # 验证输出尺寸是否符合要求 31*128*128
    expected_output_shape = (batch_size, OUTPUT_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    assert output.shape == expected_output_shape, "输出尺寸不匹配预期！"
    print("模型输出尺寸符合预期！")

    # 可以打印模型结构，以便更好地理解
    # print(model)