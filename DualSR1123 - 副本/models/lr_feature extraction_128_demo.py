import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 辅助模块 (Auxiliary Modules) ---

class PatchEmbedding(nn.Module):
    def __init__(self, img_height: int, img_width: int, patch_size: int, in_channels: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError(f"图像尺寸 ({img_height}x{img_width}) 必须能被 patch_size ({patch_size}) 整除。")

        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        x = self.proj(x)  # (N, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 替换 ReLU 为 GELU

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ChannelExpansionNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 31,
                 use_transposed_as_final: bool = False):
        super(ChannelExpansionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.use_transposed_as_final = use_transposed_as_final
        if self.use_transposed_as_final:
            self.final_layer = nn.ConvTranspose2d(256, out_channels, kernel_size=1, stride=1)
        else:
            self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        output = self.final_layer(x)
        return output

# --- 主要模块 (Main Module) ---

class MultiInputImageTransformNet(nn.Module):
    def __init__(self,
                 in_channels_hsi: int = 31,
                 hsi_img_size: int = 32,
                 patch_size_hsi: int = 2,  # 调整为 2 以保留更多细节
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 num_transformer_blocks: int = 2,
                 aux_input_channels: int = 64,
                 final_upsampling_method: str = 'convt'):
        super(MultiInputImageTransformNet, self).__init__()
        self.embed_dim = embed_dim
        self.hsi_img_size = hsi_img_size
        self.patch_size_hsi = patch_size_hsi
        self.num_patches_side = (hsi_img_size // patch_size_hsi)
        self.final_upsampling_method = final_upsampling_method
        self.output_channels = 64

        self.patch_embed = PatchEmbedding(hsi_img_size, hsi_img_size, patch_size_hsi, in_channels_hsi, embed_dim)
        self.pos_embedding = PositionalEncoding(d_model=embed_dim, max_len=self.num_patches_side**2)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True, norm_first=True),
            num_layers=num_transformer_blocks
        )
        self.vit_output_to_channels_64 = nn.Conv2d(embed_dim, 64, kernel_size=1)
        self.bn_vit_output_channels = nn.BatchNorm2d(64)
        self.upsample_to_64x64_spatial = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8)
        self.bn_upsample_64x64 = nn.BatchNorm2d(64)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 + aux_input_channels, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, self.output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.GELU()
        )
        if self.final_upsampling_method == 'convt':
            self.final_upsampler = nn.Sequential(
                nn.ConvTranspose2d(self.output_channels, self.output_channels, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1)
            )
        elif self.final_upsampling_method == 'pixelshuffle':
            self.pixelshuffle_preconv = nn.Conv2d(self.output_channels, self.output_channels * 4, kernel_size=1)
            self.final_upsampler = nn.Sequential(
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(self.output_channels, self.output_channels, kernel_size=3, padding=1)
            )
        else:
            raise ValueError(f"不支持的上采样方法: {final_upsampling_method}。请选择 'convt' 或 'pixelshuffle'。")
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_main, x_aux_features):
        # x_main: (N, 31, 32, 32)
        # x_aux_features: (N, 64, 64, 64)

        # 阶段 1: HSI 主输入特征提取 (ViT 风格)
        x_patches = self.patch_embed(x_main)
        x = self.pos_embedding(x_patches)  # 添加位置编码
        x_transformer = self.transformer_encoder(x)
        batch_size = x_transformer.shape[0]
        x_reshaped = x_transformer.permute(0, 2, 1).reshape(batch_size, self.embed_dim, self.num_patches_side, self.num_patches_side)
        x_channels_adjusted = F.gelu(self.bn_vit_output_channels(self.vit_output_to_channels_64(x_reshaped)))
        intermediate_main_features = F.gelu(self.bn_upsample_64x64(self.upsample_to_64x64_spatial(x_channels_adjusted)))

        # 融合阶段
        fused_features_input = torch.cat([intermediate_main_features, x_aux_features], dim=1)
        fused_and_processed_features = self.fusion_conv(fused_features_input)

        # 阶段 2: 最终上采样
        if self.final_upsampling_method == 'pixelshuffle':
            x = self.pixelshuffle_preconv(fused_and_processed_features)
            final_output = self.final_upsampler(x)
        else:
            final_output = self.final_upsampler(fused_and_processed_features)

        return final_output

# --- 第二段代码 (ChannelExpansionNet) 保持不变 ---

# --- 演示部分 (Demo) ---
if __name__ == "__main__":
    print("--- MultiInputImageTransformNet (HSI HSI 双输入融合转换) 演示 ---")

    IN_CHANNELS_HSI = 31
    HSI_IMG_SIZE = 32
    PATCH_SIZE_HSI = 2
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_TRANSFORMER_BLOCKS = 2
    AUX_INPUT_CHANNELS = 64
    BATCH_SIZE = 1

    print("\n--- 模式一：使用 ConvTranspose2d 作为最终上采样 ---")
    model_convt = MultiInputImageTransformNet(
        in_channels_hsi=IN_CHANNELS_HSI,
        hsi_img_size=HSI_IMG_SIZE,
        patch_size_hsi=PATCH_SIZE_HSI,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        aux_input_channels=AUX_INPUT_CHANNELS,
        final_upsampling_method='convt'
    )

    total_params_convt = sum(p.numel() for p in model_convt.parameters())
    print(f"模型总参数量 (ConvTranspose2d): {total_params_convt / 1e6:.2f} M")

    dummy_main_input_convt = torch.randn(BATCH_SIZE, IN_CHANNELS_HSI, HSI_IMG_SIZE, HSI_IMG_SIZE)
    dummy_aux_input_convt = torch.randn(BATCH_SIZE, AUX_INPUT_CHANNELS, 64, 64)
    print(f"\n模拟 HSI 主输入尺寸: {dummy_main_input_convt.shape}")
    print(f"模拟辅助输入尺寸: {dummy_aux_input_convt.shape}")

    model_convt.eval()
    with torch.no_grad():
        output_convt = model_convt(dummy_main_input_convt, dummy_aux_input_convt)

    print(f"最终输出尺寸 (ConvTranspose2d): {output_convt.shape}")
    expected_output_shape_final = (BATCH_SIZE, 64, 128, 128)
    assert output_convt.shape == expected_output_shape_final, \
        f"ConvTranspose2d 输出尺寸不匹配预期！期望 {expected_output_shape_final}, 得到 {output_convt.shape}"
    print("ConvTranspose2d 模型输出尺寸符合预期！✅")

    print("\n--- 模式二：使用 PixelShuffle 作为最终上采样 ---")
    model_pixelshuffle = MultiInputImageTransformNet(
        in_channels_hsi=IN_CHANNELS_HSI,
        hsi_img_size=HSI_IMG_SIZE,
        patch_size_hsi=PATCH_SIZE_HSI,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        aux_input_channels=AUX_INPUT_CHANNELS,
        final_upsampling_method='pixelshuffle'
    )

    total_params_pixelshuffle = sum(p.numel() for p in model_pixelshuffle.parameters())
    print(f"模型总参数量 (PixelShuffle): {total_params_pixelshuffle / 1e6:.2f} M")

    dummy_main_input_pixelshuffle = torch.randn(BATCH_SIZE, IN_CHANNELS_HSI, HSI_IMG_SIZE, HSI_IMG_SIZE)
    dummy_aux_input_pixelshuffle = torch.randn(BATCH_SIZE, AUX_INPUT_CHANNELS, 64, 64)
    print(f"\n模拟 HSI 主输入尺寸: {dummy_main_input_pixelshuffle.shape}")
    print(f"模拟辅助输入尺寸: {dummy_aux_input_pixelshuffle.shape}")

    model_pixelshuffle.eval()
    with torch.no_grad():
        output_pixelshuffle = model_pixelshuffle(dummy_main_input_pixelshuffle, dummy_aux_input_pixelshuffle)

    print(f"最终输出尺寸 (PixelShuffle): {output_pixelshuffle.shape}")
    assert output_pixelshuffle.shape == expected_output_shape_final, \
        f"PixelShuffle 输出尺寸不匹配预期！期望 {expected_output_shape_final}, 得到 {output_pixelshuffle.shape}"
    print("PixelShuffle 模型输出尺寸符合预期！✅")


# --- 第二段代码 (ChannelExpansionNet) 的演示部分 ---
if __name__ == "__main__":
    # 演示 ChannelExpansionNet
    print("\n--- 演示 ChannelExpansionNet (保持不变) ---")
    IN_CHANNELS = 3
    OUT_CHANNELS = 31
    HEIGHT = 128
    WIDTH = 128
    BATCH_SIZE = 1

    print("\n--- 演示模式一：使用 Conv2d (1x1 卷积) 作为最终通道调整 ---")
    model_conv = ChannelExpansionNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, use_transposed_as_final=False)
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    print(f"模拟输入尺寸: {dummy_input.shape}")

    model_conv.eval()
    with torch.no_grad():
        output_conv = model_conv(dummy_input)
    print(f"使用 Conv2d 输出尺寸: {output_conv.shape}")
    expected_output_shape_channel_expansion = (BATCH_SIZE, OUT_CHANNELS, HEIGHT, WIDTH)
    assert output_conv.shape == expected_output_shape_channel_expansion, "Conv2d 输出尺寸不匹配预期！"
    print("Conv2d 模型输出尺寸符合预期！\n")

    print("--- 演示模式二：使用 ConvTranspose2d (1x1 转置卷积) 作为最终通道调整 ---")
    model_transposed = ChannelExpansionNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                                         use_transposed_as_final=True)
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    print(f"模拟输入尺寸: {dummy_input.shape}")

    model_transposed.eval()
    with torch.no_grad():
        output_transposed = model_transposed(dummy_input)
    print(f"使用 ConvTranspose2d 输出尺寸: {output_transposed.shape}")
    assert output_transposed.shape == expected_output_shape_channel_expansion, "ConvTranspose2d 输出尺寸不匹配预期！"
    print("ConvTranspose2d 模型输出尺寸符合预期！\n")