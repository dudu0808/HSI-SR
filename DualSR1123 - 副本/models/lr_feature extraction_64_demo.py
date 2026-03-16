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

class ViTFeatureDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_patches: int, out_channels: int,
                 output_height: int, output_width: int, patch_size: int):
        super(ViTFeatureDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.out_channels = out_channels
        self.output_height = output_height
        self.output_width = output_width
        self.patch_size = patch_size
        self.linear_reconstruct = nn.Linear(embed_dim, out_channels * (patch_size * patch_size))
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.linear_reconstruct.weight)
        if self.linear_reconstruct.bias is not None:
            nn.init.constant_(self.linear_reconstruct.bias, 0)
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1x1.bias is not None:
            nn.init.constant_(self.conv1x1.bias, 0)

    def forward(self, x):
        x = self.linear_reconstruct(x)
        h_patches = self.output_height // self.patch_size
        w_patches = self.output_width // self.patch_size
        x = x.permute(0, 2, 1).view(x.shape[0], self.out_channels * (self.patch_size ** 2), h_patches, w_patches)
        x = nn.PixelShuffle(self.patch_size)(x)
        x = self.relu(self.conv1x1(x))
        return x

# --- 主要模块 (Main Module) ---

class FeatureExtractionAndUpsampleNet(nn.Module):
    def __init__(self,
                 in_channels: int = 31,
                 img_size: int = 32,
                 output_channels: int = 64,
                 patch_size: int = 2,  # 调整为 2 以保留更多细节
                 embed_dim: int = 256,
                 num_transformer_layers: int = 4,
                 num_heads: int = 4,
                 vit_feature_map_channels: int = 64):
        super(FeatureExtractionAndUpsampleNet, self).__init__()
        self.input_img_height = img_size
        self.input_img_width = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.vit_feature_map_channels = vit_feature_map_channels
        self.final_output_channels = output_channels
        self.num_patches = (self.input_img_height // patch_size) * (self.input_img_width // patch_size)

        self.patch_embedding = PatchEmbedding(
            img_height=self.input_img_height,
            img_width=self.input_img_width,
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=self.embed_dim
        )
        self.pos_embedding = PositionalEncoding(d_model=self.embed_dim, max_len=self.num_patches)
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads)
            for _ in range(num_transformer_layers)
        ])
        self.vit_feature_decoder = ViTFeatureDecoder(
            embed_dim=self.embed_dim,
            num_patches=self.num_patches,
            out_channels=self.vit_feature_map_channels,
            output_height=self.input_img_height,
            output_width=self.input_img_width,
            patch_size=self.patch_size
        )
        self.upsample = nn.ConvTranspose2d(
            in_channels=self.vit_feature_map_channels,
            out_channels=self.final_output_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.upsample_relu = nn.ReLU(inplace=True)
        self.final_point_conv = nn.Conv2d(
            in_channels=self.final_output_channels,
            out_channels=self.final_output_channels,
            kernel_size=1
        )
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

    def forward(self, x):
        x = self.patch_embedding(x)  # (N, num_patches, embed_dim)
        x = self.pos_embedding(x)  # (N, num_patches, embed_dim)
        for layer in self.transformer_encoder_layers:
            x = layer(x)  # (N, num_patches, embed_dim)
        vit_output_feature_map = self.vit_feature_decoder(x)  # (N, 64, 32, 32)
        upsampled_feature = self.upsample(vit_output_feature_map)  # (N, 64, 64, 64)
        upsampled_feature = self.upsample_relu(upsampled_feature)
        final_output = self.final_point_conv(upsampled_feature)
        return final_output

# --- 演示部分 (Demo) ---
if __name__ == "__main__":
    print("--- 带有 ViT 特征提取和上采样的新模块演示 ---")

    IN_CHANNELS = 31
    IMG_SIZE = 32
    OUTPUT_CHANNELS = 64
    PATCH_SIZE = 2
    EMBED_DIM = 256
    NUM_TRANSFORMER_LAYERS = 2
    NUM_HEADS = 4
    VIT_FEATURE_MAP_CHANNELS = 64

    model = FeatureExtractionAndUpsampleNet(
        in_channels=IN_CHANNELS,
        img_size=IMG_SIZE,
        output_channels=OUTPUT_CHANNELS,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        num_heads=NUM_HEADS,
        vit_feature_map_channels=VIT_FEATURE_MAP_CHANNELS
    )

    model.eval()
    dummy_input = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)
    print(f"\n模拟输入尺寸: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"最终输出尺寸: {output.shape}")
    expected_output_shape = (1, OUTPUT_CHANNELS, IMG_SIZE * 2, IMG_SIZE * 2)
    assert output.shape == expected_output_shape, "输出尺寸不匹配预期！"
    print("模型输出尺寸符合预期！✅")