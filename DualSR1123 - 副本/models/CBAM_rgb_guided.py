import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 3D卷积模块定义 ---
class Conv_3D_Block(nn.Module):
    def __init__(self, device=None):
        super(Conv_3D_Block, self).__init__()
        self.body = nn.Conv3d(1, 1, (3, 3, 3), 1, (1, 1, 1), bias=True)
        if device:
            self.to(device)

    def forward(self, x):
        x = self.body(x.unsqueeze(1))
        return x.squeeze(1)


class Res3DBlock(nn.Module):
    def __init__(self, n_feats, bias=True, act=nn.ReLU(), res_scale=1, device=None):
        super(Res3DBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(1, n_feats, (3, 1, 1), 1, (1, 0, 0), bias=bias),
            act,
            nn.Conv3d(n_feats, 1, (1, 3, 3), 1, (0, 1, 1), bias=bias)
        )
        self.res_scale = res_scale
        if device:
            self.to(device)

    def forward(self, x):
        x = self.body(x.unsqueeze(1)) + x.unsqueeze(1)
        return x.squeeze(1)


# --- 注意力模块定义 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, device=None):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        if device:
            self.to(device)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, device=None):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        if device:
            self.to(device)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class EnhancedCBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7, device=None):
        super(EnhancedCBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, device=device)
        self.spatial_attention = SpatialAttention(kernel_size, device=device)
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        if device:
            self.to(device)

    def forward(self, x):
        identity = x
        ca_weights = self.channel_attention(x)
        x_ca = x * ca_weights
        sa_weights = self.spatial_attention(x_ca)
        x_sa = x_ca * sa_weights
        x_out = torch.cat([x_sa, identity], dim=1)
        x_out = self.conv(x_out)
        return x_out


class EnhancedSEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, device=None):
        super(EnhancedSEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        if device:
            self.to(device)

    def forward(self, x):
        identity = x
        weights = self.se(x)
        x_se = x * weights
        x_out = torch.cat([x_se, identity], dim=1)
        x_out = self.conv(x_out)
        return x_out


# --- 特征提取模块 ---
class RGBFeatureExtractor64(nn.Module):
    def __init__(self, device=None):
        super(RGBFeatureExtractor64, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if device:
            self.to(device)

    def forward(self, x_rgb):
        x = self.relu(self.conv1(x_rgb))
        x = self.relu(self.conv2(x))
        return x


class RGBFeatureExtractor128(nn.Module):
    def __init__(self, device=None):
        super(RGBFeatureExtractor128, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if device:
            self.to(device)

    def forward(self, x_rgb):
        x = self.relu(self.conv1(x_rgb))
        x = self.relu(self.conv2(x))
        return x


class LrHsiFeatureExtractor64(nn.Module):
    def __init__(self, device=None):
        super(LrHsiFeatureExtractor64, self).__init__()
        self.conv1 = nn.Conv2d(31, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if device:
            self.to(device)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class LrHsiFeatureFusion128(nn.Module):
    def __init__(self, device=None):
        super(LrHsiFeatureFusion128, self).__init__()
        self.conv1 = nn.Conv2d(31, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        if device:
            self.to(device)

    def forward(self, lrhsi_input, lrhsi_feature_64):
        x = self.relu(self.conv1(lrhsi_input))
        x = self.relu(self.conv2(x))
        return x


# --- 跨注意力模块 ---
class SimpleCrossAttention64(nn.Module):
    def __init__(self, query_channels=64, context_channels=64, output_channels=31, device=None):
        super(SimpleCrossAttention64, self).__init__()
        self.query_conv = nn.Conv2d(query_channels, 64, 1)
        self.context_conv = nn.Conv2d(context_channels, 64, 1)
        self.cbam = EnhancedCBAM(64, reduction_ratio=16, device=device)
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, 3, padding=1)
        )
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        if device:
            self.to(device)

    def forward(self, query_feature, context_feature):
        if query_feature.shape[-2:] != context_feature.shape[-2:]:
            context_feature = F.interpolate(context_feature, size=query_feature.shape[-2:],
                                            mode='bilinear', align_corners=False)
        query = self.query_conv(query_feature)
        context = self.context_conv(context_feature)
        fused = query + context
        attended = self.cbam(fused)
        output = self.output_conv(attended)
        output = self.upsample(output)
        return output


class SimpleCrossAttention128(nn.Module):
    def __init__(self, query_channels=64, context_channels=64, output_channels=31, device=None):
        super(SimpleCrossAttention128, self).__init__()
        self.query_conv = nn.Conv2d(query_channels, 64, 1)
        self.context_conv = nn.Conv2d(context_channels, 64, 1)
        self.se_block = EnhancedSEBlock(64, reduction_ratio=16, device=device)
        self.conv3x3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5x5 = nn.Conv2d(64, 64, 5, padding=2)
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, 3, padding=1)
        )
        if device:
            self.to(device)

    def forward(self, query_image, context_image):
        if query_image.shape[-2:] != context_image.shape[-2:]:
            query_image = F.interpolate(query_image, size=context_image.shape[-2:],
                                        mode='bilinear', align_corners=False)
        query = self.query_conv(query_image)
        context = self.context_conv(context_image)
        fused = query + context
        attended = self.se_block(fused)
        feat_3x3 = self.conv3x3(attended)
        feat_5x5 = self.conv5x5(attended)
        multi_scale = feat_3x3 + feat_5x5
        output = self.output_conv(multi_scale)
        return output


class AdaptiveFineStageNet(nn.Module):
    def __init__(self, low_res_channels=31, high_res_channels=31, texture_channels=1, output_channels=31, device=None):
        super(AdaptiveFineStageNet, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(low_res_channels + high_res_channels + texture_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, output_channels, 3, padding=1)
        if device:
            self.to(device)

    def forward(self, low_res_feat, high_res_feat, texture_map):
        target_size = texture_map.shape[-2:]
        if low_res_feat.shape[-2:] != target_size:
            low_res_feat = F.interpolate(low_res_feat, size=target_size, mode='bilinear', align_corners=False)
        if high_res_feat.shape[-2:] != target_size:
            high_res_feat = F.interpolate(high_res_feat, size=target_size, mode='bilinear', align_corners=False)
        fused = torch.cat([low_res_feat, high_res_feat, texture_map], dim=1)
        x = self.fusion_conv(fused)
        output = self.final_conv(x)
        return output


class MSIToHSIResidual(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, device=None):
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
        if device:
            self.to(device)

    def forward(self, msi):
        return self.conv_layers(msi)


class Final3DRefinement(nn.Module):
    def __init__(self, n_feats=32, device=None):
        super(Final3DRefinement, self).__init__()
        self.pseudo_3d_conv1 = Res3DBlock(n_feats=n_feats, act=nn.ReLU(), device=device)
        self.pseudo_3d_conv2 = Res3DBlock(n_feats=n_feats, act=nn.ReLU(), device=device)
        self.full_3d_conv = Conv_3D_Block(device=device)
        if device:
            self.to(device)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 3, 1)
        x_refined = self.pseudo_3d_conv1(x_permuted)
        x_refined = self.pseudo_3d_conv2(x_refined)
        x_refined = self.full_3d_conv(x_refined)
        x_refined = x_refined.permute(0, 3, 1, 2)
        return x_refined


class FullSuperResolutionNet_CBAM(nn.Module):
    def __init__(self, device=None):
        super(FullSuperResolutionNet_CBAM, self).__init__()
        self.device = device

        # 初始化所有子模块时传递device参数
        self.rgb_extractor_64 = RGBFeatureExtractor64(device=device)
        self.rgb_extractor_128 = RGBFeatureExtractor128(device=device)

        self.lrhsi_feature_extractor_64 = LrHsiFeatureExtractor64(device=device)
        self.lrhsi_feature_fusion_128 = LrHsiFeatureFusion128(device=device)

        self.transform_64 = SimpleCrossAttention64(
            query_channels=64, context_channels=64, output_channels=31, device=device
        )
        self.transform_128 = SimpleCrossAttention128(
            query_channels=64, context_channels=64, output_channels=31, device=device
        )

        self.fine_stage_net = AdaptiveFineStageNet(
            low_res_channels=31, high_res_channels=31, texture_channels=1, output_channels=31, device=device
        )

        self.msi_to_hsi = MSIToHSIResidual(in_channels=3, out_channels=31, device=device)

        self.final_3d_refinement = Final3DRefinement(n_feats=32, device=device)

        # Sobel算子 - 在forward中动态移动到设备
        self.sobel_x_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                                           dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                                           dtype=torch.float32).view(1, 1, 3, 3)

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
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _generate_texture_map(self, hr_rgb):
        band_for_texture = hr_rgb[:, 0:1]
        # 动态将Sobel算子移动到输入数据所在的设备
        sobel_x = self.sobel_x_kernel.to(band_for_texture.device)
        sobel_y = self.sobel_y_kernel.to(band_for_texture.device)
        grad_x = F.conv2d(band_for_texture, sobel_x, padding=1)
        grad_y = F.conv2d(band_for_texture, sobel_y, padding=1)
        texture_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return texture_map

    def forward(self, lrhsi_input, msi_input):
        target_size = msi_input.shape[-1]

        # 1. 直接使用输入的MSI作为HR RGB
        hr_rgb = msi_input

        # 2. 生成纹理图
        texture_map = self._generate_texture_map(hr_rgb)

        # 3. RGB特征提取
        rgb_feature_128 = self.rgb_extractor_128(hr_rgb)

        # 4. 复用到64路
        intermediate_size = target_size // 2
        hr_rgb_64 = F.interpolate(hr_rgb, size=(intermediate_size, intermediate_size),
                                  mode='bilinear', align_corners=False)
        rgb_feature_64 = F.interpolate(rgb_feature_128, size=(intermediate_size, intermediate_size),
                                       mode='bilinear', align_corners=False)

        # 5. LR HSI特征提取
        lrhsi_feature_64 = self.lrhsi_feature_extractor_64(lrhsi_input)
        lrhsi_feature_128 = self.lrhsi_feature_fusion_128(lrhsi_input, lrhsi_feature_64)
        lrhsi_feature_128 = F.interpolate(lrhsi_feature_128, size=(target_size, target_size),
                                          mode='bilinear', align_corners=False)

        # 6. 跨注意力变换
        hr_feature_64 = self.transform_64(lrhsi_feature_64, rgb_feature_64)
        hr_feature_128 = self.transform_128(lrhsi_feature_128, rgb_feature_128)

        # 7. 最终融合
        final_output = self.fine_stage_net(hr_feature_64, hr_feature_128, texture_map)

        # 8. MSI残差连接
        msi_residual = self.msi_to_hsi(msi_input)
        final_output = final_output + msi_residual

        # 9. 3D精炼
        final_output = self.final_3d_refinement(final_output)

        # 10. 限制输出范围
        final_output = final_output.clamp(0.0, 1.0)

        return final_output