import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os

# 假设 rcan_model.py 中包含了所有必要的模型定义 (RCAN, MeanShift, ResidualGroup, RCAB, CALayer)
# 如果这些辅助类在 rcan_model.py 之外，您需要根据实际文件路径进行调整导入
from .rcan_model import RCAN, MeanShift, ResidualGroup, RCAB, CALayer  # 导入 RCAN 模型及所有必要的子模块


def load_image(image_path):
    """加载图像并转换为 PyTorch Tensor。"""
    img = Image.open(image_path).convert('RGB')
    transform = ToTensor()
    return transform(img).unsqueeze(0)  # 添加批次维度


def save_image(tensor, output_path):
    """将 PyTorch Tensor 转换为图像并保存。"""
    transform = ToPILImage()
    img = transform(tensor.squeeze(0).cpu().clamp(0, 1))  # 移除批次维度，clamp到[0,1]
    img.save(output_path)


def main():
    # --- 配置参数 ---
    scale = 2  # 根据您下载的权重文件选择放大倍数 (例如：RCAN_BIX2.pt 对应 scale=2)
    model_weights_path = r"/home/shiyanshi/dbq/models_ECCV2018RCAN/RCAN_BIX2.pt"
    input_image_path = 'input_images/low_res_image.png'
    output_image_dir = 'output_images'
    output_image_name = f'high_res_x{scale}.png'

    os.makedirs(output_image_dir, exist_ok=True)
    output_image_path = os.path.join(output_image_dir, output_image_name)

    # --- 1. 检查设备 (GPU/CPU) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. 实例化模型 ---
    # RCAN 模型初始化时需要一些参数，通常在原始代码的 option.py 或直接在模型定义中提供默认值
    # 请根据 rcan_model.py 中 RCAN 类的 __init__ 方法来确定正确的参数
    # 示例参数（这些是RCAN的常见参数，可能需要根据您复制的实际代码调整）：
    args_model = {
        'n_resgroups': 10,  # G: Number of Residual Groups
        'n_resblocks': 20,  # B: Number of Residual Blocks in each Residual Group
        'n_feats': 64,  # N: Number of feature maps
        'rgb_range': 255,  # RGB range of input and output
        'n_colors': 3,  # Number of color channels (3 for RGB)
        'res_scale': 1,  # Residual scaling
        ''
        'reduction': 16,  # C: Channel reduction for CALayer
        'scale': scale,  # X: Super-resolution scale (2, 3, 4, 8)
        'no_upsampling': False,  # Whether to use pixel shuffle for upsampling (False for standard RCAN)
        'act': nn.ReLU(True),  # Activation function
    }

    print("Initializing RCAN model...")
    # 注意：这里的 RCAN 实例化参数必须与您下载的权重文件训练时的模型参数匹配
    model = RCAN(args_model).to(device)
    print("Model instantiated.")
    # print(model) # 可以打印模型结构，检查是否正确

    # --- 3. 加载预训练权重 ---
    print(f"Loading weights from {model_weights_path}...")
    # 加载状态字典
    state_dict = torch.load(model_weights_path, map_location=device)

    # 某些模型可能在state_dict的键前面有'model.'前缀，需要移除
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[len('model.'):]] = v
        else:
            new_state_dict[k] = v

    # 尝试加载
    try:
        model.load_state_dict(new_state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch in model architecture or unexpected keys.")
        print("Attempting to load with strict=False (may lead to incomplete model).")
        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded with strict=False.")

    # --- 4. 设置模型为评估模式 ---
    model.eval()  # 关闭 Dropout 和 BatchNorm，确保推理结果稳定
    print("Model set to evaluation mode.")

    # --- 5. 加载输入图像 ---
    print(f"Loading input image from {input_image_path}...")
    lr_image_tensor = load_image(input_image_path).to(device)
    print(f"Input image shape: {lr_image_tensor.shape}")

    # --- 6. 执行超分辨率推理 ---
    print("Performing super-resolution...")
    with torch.no_grad():  # 在推理阶段禁用梯度计算，节省内存和计算
        sr_image_tensor = model(lr_image_tensor)
    print("Super-resolution completed.")

    # --- 7. 保存结果图像 ---
    save_image(sr_image_tensor, output_image_path)
    print(f"Super-resolved image saved to {output_image_path}")


if __name__ == "__main__":
    main()