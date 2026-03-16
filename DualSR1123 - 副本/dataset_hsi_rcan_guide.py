import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image


# =========================
# SPF (same as your RCAN training)
# =========================
def create_F():
    F = np.array([
        [2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
        [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
        [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.float32)
    return F / F.sum(axis=1, keepdims=True)


# =========================
# Utils
# =========================
def modcrop_hwc(x, scale: int):
    h, w = x.shape[:2]
    return x[:h - (h % scale), :w - (w % scale)]


def bicubic_downsample_hsi(hsi_hwc_01: np.ndarray, scale: int):
    """HSI HxWxC float32 0~1 -> (H/scale)x(W/scale)xC float32 0~1"""
    h, w, c = hsi_hwc_01.shape
    out_h, out_w = h // scale, w // scale
    out = np.empty((out_h, out_w, c), dtype=np.float32)
    for i in range(c):
        band = (np.clip(hsi_hwc_01[..., i], 0, 1) * 255.0).astype(np.uint8)
        band_lr = Image.fromarray(band, mode="L").resize((out_w, out_h), resample=Image.BICUBIC)
        out[..., i] = np.asarray(band_lr, dtype=np.float32) / 255.0
    return out


def hsi_to_rgb_lr_255(hsi_lr_hwc_01: np.ndarray, F: np.ndarray):
    """LR_HSI(0~1) -> LR_RGB(0~255 float32) by SPF"""
    rgb = hsi_lr_hwc_01 @ F.T
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).astype(np.float32)


def hwc01_to_chw_tensor(x: np.ndarray):
    """HxWxC 0~1 -> torch [C,H,W] float32"""
    if x.ndim == 2:
        x = x[..., None]
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def hwc255_to_chw_tensor(x: np.ndarray):
    """HxWxC 0~255 -> torch [C,H,W] float32"""
    if x.ndim == 2:
        x = x[..., None]
    return torch.from_numpy(np.transpose(x.astype(np.float32), (2, 0, 1)))


def _load_hsi_as_hwc01(mat_path: str, key: str, scale: int):
    """load mat -> HWC float32 in 0~1, and modcrop to divisible by scale"""
    d = loadmat(mat_path)
    hsi = d[key]

    # to HWC
    if hsi.ndim == 3 and hsi.shape[0] == 31 and hsi.shape[-1] != 31:
        hsi = np.transpose(hsi, (1, 2, 0))
    hsi = hsi.astype(np.float32)

    # normalize fallback
    mx = float(hsi.max())
    if mx > 1.5:
        hsi = hsi / (mx + 1e-8)
    hsi = np.clip(hsi, 0.0, 1.0)

    # modcrop
    hsi = modcrop_hwc(hsi, scale)
    return hsi


# =========================
# Dataset
# =========================
class HSIWithRCANGuideDataset(Dataset):
    """
    单数据集训练/验证（不混训）

    Train:
      - slide-window patches on HR HSI: hr_patch=128, stride=64
      - build LR HSI by bicubic downsample /scale
      - build LR RGB (0~255) by SPF from LR HSI (for RCAN input)
      - return patch-aligned (HR_HSI, LR_HSI, LR_RGB, name)

    Val:
      - use full images (no crop)
      - return (HR_HSI, LR_HSI, LR_RGB, name)
    """

    def __init__(
        self,
        mode="train",
        scale=4,
        hr_patch=128,
        stride=128,
        cave_root="/home/shiyanshi/dbq/CAVE",
        icvl_root="/home/shiyanshi/dbq/ICVL",
        harvard_root="/home/shiyanshi/dbq/Harvard/CZ_hsdb",
        dataset="icvl",   # ✅ 新增：单选数据集（'cave' / 'icvl'/'harvard'，以后可扩展）
    ):
        assert mode in ["train", "val"], "mode must be 'train' or 'val'"
        self.mode = mode
        self.scale = int(scale)
        self.F = create_F()

        # ✅ 数据集注册表：以后你加新数据集就在这里加一项
        # 每项：train_dir, train_key, val_dir, val_key
        registry = {
            "cave": (
                os.path.join(cave_root, "train"), "Z",
                os.path.join(cave_root, "validate"), "Z",
            ),
            "icvl": (
                os.path.join(icvl_root, "train"), "gt",
                os.path.join(icvl_root, "validate"), "gt",
            ),
            "harvard": (
                os.path.join(harvard_root, "train"), "ref",
                os.path.join(harvard_root, "validate"), "ref",
            ),
        }

        ds = dataset.lower()
        if ds not in registry:
            raise ValueError(f"Unknown dataset='{dataset}'. Available: {list(registry.keys())}")

        train_dir, train_key, val_dir, val_key = registry[ds]
        if mode == "train":
            hsi_dirs = [(train_dir, train_key)]
        else:
            hsi_dirs = [(val_dir, val_key)]

        # collect items
        self.items = []
        for d, key in hsi_dirs:
            mats = sorted(glob.glob(os.path.join(d, "*.mat")))
            for m in mats:
                self.items.append((m, key))
        if not self.items:
            raise RuntimeError(f"No HSI mats found for dataset={dataset}, mode={mode} in {hsi_dirs}")

        # ---- train patch config (ONLY for train) ----
        self.hr_patch = None
        self.stride = None
        self.lr_patch = None
        self.lr_stride = None
        self.patch_index = None

        if self.mode == "train":
            assert hr_patch is not None, "train mode requires hr_patch"
            hr_patch = int(hr_patch)
            stride = int(stride)
            assert hr_patch % self.scale == 0, "hr_patch must be divisible by scale"
            assert stride % self.scale == 0, "stride must be divisible by scale (for clean LR alignment)"

            self.hr_patch = hr_patch
            self.stride = stride
            self.lr_patch = hr_patch // self.scale
            self.lr_stride = stride // self.scale

            # prebuild patch index for train
            self.patch_index = []
            for (mat_path, key) in self.items:
                hsi_hr_full = _load_hsi_as_hwc01(mat_path, key, self.scale)  # already modcrop
                H, W, _ = hsi_hr_full.shape

                for y in range(0, H - self.hr_patch + 1, self.stride):
                    for x in range(0, W - self.hr_patch + 1, self.stride):
                        self.patch_index.append((mat_path, key, y, x))

            print(
                f"[HSIWithRCANGuideDataset] dataset={ds} mode=train imgs={len(self.items)} "
                f"patches={len(self.patch_index)} hr_patch={self.hr_patch} stride={self.stride}"
            )
        else:
            print(f"[HSIWithRCANGuideDataset] dataset={ds} mode=val imgs={len(self.items)} (full image eval)")

    def __len__(self):
        return len(self.patch_index) if self.mode == "train" else len(self.items)

    def __getitem__(self, idx):
        if self.mode == "train":
            mat_path, key, y, x = self.patch_index[idx]
        else:
            mat_path, key = self.items[idx]
            y = x = None

        # HR full (HWC 0~1, modcrop)
        hsi_hr_full = _load_hsi_as_hwc01(mat_path, key, self.scale)

        # build LR full (HWC 0~1) + LR_RGB full (0~255)
        hsi_lr_full = bicubic_downsample_hsi(hsi_hr_full, self.scale)
        lr_rgb_full = hsi_to_rgb_lr_255(hsi_lr_full, self.F)

        if self.mode == "train":
            # crop HR patch
            hr = hsi_hr_full[y:y + self.hr_patch, x:x + self.hr_patch, :]

            # aligned LR crop
            ly, lx = y // self.scale, x // self.scale
            lr = hsi_lr_full[ly:ly + self.lr_patch, lx:lx + self.lr_patch, :]
            lr_rgb = lr_rgb_full[ly:ly + self.lr_patch, lx:lx + self.lr_patch, :]
        else:
            # full image
            hr = hsi_hr_full
            lr = hsi_lr_full
            lr_rgb = lr_rgb_full

        # to tensor
        hsi_hr_t = hwc01_to_chw_tensor(hr)        # [31,H,W] 0~1
        hsi_lr_t = hwc01_to_chw_tensor(lr)        # [31,h,w] 0~1
        lr_rgb_t = hwc255_to_chw_tensor(lr_rgb)   # [3,h,w]  0~255

        name = os.path.basename(mat_path)
        return hsi_hr_t, hsi_lr_t, lr_rgb_t, name