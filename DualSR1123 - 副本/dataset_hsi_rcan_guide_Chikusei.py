import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image


# =========================
# Chikusei SPF/SRF loader
# =========================
def load_chikusei_F(spf_mat_path: str):
    """
    Load Chikusei SPF/SRF .mat and return F: (3, B) float32, normalized row-wise.
    Your file example:
      key 'R' shape (128,4) -> [wavelength, R, G, B]
    We convert it to (3,128).
    """
    d = loadmat(spf_mat_path)
    keys = [k for k in d.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise KeyError(f"No valid keys in {spf_mat_path}")

    # Try common names first
    prefer = ["R", "F", "spf", "srf", "resp", "response"]
    key = None
    for k in prefer:
        if k in d:
            key = k
            break
    if key is None:
        # fallback: first non-private key
        key = keys[0]

    R = np.asarray(d[key], dtype=np.float32)
    if R.ndim != 2:
        raise ValueError(f"SPF/SRF must be 2D. got {R.shape} from key={key}")

    # handle (B,4) -> keep columns 1:4 = RGB
    if R.shape[1] == 4 and R.shape[0] > 10:
        R = R[:, 1:4]  # (B,3)

    # to (3,B)
    if R.shape[0] != 3 and R.shape[1] == 3:
        R = R.T
    if R.shape[0] != 3:
        raise ValueError(f"SPF must be (3,B). got {R.shape} from key={key}")

    # normalize each RGB row sum=1
    R = R / (R.sum(axis=1, keepdims=True) + 1e-8)
    return R.astype(np.float32), key


# =========================
# Utils (same style as yours)
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
    """LR_HSI(0~1) -> LR_RGB(0~255 float32) by SPF/SRF"""
    rgb = hsi_lr_hwc_01 @ F.T
    rgb = np.clip(rgb, 0.0, None)

    # normalize per-image to [0,1] (important for Chikusei SRF)
    rgb = rgb / (rgb.max() + 1e-8)
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


def _load_hsi_as_hwc01(mat_path: str, key: str, scale: int, expect_bands=None):
    """
    load mat -> HWC float32 in 0~1, and modcrop to divisible by scale
    support both:
      - HWC: (H,W,B)
      - CHW: (B,H,W)
    """
    d = loadmat(mat_path)
    if key not in d:
        keys = [k for k in d.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {mat_path}. keys={keys}")

    hsi = np.asarray(d[key], dtype=np.float32)

    # to HWC
    if hsi.ndim == 3:
        if (expect_bands is not None) and (hsi.shape[0] == expect_bands) and (hsi.shape[-1] != expect_bands):
            hsi = np.transpose(hsi, (1, 2, 0))  # (H,W,B)
    else:
        raise ValueError(f"Expected 3D cube. got {hsi.shape} in {mat_path}")

    # normalize fallback
    mx = float(hsi.max())
    if mx > 1.5:
        hsi = hsi / (mx + 1e-8)
    hsi = np.clip(hsi, 0.0, 1.0)

    # modcrop
    hsi = modcrop_hwc(hsi, scale)
    return hsi


# =========================
# Dataset (Chikusei style like your original)
# =========================
class HSIWithRCANGuideDataset(Dataset):
    """
    Chikusei dataset in the SAME style as your original code.

    Train:
      - slide-window patches on HR HSI (gt): hr_patch, stride
      - build LR HSI by bicubic downsample /scale
      - build LR RGB (0~255) by SPF/SRF from LR HSI (for RCAN input)

    Val:
      - use full images (no crop)
      - same LR+RGB generation

    Return:
      hsi_hr_t: [B,H,W] (0~1)
      hsi_lr_t: [B,h,w] (0~1)
      lr_rgb_t: [3,h,w] (0~255)
      name: str
    """

    def __init__(
        self,
        mode="train",
        scale=4,
        hr_patch=128,
        stride=64,
        chikusei_root="/home/shiyanshi/dbq/chikusei1",  # root/train root/val
        spf_mat_path="/home/shiyanshi/dbq/chikusei_128_4.mat",
        gt_key="gt",
        bands=128,
    ):
        assert mode in ["train", "val"], "mode must be 'train' or 'val'"
        self.mode = mode
        self.scale = int(scale)
        self.bands = int(bands)

        # ✅ load Chikusei SPF/SRF
        self.F, self.spf_key = load_chikusei_F(spf_mat_path)

        # ✅ choose split dir
        split_dir = os.path.join(chikusei_root, "train" if mode == "train" else "val")
        self.items = sorted(glob.glob(os.path.join(split_dir, "*.mat")))
        if not self.items:
            raise RuntimeError(f"No mats found in {split_dir}")

        # ---- train patch config (ONLY for train) ----
        self.hr_patch = None
        self.stride = None
        self.lr_patch = None
        self.lr_stride = None
        self.patch_index = None

        if self.mode == "train":
            hr_patch = int(hr_patch)
            stride = int(stride)
            assert hr_patch % self.scale == 0, "hr_patch must be divisible by scale"
            assert stride % self.scale == 0, "stride must be divisible by scale"

            self.hr_patch = hr_patch
            self.stride = stride
            self.lr_patch = hr_patch // self.scale
            self.lr_stride = stride // self.scale

            # prebuild patch index like your original
            self.patch_index = []
            for mat_path in self.items:
                hsi_hr_full = _load_hsi_as_hwc01(mat_path, gt_key, self.scale, expect_bands=self.bands)
                H, W, _ = hsi_hr_full.shape

                for y in range(0, H - self.hr_patch + 1, self.stride):
                    for x in range(0, W - self.hr_patch + 1, self.stride):
                        self.patch_index.append((mat_path, y, x))

            print(
                f"[HSIWithRCANGuideDataset|Chikusei] mode=train imgs={len(self.items)} "
                f"patches={len(self.patch_index)} hr_patch={self.hr_patch} stride={self.stride} "
                f"SPF_key={self.spf_key} F_shape={self.F.shape}"
            )
        else:
            print(
                f"[HSIWithRCANGuideDataset|Chikusei] mode=val imgs={len(self.items)} full-image eval "
                f"SPF_key={self.spf_key} F_shape={self.F.shape}"
            )

        self.gt_key = gt_key

    def __len__(self):
        return len(self.patch_index) if self.mode == "train" else len(self.items)

    def __getitem__(self, idx):
        if self.mode == "train":
            mat_path, y, x = self.patch_index[idx]
        else:
            mat_path = self.items[idx]
            y = x = None

        # HR full
        hsi_hr_full = _load_hsi_as_hwc01(mat_path, self.gt_key, self.scale, expect_bands=self.bands)

        # build LR full + LR_RGB full
        hsi_lr_full = bicubic_downsample_hsi(hsi_hr_full, self.scale)
        lr_rgb_full = hsi_to_rgb_lr_255(hsi_lr_full, self.F)

        if self.mode == "train":
            hr = hsi_hr_full[y:y + self.hr_patch, x:x + self.hr_patch, :]
            ly, lx = y // self.scale, x // self.scale
            lr = hsi_lr_full[ly:ly + self.lr_patch, lx:lx + self.lr_patch, :]
            lr_rgb = lr_rgb_full[ly:ly + self.lr_patch, lx:lx + self.lr_patch, :]
        else:
            hr = hsi_hr_full
            lr = hsi_lr_full
            lr_rgb = lr_rgb_full

        # to tensor
        hsi_hr_t = hwc01_to_chw_tensor(hr)      # [B,H,W]
        hsi_lr_t = hwc01_to_chw_tensor(lr)      # [B,h,w]
        lr_rgb_t = hwc255_to_chw_tensor(lr_rgb) # [3,h,w]

        name = os.path.basename(mat_path)
        return hsi_hr_t, hsi_lr_t, lr_rgb_t, name
