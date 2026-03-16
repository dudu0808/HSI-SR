#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PSNR_SAM1 import quality_assessment
from models.Frequency_with_ghostnet_parallel import MultiScaleSuperResolutionNet
from models.rcan_model import RCAN

# ✅ 你现在使用的是 Chikusei 版 dataset（内部有 self.F / self.spf_key）
from dataset_hsi_rcan_guide_Chikusei import HSIWithRCANGuideDataset


# =========================
# Utils
# =========================
def chw01_to_hwc01(x: torch.Tensor) -> np.ndarray:
    """torch BCHW or CHW -> numpy HWC in [0,1]"""
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0)).astype(np.float32)


def cal_sam(Itrue, Ifake):
    """
    Returns normalized SAM loss: mean(angle_rad)/pi (0~1)
    For degrees: sam_deg = sam_loss * 180
    """
    esp = 1e-6
    InnerPro = torch.sum(Itrue * Ifake, 1, keepdim=True)
    len1 = torch.norm(Itrue, p=2, dim=1, keepdim=True)
    len2 = torch.norm(Ifake, p=2, dim=1, keepdim=True)
    divisor = len1 * len2
    mask = torch.eq(divisor, 0)
    divisor = divisor + mask.float() * esp
    cosA = torch.sum(InnerPro / divisor, 1).clamp(-1 + esp, 1 - esp)
    sam = torch.acos(cosA)          # rad
    sam = torch.mean(sam) / np.pi   # rad/pi
    return sam


# =========================
# RCAN
# =========================
@torch.no_grad()
def build_rcan(device, ckpt_path: str, scale: int):
    rcan_args = dict(
        n_resgroups=10,
        n_resblocks=20,
        n_feats=64,
        reduction=16,
        scale=scale,
        rgb_range=255.0,
        n_colors=3,
        res_scale=1.0,
    )
    rcan = RCAN(rcan_args).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    rcan.load_state_dict(ckpt.get("model", ckpt), strict=True)
    rcan.eval()
    for p in rcan.parameters():
        p.requires_grad_(False)
    print(f"[RCAN] loaded & frozen: {ckpt_path}")
    return rcan


@torch.no_grad()
def make_guide_rgb(rcan, lr_rgb_255):
    guide_255 = rcan(lr_rgb_255)               # 0~255
    return (guide_255 / 255.0).clamp(0, 1)     # 0~1


# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, rcan, loader, device, scale, print_baseline: bool = False):
    """
    print_baseline:
      - True  -> print [VAL|BICUBIC] + [VAL] GT min/max once
      - False -> never print them
    """
    model.eval()
    psnr_list, sam_list, ssim_list = [], [], []
    printed_baseline = False

    for hsi_hr, hsi_lr, lr_rgb, name in loader:
        hsi_hr = hsi_hr.to(device).float()
        hsi_lr = hsi_lr.to(device).float()
        lr_rgb = lr_rgb.to(device).float()

        guide = make_guide_rgb(rcan, lr_rgb)

        # model pred
        pred_x4 = model(hsi_lr, guide).clamp(0, 1)

        # bicubic baseline: LR -> HR
        bicubic = F.interpolate(hsi_lr, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0, 1)

        pred_np = chw01_to_hwc01(pred_x4)
        bic_np  = chw01_to_hwc01(bicubic)
        gt_np   = chw01_to_hwc01(hsi_hr)

        # ✅ baseline / minmax 只在 (print_baseline=True) 时打印一次
        if print_baseline and (not printed_baseline):
            m_bi = quality_assessment(
                x_true=gt_np,
                x_pred=bic_np,
                data_range=1.0,
                ratio=scale,
                multi_dimension=False,
            )
            print(f"[VAL|BICUBIC] PSNR={m_bi['MPSNR']:.4f} SAM={m_bi['SAM']:.4f} SSIM={m_bi['MSSIM']:.4f}")
            print("[VAL] GT min/max:", float(hsi_hr.min()), float(hsi_hr.max()))
            printed_baseline = True

        metrics = quality_assessment(
            x_true=gt_np,
            x_pred=pred_np,
            data_range=1.0,
            ratio=scale,
            multi_dimension=False,
        )
        psnr_list.append(metrics["MPSNR"])
        sam_list.append(metrics["SAM"])
        ssim_list.append(metrics["MSSIM"])

    return float(np.mean(psnr_list)), float(np.mean(sam_list)), float(np.mean(ssim_list))


# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, rcan, loader, optimizer, device, F_t, args, epoch):
    model.train()
    losses = []

    for it, (hsi_hr, hsi_lr, lr_rgb, _) in enumerate(loader, start=1):
        hsi_hr = hsi_hr.to(device).float()
        hsi_lr = hsi_lr.to(device).float()
        lr_rgb = lr_rgb.to(device).float()

        with torch.no_grad():
            guide = make_guide_rgb(rcan, lr_rgb)

        # parallel forward
        pred_x4, pred_x2 = model(hsi_lr, guide, return_x2=True)
        pred_x4_01 = pred_x4.clamp(0, 1)
        pred_x2_01 = pred_x2.clamp(0, 1)

        # x2 GT
        hsi_hr_x2 = F.interpolate(hsi_hr, scale_factor=0.5, mode="bicubic", align_corners=False).clamp(0, 1)

        # ---- x4 loss ----
        loss_mse_x4 = F.mse_loss(pred_x4_01, hsi_hr)
        loss_sam_x4 = cal_sam(pred_x4_01, hsi_hr)

        rgb_pred_x4 = torch.einsum("oc,bchw->bohw", F_t, pred_x4_01)
        rgb_gt_x4   = torch.einsum("oc,bchw->bohw", F_t, hsi_hr)
        loss_rgb_x4 = F.l1_loss(rgb_pred_x4, rgb_gt_x4)

        loss_x4 = args.w_mse * loss_mse_x4 + args.w_sam * loss_sam_x4 + args.w_rgb * loss_rgb_x4

        # ---- x2 loss ----
        loss_mse_x2 = F.mse_loss(pred_x2_01, hsi_hr_x2)
        loss_sam_x2 = cal_sam(pred_x2_01, hsi_hr_x2)

        rgb_pred_x2 = torch.einsum("oc,bchw->bohw", F_t, pred_x2_01)
        rgb_gt_x2   = torch.einsum("oc,bchw->bohw", F_t, hsi_hr_x2)
        loss_rgb_x2 = F.l1_loss(rgb_pred_x2, rgb_gt_x2)

        loss_x2 = args.w_mse2 * loss_mse_x2 + args.w_sam2 * loss_sam_x2 + args.w_rgb2 * loss_rgb_x2

        # total
        loss = loss_x4 + args.w_x2 * loss_x2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        losses.append(loss.item())

        if it % args.log_every == 0:
            sam_x4_deg = loss_sam_x4.item() * 180.0
            sam_x2_deg = loss_sam_x2.item() * 180.0
            print(
                f"[Train] Epoch[{epoch}/{args.epochs}] Iter[{it}/{len(loader)}] "
                f"Loss={np.mean(losses):.6f} | "
                f"x4(MSE={loss_mse_x4.item():.6f}, SAM={sam_x4_deg:.6f}, RGB={loss_rgb_x4.item():.6f}) | "
                f"x2(MSE={loss_mse_x2.item():.6f}, SAM={sam_x2_deg:.6f}, RGB={loss_rgb_x2.item():.6f})"
            )

    return float(np.mean(losses))


# =========================
# Checkpoint IO
# =========================
def save_ckpt(path, epoch, best_psnr, model, optimizer, args, best_epoch=None):
    ckpt = {
        "epoch": epoch,
        "best_psnr": best_psnr,
        "best_epoch": best_epoch if best_epoch is not None else epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


def try_resume(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_psnr = float(ckpt.get("best_psnr", -1e9))
    best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
    print(f"[RESUME] loaded: {path}")
    print(f"[RESUME] start_epoch={start_epoch}, best_psnr={best_psnr:.4f}, best_epoch={best_epoch}")
    return start_epoch, best_psnr, best_epoch


# =========================
# Main
# =========================
def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # RCAN frozen
    rcan = build_rcan(device, args.rcan_ckpt, args.scale)

    # dataset
    train_set = HSIWithRCANGuideDataset(
        mode="train",
        scale=args.scale,
        hr_patch=args.patch_size,
        stride=args.stride,
        chikusei_root=args.chikusei_root,
        spf_mat_path=args.chikusei_spf_mat,
        gt_key=args.chikusei_gt_key,
        bands=args.seq_len,
    )
    val_set = HSIWithRCANGuideDataset(
        mode="val",
        scale=args.scale,
        hr_patch=args.patch_size,
        stride=args.stride,
        chikusei_root=args.chikusei_root,
        spf_mat_path=args.chikusei_spf_mat,
        gt_key=args.chikusei_gt_key,
        bands=args.seq_len,
    )

    # ✅ Build F_t from dataset (no extra import!)
    if not hasattr(train_set, "F"):
        raise AttributeError("Your dataset must have attribute F (3,B).")
    F_np = train_set.F.astype(np.float32)
    spf_key = getattr(train_set, "spf_key", "unknown")

    if F_np.ndim != 2 or F_np.shape[0] != 3:
        raise ValueError(f"[F] Bad F shape: {F_np.shape}, expected (3,C)")
    if F_np.shape[1] != args.seq_len:
        raise ValueError(f"[F] Channel mismatch: F is (3,{F_np.shape[1]}) but seq_len={args.seq_len}")

    F_t = torch.from_numpy(F_np).to(device).float()
    print(f"[F] loaded from dataset: SPF_key={spf_key}, shape={tuple(F_t.shape)}")

    # model
    model = MultiScaleSuperResolutionNet(
        in_channels=args.seq_len,
        guide_channels=3,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        pin_memory=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # resume
    start_epoch = 1
    best_psnr = -1e9
    best_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_psnr, best_epoch = try_resume(args.resume, model, optimizer, device)
    elif args.resume:
        print(f"[RESUME] WARNING: resume path not found: {args.resume} (start from scratch)")

    best_path = os.path.join(args.save_dir, "best.pth")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, rcan, train_loader, optimizer, device, F_t, args, epoch)

        # ✅ baseline 只在 “本次训练的第一轮” 打印一次（resume 也一样）
        print_baseline = (epoch == start_epoch)
        psnr, sam, ssim = validate(model, rcan, val_loader, device, args.scale, print_baseline=print_baseline)

        print(
            f"Epoch[{epoch}] TrainLoss={train_loss:.6f} | "
            f"VAL PSNR={psnr:.4f} SAM={sam:.4f} SSIM={ssim:.4f} | "
            f"time={time.time()-t0:.2f}s"
        )

        # best
        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            save_ckpt(best_path, epoch, best_psnr, model, optimizer, args, best_epoch=best_epoch)
            print(f"[BEST] Epoch {epoch} | PSNR={best_psnr:.4f} saved -> {best_path}")

        # periodic save
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
            save_ckpt(ckpt_path, epoch, best_psnr, model, optimizer, args, best_epoch=best_epoch)
            print(f"[CKPT] saved -> {ckpt_path}")

    print(f"Training finished. Best PSNR={best_psnr:.4f} @ epoch {best_epoch}. Best model: {best_path}")


# =========================
# Args
# =========================
def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--gpus", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)

    # dataset
    p.add_argument("--dataset", type=str, default="chikusei", choices=["chikusei"])
    p.add_argument("--chikusei_root", type=str, default="/home/shiyanshi/dbq/chikusei1")
    p.add_argument("--chikusei_spf_mat", type=str, default="/home/shiyanshi/dbq/chikusei_128_4.mat")
    p.add_argument("--chikusei_gt_key", type=str, default="gt")

    # model
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--base_channels", type=int, default=64)

    # train
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=0.5)
    p.add_argument("--log_every", type=int, default=100)

    # patching
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)

    # RCAN ckpt
    p.add_argument("--rcan_ckpt", type=str, default="/home/shiyanshi/dbq/DualSR1123/rcan_fullmix_hsival_best.pth")

    # save
    p.add_argument("--save_dir", type=str, default="/home/shiyanshi/dbq/DualSR1123/checkpoints_parallel1_chikusei")
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)

    # loss weights
    p.add_argument("--w_mse", type=float, default=0.85)
    p.add_argument("--w_sam", type=float, default=0.13)
    p.add_argument("--w_rgb", type=float, default=0.02)

    p.add_argument("--w_x2", type=float, default=0.3)
    p.add_argument("--w_mse2", type=float, default=0.85)
    p.add_argument("--w_sam2", type=float, default=0.13)
    p.add_argument("--w_rgb2", type=float, default=0.02)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
