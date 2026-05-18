import torch
import torch.nn as nn


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path, extra=None):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    payload = {
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Load checkpoint into model (and optionally optimizer / scheduler).

    Handles three checkpoint formats:
      1. Full checkpoint dict with 'model' key  (saved by save_checkpoint)
      2. Legacy format with 'policy' key
      3. Raw state-dict (weights only)

    Returns:
        start_epoch (int), best_val_loss (float)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if optimizer is not None and "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch   = ckpt.get("epoch",    0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"[INFO] Resumed full checkpoint from epoch {ckpt.get('epoch', '?')} "
                  f"(val_loss={best_val_loss:.4f})")
        elif "policy" in ckpt:
            model.load_state_dict(ckpt["policy"])
            start_epoch   = 0
            best_val_loss = float("inf")
            print("[INFO] Loaded legacy 'policy' checkpoint (weights only).")
        else:
            model.load_state_dict(ckpt)
            start_epoch   = 0
            best_val_loss = float("inf")
            print("[INFO] Loaded raw state-dict checkpoint.")
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    return start_epoch, best_val_loss


def freeze_backbone(model: nn.Module):
    """
    Freeze every parameter whose name starts with 'backbone', 'cnn', 'encoder', or 'feature'.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(("backbone", "cnn", "encoder", "feature")):
            param.requires_grad = False
            frozen += 1
    if frozen:
        print(f"[INFO] Frozen {frozen} backbone parameter tensors.")
    else:
        print("[WARNING] --freeze_backbone was set but no parameters matched "
              "prefixes ('backbone', 'cnn', 'encoder', 'feature'). "
              "All parameters will be trained.")


def print_trainable_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")