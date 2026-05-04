import torch
import torch.nn as nn


def _arch_from_model(model) -> dict:
    """Extract serialisable arch metadata from a model instance."""
    sd = model.state_dict()
    is_fast = model.__class__.__name__ == "DrivingPolicyNetFast"
    key = "conv1.weight" if is_fast else "cnn.0.0.weight"
    in_channels = int(sd[key].shape[1]) if key in sd else 4
    return {
        "class":       model.__class__.__name__,
        "use_ego":     bool(getattr(model, "use_ego", True)),
        "in_channels": in_channels,
    }


def detect_arch_from_ckpt(ckpt: dict) -> dict:
    """
    Return arch metadata from a checkpoint dict.

    Checks the saved 'arch' key first (new checkpoints), then falls back to
    inferring from state-dict key names and weight shapes (old checkpoints).
    """
    if "arch" in ckpt:
        return ckpt["arch"]

    # Fallback: infer from state dict
    sd = ckpt.get("model") or ckpt.get("policy") or ckpt
    if not isinstance(sd, dict):
        return {"class": "DrivingPolicyNet", "use_ego": True, "in_channels": 4}

    keys = set(sd.keys())
    is_fast = "conv1.weight" in keys
    has_ego = any(k.startswith("ego_fc.") for k in keys)

    if is_fast and "conv1.weight" in sd:
        in_channels = int(sd["conv1.weight"].shape[1])
    elif "cnn.0.0.weight" in sd:
        in_channels = int(sd["cnn.0.0.weight"].shape[1])
    else:
        in_channels = 4

    return {
        "class":       "DrivingPolicyNetFast" if is_fast else "DrivingPolicyNet",
        "use_ego":     has_ego,
        "in_channels": in_channels,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path, extra=None):
    """Save a full training checkpoint (model + optimizer + scheduler state)."""
    payload = {
        "epoch":     epoch,
        "val_loss":  val_loss,
        "arch":      _arch_from_model(model),
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
    ckpt = torch.load(path, map_location=device)

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