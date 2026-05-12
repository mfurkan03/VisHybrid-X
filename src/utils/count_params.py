import sys
import torch
sys.path.insert(0, "src")
from models import build_policy

IMAGE_SIZE = 84

def count_params(model):
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable
    return total, trainable, frozen

def report(arch):
    model = build_policy(arch, image_size=IMAGE_SIZE)
    model.eval()
    total, trainable, frozen = count_params(model)

    print(f"\n{'='*50}")
    print(f"  arch        : {arch}  (image_size={IMAGE_SIZE})")
    print(f"  total       : {total:>12,}")
    print(f"  trainable   : {trainable:>12,}")
    print(f"  frozen      : {frozen:>12,}")
    print(f"{'='*50}")

    print(f"\n  {'Module':<30} {'Params':>10}")
    print(f"  {'-'*42}")
    for name, module in model.named_modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        if own > 0:
            print(f"  {name:<30} {own:>10,}")

if __name__ == "__main__":
    archs = sys.argv[1:] or ["impala_v2"]
    for arch in archs:
        report(arch)
