import sys, unittest.mock as mock
sys.path.insert(0, r'src')
with mock.patch.dict('sys.modules', {
    'depth_anything_v2': mock.MagicMock(),
    'depth_anything_v2.dpt': mock.MagicMock(),
    'cv2': mock.MagicMock(),
    'metadrive': mock.MagicMock(),
}):
    from models import ImpalaNet, ImpalaNetV2, DrivingPolicyNet

def count(model):
    total = sum(p.numel() for p in model.parameters())
    bd = {name: sum(p.numel() for p in mod.parameters()) for name, mod in model.named_children()}
    return total, bd

for arch, cls in [('ImpalaNet', ImpalaNet), ('ImpalaNetV2', ImpalaNetV2), ('DrivingPolicyNet', DrivingPolicyNet)]:
    m = cls(image_size=84)
    total, bd = count(m)
    print(f'\n{arch}: {total:,} params')
    for k, v in bd.items():
        print(f'  {k:25s} {v:>10,}')
