import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# (layer2 channels @ stride 8, layer3 channels @ stride 16)
CHANNELS = {"resnet18": (128, 256), "resnet34": (128, 256), "resnet50": (512, 1024), "resnet101": (512, 1024)}


class FrozenResNet(nn.Module):
    """ImageNet ResNet truncated after layer3, fully frozen (weights + BatchNorm statistics).

    The same frozen instance encodes both modalities: RGB frames and 3-channel motion maps
    (|I_t - I_a|, |I_t - I_b|, min). Since nothing here is trained, sharing costs no accuracy and halves memory.
    """

    def __init__(self, name="resnet18", pretrained=True):
        super().__init__()
        if name not in CHANNELS:
            raise ValueError(f"unsupported backbone {name}; choose from {list(CHANNELS)}")
        net = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.channels = CHANNELS[name]
        for p in self.parameters():
            p.requires_grad_(False)
        super().train(False)

    def train(self, mode=True):  # always stay in eval mode (frozen BN)
        return super().train(False)

    def forward(self, x):
        with torch.no_grad():
            x = self.layer1(self.stem(x))
            c2 = self.layer2(x)
            c3 = self.layer3(c2)
        return c2, c3


def conv_gn_act(cin, cout, k=3, groups_gn=32):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.GroupNorm(min(groups_gn, cout), cout),
                         nn.GELU())


class Neck(nn.Module):
    """Trainable per-modality projection: merges stride-8 and stride-16 features into one d-channel map."""

    def __init__(self, c2, c3, d):
        super().__init__()
        self.p2 = nn.Sequential(nn.Conv2d(c2, d, 1, bias=False), nn.GroupNorm(32, d))
        self.p3 = nn.Sequential(nn.Conv2d(c3, d, 1, bias=False), nn.GroupNorm(32, d))
        self.smooth = conv_gn_act(d, d, 3)

    def forward(self, c2, c3):
        x = self.p2(c2) + F.interpolate(self.p3(c3), size=c2.shape[-2:], mode="nearest")
        return self.smooth(x)
