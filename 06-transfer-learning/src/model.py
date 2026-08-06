import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CIFAR10ResNet18(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the final FC layer (1000 ImageNet classes -> 10 CIFAR-10 classes)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        # New head is always trainable, even if backbone is frozen
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Call this to unfreeze the backbone for phase-2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True