import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

TASK = "abnormal"
PLANE = "sagittal"
CKPT = f"mrnet_{TASK}_{PLANE}.pth"
OUT = f"mrnet_{TASK}_{PLANE}.onnx"

class MRNetSingle(nn.Module):
    """
    Browser-friendly variant: takes a single image tensor [N, 3, 224, 224]
    and outputs logits [N, 1].
    (This is not slice-pooled multi-slice MRNet.)
    """
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # -> [N, 512, 1, 1]
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        feats = self.feature_extractor(x)         # [N, 512, 1, 1]
        feats = feats.view(feats.size(0), -1)     # [N, 512]
        out = self.classifier(feats)              # [N, 1]
        return out

if __name__ == "__main__":
    device = torch.device("cpu")
    model = MRNetSingle().to(device)

    # Load weights from your trained MRNet checkpoint:
    # Your MRNet class has the same classifier + feature extractor names,
    # so this should load as long as shapes match.
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy,
        OUT,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print("Exported:", OUT)
