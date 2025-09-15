import torch
from config import LoopFormerForImageClassificationConfig
from model import LoopFormerForImageClassification

def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # ensure bf16, no TF32

    cfg = LoopFormerForImageClassificationConfig(
        hidden_size=256, num_loops=2, num_heads=8, intermediate_size=1024,
        num_classes=10, image_size=224, patch_size=16, num_channels=1,
        modules={"mixer_1": 1, "mixer_2": 1}
    )
    model = LoopFormerForImageClassification(cfg).to(device, dtype=torch.bfloat16).eval()
    x = torch.randn(64, cfg.num_channels, cfg.image_size, cfg.image_size,
                    device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        # warmup
        for _ in range(5):
            _ = model(x)
        torch.cuda.synchronize()

        # profiled region
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()