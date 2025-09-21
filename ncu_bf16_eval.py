import torch
from config import LoopFormerForImageClassificationConfig
from model import LoopFormerForImageClassification

from flazoo import DeltaNetForImageClassification, DeltaNetVisionConfig, log_model

def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # ensure bf16, no TF32

    # NOTE: loopformer
    # cfg = LoopFormerForImageClassificationConfig(
    #     hidden_size=256, num_loops=4, num_heads=8, intermediate_size=1024,
    #     num_classes=10, image_size=224, patch_size=16, num_channels=1,
    #     modules={"mixer_2": 2}
    # )
    # model = LoopFormerForImageClassification(cfg).to(device, dtype=torch.bfloat16).eval()

    # NOTE: deltanet
    attn_config = {
        'layers': [],
        'num_heads': 16,
        "window_size_h": 48,
        "window_size_w": 24,
        "tile_size_h": 16,
        "tile_size_w": 8,
        "seq_len" : 4096,
    }
    cfg = DeltaNetVisionConfig(
        hidden_size=256, num_hidden_layers=4, num_heads=8, intermediate_size=1024,
        num_classes=10, image_size=224, patch_size=16, num_channels=1,
        attn_type="full_attn",
        train_scan_type="uni-scan",
        attn=attn_config,
    )
    model = DeltaNetForImageClassification(cfg).to(device, dtype=torch.bfloat16).eval()

    x = torch.randn(64, cfg.num_channels, cfg.image_size, cfg.image_size,
                    device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("steady_eval")
    
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
        
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

if __name__ == "__main__":
    main()