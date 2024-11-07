import torch
import time
from timm.models import create_model

# import models

    # "tf_efficientnet_b7": (70, 600),
    # "tf_efficientnet_b6": (80, 528),
    # "tf_efficientnet_b5": (160, 456),
    # "tf_efficientnet_b4": (240, 380),
    # "tf_efficientnet_b3": (350, 300),
    # "tf_efficientnet_b2": (700, 260),
    # "tf_efficientnet_b1": (900, 240),
    # "tf_efficientnet_b0": (1000, 224),
    # "deit_tiny_patch16_224": (3800, 224),
    # "deit_small_patch16_224": (3100, 224),
    # "deit_base_patch16_224": (1600, 224),
# _MODEL_PARAMS = {
#     "vit_small_patch16_224": (256, 224),
#     "vit_small_patch32_224": (256, 224),
#     "vit_base_patch32_224": (256, 224),
#     "vit_base_patch16_224": (256, 224),
#     "vit_large_patch32_224": (256, 224),
#     "vit_large_patch16_224": (256, 224),
#     "vit_huge_patch14_224": (256, 224),
# }
_MODEL_PARAMS = {
    "vit_small_patch8_224.dino": (256, 224),
    "vit_base_patch8_224.dino": (256, 224),
    "vit_small_patch8_224": (256, 224),
    "vit_small_patch16_224": (256, 224),
    "vit_small_patch32_224": (256, 224),
}

@torch.no_grad()
def compute_throughput(model_name):
    torch.cuda.empty_cache()
    warmup_iters = 3
    num_iters = 30
    device = torch.device('cuda')

    model = create_model(model_name, pretrained=False)
    model.eval()
    model.to(device)
    timing = []

    batch_size, resolution = _MODEL_PARAMS[model_name]

    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # warmup
    for _ in range(warmup_iters):
        model(inputs)

    torch.cuda.synchronize()
    for _ in range(num_iters):
        start = time.time()
        model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    return batch_size / timing.mean()


if __name__ == "__main__":
    for model_name in _MODEL_PARAMS.keys():
        imgs_per_sec = compute_throughput(model_name)
        print(f"{model_name}: {imgs_per_sec:.2f}")