import torch
import time
import torchvision

import numpy as np
import tqdm
from data_utils import *
from modeling import *
from fvcore.nn import FlopCountAnalysis
from timm.models import create_model
from ptflops import get_model_complexity_info

transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

testset = (
            torchvision.datasets.ImageFolder(
                root="/DATA/nfsshare/Amartya/EMNLP-WACV/imagenet/val",
                transform=transform_test,
            )

        )

test_loader = (
    DataLoader(
        testset,
        batch_size=32,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )
    if testset is not None
    else None
)

def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t


def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()


# get the first 100 images of COCO val2017

images = []
for idx in range(100):
    img, t = testset[idx]
    images.append(img)

device = torch.device('cuda')
results = {}
for model_name in ['vit_base_patch32_224_ucb']:
    model = create_model(model_name, pretrained=False)
    model.eval()
    # model = torch.hub.load('vit_base_patch32_224', model_name, pretrained=True)
    model.cuda()
    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(images):
            inputs = img.unsqueeze(0).cuda()
            res = FlopCountAnalysis(model, inputs)
            # macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, backend='pytorch',
            #                                print_per_layer_stat=True, verbose=True)
            t = measure_time(model, inputs)
            tmp.append(res.total())
            tmp2.append(t)
    # with torch.no_grad():
    #     tmp = []
    #     tmp2 = []
    #     for img in tqdm.tqdm(images):
    #         inputs = [img.to(device)]
    #         res = flop_count(model, (inputs,))
    #         t = measure_time(model, inputs)
    #         tmp.append(sum(res.values()))
    #         tmp2.append(t)

    results[model_name] = {'flops': fmt_res(np.array(sum(tmp))), 'time': fmt_res(np.array(tmp2))}


print('=============================')
print('')
for r in results:
    print(r)
    for k, v in results[r].items():
        print(' ', k, ':', v)