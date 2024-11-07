import torch
import time
import torchvision

import numpy as np
import tqdm

from models import build_model
from datasets import build_dataset

from flop_count import flop_count


def get_dataset(coco_path):
    """
    Gets the COCO dataset used for computing the flops on
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.dataset_file = "coco"
    args.coco_path = coco_path
    args.masks = False
    dataset = build_dataset(image_set='val', args=args)
    return dataset


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
PATH_TO_COCO = "/path/to/coco/"
dataset = get_dataset(PATH_TO_COCO)
images = []
for idx in range(100):
    img, t = dataset[idx]
    images.append(img)

device = torch.device('cuda')
results = {}
for model_name in ['detr_resnet50']:
    model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
    model.to(device)

    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(images):
            inputs = [img.to(device)]
            res = flop_count(model, (inputs,))
            t = measure_time(model, inputs)
            tmp.append(sum(res.values()))
            tmp2.append(t)

    results[model_name] = {'flops': fmt_res(np.array(tmp)), 'time': fmt_res(np.array(tmp2))}


print('=============================')
print('')
for r in results:
    print(r)
    for k, v in results[r].items():
        print(' ', k, ':', v)