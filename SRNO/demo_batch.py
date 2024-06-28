import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--scale')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    scale_max = 4
    
    for filename in os.listdir(args.input):
        if filename.endswith('.png'):
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, f"l8_{filename}")
            
            img = transforms.ToTensor()(Image.open(input_path).convert('RGB'))
            model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
            h = int(img.shape[-2] * int(args.scale))
            w = int(img.shape[-1] * int(args.scale))
            scale = h / img.shape[-2]
            coord = make_coord((h, w), flatten=False).cuda()
            cell = torch.ones(1,2).cuda()
            cell[:, 0] *= 2 / h
            cell[:, 1] *= 2 / w
            
            cell_factor = max(scale/scale_max, 1)
            pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0),coord.unsqueeze(0), cell_factor*cell).squeeze(0)
            pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(3, h, w).cpu()
            transforms.ToPILImage()(pred).save(output_path)
