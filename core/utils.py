import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def position_encoding(d, output_size, args):
    xyxy1 = d[:, :, :, 2:6]
    xyxy2 = d[:, :, :, 10:14]
    offset = d[:, :, :, 6:8]

    xy1_list = torch.zeros(d.shape[0], d.shape[1], d.shape[2], 4, output_size[0], output_size[1])    # 偏移量就按照xy的顺序
    xy2_list = torch.zeros(d.shape[0], d.shape[1], d.shape[2], 4, output_size[0], output_size[1])

    w1 = (xyxy1[:, :, :,2] - xyxy1[:, :, :,0])/(output_size[1]*2)
    h1 = (xyxy1[:, :, :,3] - xyxy1[:, :, :,1])/(output_size[0]*2)
    w2 = (xyxy2[:, :, :,2] - xyxy2[:, :, :,0])/(output_size[1]*2)
    h2 = (xyxy2[:, :, :,3] - xyxy2[:, :, :,1])/(output_size[0]*2)

    for i in range(output_size[0]):       # 高
        for j in range(output_size[1]):   # 宽
            x1 = xyxy1[:, :, :,0] + w1*(j*2+1)
            y1 = xyxy1[:, :, :,1] + h1*(i*2+1)
            x2 = xyxy2[:, :, :,0] + w2 * (j * 2 + 1)
            y2 = xyxy2[:, :, :,1] + h2 * (i * 2 + 1)

            xy1_list[:, :, :, 0, i, j] = x1
            xy1_list[:, :, :, 1, i, j] = y1
            xy1_list[:, :, :, 2, i, j] = 0
            xy1_list[:, :, :, 3, i, j] = 0
            xy2_list[:, :, :, 0, i, j] = x2
            xy2_list[:, :, :, 1, i, j] = y2
            xy2_list[:, :, :, 2, i, j] = offset[:, :, :,0]
            xy2_list[:, :, :, 3, i, j] = offset[:, :, :,1]


    pe = xy2_list - xy1_list
    return pe.cuda()