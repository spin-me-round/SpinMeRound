# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
# 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

from .model import BiSeNet

import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

class FaceMaskDetector(nn.Module):

    def __init__(self, save_path, ids2keep = [1,2,3,4,5,6,10,12,13, 14, 15]):

        super(FaceMaskDetector, self).__init__()
        self.n_classes = 19
        self.net = BiSeNet(n_classes=self.n_classes)
        self.net.load_state_dict(torch.load(save_path,map_location='cpu', weights_only=False))
        self.net.eval()
        self.trans = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.ids_to_keep = ids2keep

    @torch.no_grad()
    def forward(self, img, ret_pytorch=True, ids2keep=None, ret_all=False):

        img = self.trans(img)
        out = self.net(img)[0]
        if ret_pytorch :
            out = out.squeeze(0).argmax(0)
        else:
            out = out.squeeze(0).cpu().numpy().argmax(0)
        if ids2keep is None:
            ids2keep = self.ids_to_keep

        if ret_all:
            return out
        else:
            return (sum(out==i for i in ids2keep) != 0) * 1
