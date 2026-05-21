import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=4, num_layers=6):
        super(TransformerBlock, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.max_hw = 256 // patch_size
        self.pos_embedding = nn.Parameter(torch.zeros(1, embed_dim, self.max_hw, self.max_hw))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        patches = self.patch_embed(x)
        B, E, H, W = patches.shape
        patches = patches.flatten(2).permute(0, 2, 1) 
        pos = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear', align_corners=False) 
        pos = pos.flatten(2).transpose(1, 2)
        patches = patches + pos
        patches = self.transformer(patches) 
        patches = patches.permute(0, 2, 1).view(B, E, H, W) 
        return patches

class CBVIT(nn.Module):
    def __init__(self, resnet, type=0):
        super(CBVIT, self).__init__()
        self.type = type # MAX, MEAN, SUM
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.down = ConvBlock(128, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        self.num_inputs = 2 
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64 * self.num_inputs, 128), 
            nn.ReLU(),
            nn.Linear(128, 64 * self.num_inputs),  # 输出通道级权重
            nn.Sigmoid() 
        )

        self.transformer = TransformerBlock(in_channels=64, patch_size=8, embed_dim=128, num_heads=4, num_layers=6)

        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=8, stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)
        print("------------------------------self.type = ", self.type)
    
    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding_to_multiple(self, tensors, divisor=16, mode='replicate'):
        out_tensors = []
        for tensor in tensors:
            h, w = tensor.shape[-2:]
            pad_h = (divisor - h % divisor) % divisor
            pad_w = (divisor - w % divisor) % divisor
            padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
            out_tensor = F.pad(tensor, padding, mode=mode)
            out_tensors.append(out_tensor)
        return out_tensors
    
    def forward(self, *tensors):
        
        orig_h, orig_w = tensors[0].shape[-2:]
        tensors = self.tensor_padding_to_multiple(tensors, divisor=8)
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)
        skip_features = outs
        outs = [self.transformer(out) for out in outs]
        outs = [self.upconv(out) for out in outs]
        outs = [torch.cat((out, skip), dim=1) for out, skip in zip(outs, skip_features)]
        outs = self.operate(self.down, outs)
        out_stack = torch.stack(outs, dim=1)  # [B, N, C, H, W]
        B, N, C, H, W = out_stack.shape
        avg_pool = out_stack.mean(dim=[3, 4])  # [B, N, C] 
        fusion_input = avg_pool.view(B, -1)  # [B, N * C]
        # weights = self.fusion_mlp(fusion_input).view(B, N, C) 
        weights = self.fusion_mlp(fusion_input).view(B, N, C, 1, 1)  # [B, N, C, 1, 1]
        if self.type == 0: 
            weighted_stack = weights * out_stack  # [B, N, C, H, W]
            out = torch.max(weighted_stack, dim=1)[0]  # [B, C, H, W] 
        elif self.type == 2:
            out = (weights * out_stack).mean(dim=1) 

        out = self.conv3(out)
        out = self.conv4(out)
        out = out[:, :, :orig_h, :orig_w] 
        return out

def build_model(type):
    resnet = models.resnet101(pretrained=True)
    model = CBVIT(resnet, type)
    return model
