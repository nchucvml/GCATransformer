import torch
from torch import nn
import torch.nn.functional as F


class Group_Att(nn.Module):
    def __init__(self): #(self, group_size=5)
        super().__init__()
        # self.group_size = group_size

    def forward_one_level(self, x):
        """
        x shape:
            [B * G, C, H, W]

        B = batch size，也就是一次幾個 group
        G = group size，預設 5
        """

        BG, C, H, W = x.shape
        # G = self.group_size
        G = 5

        assert BG % G == 0, \
            f"Input batch size {BG} must be divisible by group_size {G}"

        B = BG // G

        # [B*G, C, H, W] -> [B, G, C, H, W]
        x = x.view(B, G, C, H, W)

        # [B, G, C, H, W] -> [B, G, H, W, C]
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        # normalize channel dimension
        x = F.normalize(x, dim=-1)

        N = H * W

        # [B, G, H, W, C] -> [B, G, N, C]
        f = x.view(B, G, N, C)

        output_groups = []

        for i in range(G):
            fi = f[:, i]   # [B, N, C]

            attmap = None

            for j in range(G):
                if i == j:
                    continue

                fj = f[:, j]   # [B, N, C]

                # similarity: [B, N, N]
                sim = torch.einsum("bnc,bmc->bnm", fi, fj)

                # 對 reference image 所有位置取 max
                cur_attmap = sim.max(dim=2)[0]   # [B, N]
                cur_attmap = torch.clamp(cur_attmap, min=0.1)

                if attmap is None:
                    attmap = cur_attmap
                else:
                    attmap = attmap * cur_attmap

            # 對應你原本的 attmap * torch.max(attmap)
            attmap = attmap * attmap.max(dim=1, keepdim=True)[0]

            # apply attention
            fi_out = fi * attmap.unsqueeze(-1)   # [B, N, C]

            output_groups.append(fi_out)

        # list of [B, N, C] -> [B, G, N, C]
        out = torch.stack(output_groups, dim=1)

        # [B, G, N, C] -> [B, G, H, W, C]
        out = out.view(B, G, H, W, C)

        # [B, G, H, W, C] -> [B, G, C, H, W]
        out = out.permute(0, 1, 4, 2, 3).contiguous()

        # [B, G, C, H, W] -> [B*G, C, H, W]
        out = out.view(BG, C, H, W)

        return out

    def forward(self, features):
        for level in ["res2", "res3", "res4", "res5"]:
            features[level] = self.forward_one_level(features[level])

        return features