import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_height=294, img_width=531):
        super().__init__()
        self.patch_size = patch_size
        # Calculate right and bottom padding only
        self.pad_right = (patch_size - img_width % patch_size) % patch_size
        self.pad_bottom = (patch_size - img_height % patch_size) % patch_size
        self.padding = (0, self.pad_right, 0, self.pad_bottom)  # Left, Right, Top, Bottom
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Apply padding
        # print('input shape in start of PatchEmbedding:', x.shape)
        x = F.pad(x, self.padding, "constant", 0)
        # print('input shape after padding in PatchEmbedding:', x.shape)
        x = self.proj(x)
        # print('output shape after proj in PatchEmbedding:', x.shape)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        # print('output shape in end of PatchEmbedding:', x.shape)
        # exit()
        return x


class PatchReconstructor(nn.Module):
    def __init__(self, emb_size, patch_size, out_channels=1, img_height=294, img_width=531):
        super().__init__()
        self.emb_size = emb_size
        self.patch_size = patch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_h_patches = img_height // patch_size if img_height % patch_size == 0 else img_height // patch_size + 1
        self.num_w_patches = img_width // patch_size if img_width % patch_size == 0 else img_width // patch_size + 1
        self.proj = nn.ConvTranspose2d(emb_size, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        
        # print('input shape in start of PatchReconstructor:', x.shape)
        x = x.transpose(1, 2)
        x = x.view(x.shape[0], self.emb_size, self.num_h_patches, self.num_w_patches)
        # print('input shape after reshaping in PatchReconstructor:', x.shape)
        x = self.proj(x)
        # print('output shape in end of PatchReconstructor:', x.shape)
        # Crop the padded areas
        crop_height = x.shape[2] - ((self.patch_size - self.img_height % self.patch_size) % self.patch_size)
        crop_width = x.shape[3] - ((self.patch_size - self.img_width % self.patch_size) % self.patch_size)
        x = x[:, :, :crop_height, :crop_width]
        # print('output shape in end of PatchReconstructor after cropping:', x.shape)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, forward_expansion, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=num_heads,
                dim_feedforward=forward_expansion * emb_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, src):
        x = src
        for layer in self.layers:
            x = layer(x)
        return x


class gls_modules(nn.Module):
    def __init__(self, c, h, w, emb_dim, patch_size, num_heads, forward_expansion, dropout, num_layers):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=c, patch_size=patch_size, emb_size=emb_dim, img_height=h, img_width=w)
        self.transformer_g = TransformerBlock(emb_size=emb_dim, num_heads=num_heads, forward_expansion=forward_expansion, dropout=dropout, num_layers=num_layers)
        self.transformer_l = TransformerBlock(emb_size=emb_dim, num_heads=num_heads, forward_expansion=forward_expansion, dropout=dropout, num_layers=num_layers)
        self.transformer_s = TransformerBlock(emb_size=emb_dim, num_heads=num_heads, forward_expansion=forward_expansion, dropout=dropout, num_layers=num_layers)
        self.patch_recon = PatchReconstructor(emb_size=emb_dim, patch_size=patch_size, out_channels=c, img_height=h, img_width=w)

    def patchify(self, x):
        return self.patch_embed(x)

    def dePatchify(self, x):
        return self.patch_recon(x)
    
    def get_gls(self, x):
        mg = self.patch_recon(self.transformer_g(x))
        ml = self.patch_recon(self.transformer_l(x))
        ms = self.patch_recon(self.transformer_s(x))
        return mg, ml, ms

    def forward(self, x):
        x = self.patchify(x)
        mg, ml, ms = self.get_gls(x)
        return mg, ml, ms

def create_mapping(c, h, w, emb_dim, patch_size, num_heads, forward_expansion, dropout, num_layers):
    return gls_modules(c, h, w, emb_dim, patch_size, num_heads, forward_expansion, dropout, num_layers)