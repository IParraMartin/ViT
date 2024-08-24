import torch
import torch.nn.functional as F
import torch.nn as nn

from torchsummary import summary

import yaml


class PatchEmbeddingsConv(nn.Module):

    def __init__(self, img_size: int, patch_size: int = 7, in_channels: int = 1, d_model: int = 512):
        super().__init__()
        # Set a patch size and assert that it is divisible by the img_size
        # We need patches of the same size to be able to project them
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        # Calculate the number of patches (to be used in the model)
        self.n_patches = (img_size // patch_size) ** 2

        # We make the projections of patches in two steps
        self.projection = nn.Sequential(
            # We use a Conv2D layer to (1) make the patches and (2) project them to create embeddings
            # 1- in_channels: as usual
            # 2- out_channels: as usual
            # 3- kernel_size: this will be our patch size
            # 4- stride: this will be movement from patch to patch
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=d_model,
                kernel_size=patch_size, 
                stride=patch_size,
                bias=False # No bias; we are only making patches and projecting them
            ),
            # We flatten the last two dimensions (spatial) which represent the patches
            nn.Flatten(start_dim=2)
        )

    def forward(self, x) -> torch.Tensor:
        # x shape: (B, C, H, W)
        x = self.projection(x) # After Conv2D: (batch_size, d_model, H/PATCH_SIZE, W/PATCH_SIZE)
        x = x.permute(0, 2, 1) # Permute to shape: (batch_size, seq_len, d_model)
        return x


class Attention(nn.Module):
    
    def __init__(self, d_model: int = 512, heads: int = 8, dropout: int = 0.1, kv_bias: bool = False):
        super().__init__()
        # We can simplify the code by using nn.MultiheadAttention
        self.atten = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            add_bias_kv=kv_bias
        )

    def forward(self, x) -> torch.Tensor:
        # Permute to have (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x, _ = self.atten(x, x, x)
        # Permute back to have (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)
        return x


class MLP(nn.Module):

    def __init__(self, in_features: int, h_dim: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h_dim)
        self.fc2 = nn.Linear(h_dim, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x) -> torch.Tensor:
        # Usual forward pass of MLP (with gelu)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
    def initialize_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        print('FC weights initialized.')
    

class EncoderBlock(nn.Module):

    def __init__(
            self, 
            d_model: int = 512, 
            heads: int = 6, 
            h_dim: int = 1024, 
            dropout: float = 0.1, 
            kv_bias: bool = False
    ):
        super().__init__()
        # Create two layer normalization layers
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        # Set the multi-head attention
        self.attn = Attention(d_model=d_model, heads=heads, dropout=dropout, kv_bias=kv_bias)
        # Set the MLP
        self.mlp = MLP(in_features=d_model, h_dim=h_dim, out_features=d_model, dropout=dropout)

    def forward(self, x) -> torch.Tensor:
        # EncoderBlock block with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class VisionTransformer(nn.Module):

    def __init__(
            self, 
            img_size: int = 28, 
            patch_size: int = 7, 
            in_channels: int = 1, 
            num_classes: int = 10, 
            d_model: int = 512, 
            layers: int = 12, 
            heads: int = 8, 
            h_dim: int = 2048, 
            dropout: float = 0.1
        ):
        super().__init__()
        # Create the patches and embeddings
        self.patch_embed = PatchEmbeddingsConv(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels, 
            d_model=d_model
        )
        # Create the class token (learnable parameter)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Create the position embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Create the transformer layers
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model=d_model, heads=heads, h_dim=h_dim, dropout=dropout)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        n_samples = x.shape[0] # Get the number of samples
        x = self.patch_embed(x) # Get the patches
        cls_token = self.cls_token.expand(n_samples, -1, -1) # Expand the cls token
        x = torch.cat((cls_token, x), dim=1) # Concatenate the cls token with the patches
        x += self.pos_embedding # Add the position embeddings
        x = self.dropout(x) # Apply dropout
        
        for block in self.blocks:
            x = block(x) # Apply the transformer layers

        x = self.norm(x) # Apply the layer normalization
        cls_token_final = x[:, 0] # Select just cls token
        x = self.fc(cls_token_final) # Apply the final linear layer
        return x


if __name__ == '__main__':

    with open('vit_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model_config']
    model = VisionTransformer(**model_config)
    summary(model, (1, model_config['img_size'], model_config['img_size']))