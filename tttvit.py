import torch
import torch.nn as nn
from ttt import TTTConfig, TTTLinear, TTTCache

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        debug_print(f"PatchEmbedding input shape: {x.shape}")
        x = self.projection(x)
        debug_print(f"PatchEmbedding after projection shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)
        debug_print(f"PatchEmbedding output shape: {x.shape}")
        return x

class TTTViTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ttt = TTTLinear(config, layer_idx)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

    def forward(self, x, cache_params, position_ids):
        debug_print(f"TTTViTBlock {self.layer_idx} input shape: {x.shape}")
        residual = x
        x = self.norm1(x)
        x = self.ttt(x, attention_mask=None, position_ids=position_ids, cache_params=cache_params)
        x = residual + x
        debug_print(f"TTTViTBlock {self.layer_idx} after TTT shape: {x.shape}")
        
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        debug_print(f"TTTViTBlock {self.layer_idx} output shape: {x.shape}")
        return x

class TTTViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
        self.config = TTTConfig(
            hidden_size=embed_dim,
            intermediate_size=int(embed_dim * mlp_ratio),
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            max_position_embeddings=self.patch_embed.num_patches + 1,
            ttt_base_lr=0.1,
            mini_batch_size=16,
            ttt_layer_type="linear"
        )
        
        self.blocks = nn.ModuleList([TTTViTBlock(self.config, i) for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        debug_print(f"TTTViT input shape: {x.shape}")
        x = self.patch_embed(x)
        debug_print(f"TTTViT after patch embedding shape: {x.shape}")
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        debug_print(f"TTTViT after adding CLS token shape: {x.shape}")
        
        x = x + self.pos_embed
        debug_print(f"TTTViT after adding positional embedding shape: {x.shape}")
        
        cache_params = TTTCache(self, x.size(0))
        
        # Generate position_ids
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        
        for i, block in enumerate(self.blocks):
            x = block(x, cache_params, position_ids)
            debug_print(f"TTTViT after block {i} shape: {x.shape}")
        
        x = self.norm(x)
        debug_print(f"TTTViT after final norm shape: {x.shape}")
        
        x = x[:, 0]  # Take the CLS token representation
        debug_print(f"TTTViT after selecting CLS token shape: {x.shape}")
        
        x = self.head(x)
        debug_print(f"TTTViT output shape: {x.shape}")
        return x
    def create_ttt_cache(self, batch_size):
        cache_params = TTTCache(self, batch_size)
        for i, block in enumerate(self.blocks):
            for name in cache_params.ttt_param_names:
                weight = getattr(block.ttt, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(weight.device)
                cache_params.ttt_params_dict[f"{name}_states"][i] = tiled_weight
                cache_params.ttt_params_dict[f"{name}_grad"][i] = torch.zeros_like(tiled_weight)
        return cache_params
    def to(self, device):
        super().to(device)
        self.device = device
        return self


if __name__ == "__main__":
    # Set up the model
    image_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 1000
    embed_dim = 768
    depth = 12
    num_heads = 12

    model = TTTViT(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    )

    # Create a sample input
    batch_size = 4
    x = torch.randn(batch_size, in_channels, image_size, image_size)

    # Forward pass
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")