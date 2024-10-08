# TTTVit
mashup of TTT and Vit



```python
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
```

```shell
TTTViT input shape: torch.Size([4, 3, 224, 224])
PatchEmbedding input shape: torch.Size([4, 3, 224, 224])
PatchEmbedding after projection shape: torch.Size([4, 768, 14, 14])
PatchEmbedding output shape: torch.Size([4, 196, 768])
TTTViT after patch embedding shape: torch.Size([4, 196, 768])
TTTViT after adding CLS token shape: torch.Size([4, 197, 768])
TTTViT after adding positional embedding shape: torch.Size([4, 197, 768])
TTTViTBlock 0 input shape: torch.Size([4, 197, 768])
TTTViTBlock 0 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 0 output shape: torch.Size([4, 197, 768])
TTTViT after block 0 shape: torch.Size([4, 197, 768])
TTTViTBlock 1 input shape: torch.Size([4, 197, 768])
TTTViTBlock 1 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 1 output shape: torch.Size([4, 197, 768])
TTTViT after block 1 shape: torch.Size([4, 197, 768])
TTTViTBlock 2 input shape: torch.Size([4, 197, 768])
TTTViTBlock 2 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 2 output shape: torch.Size([4, 197, 768])
TTTViT after block 2 shape: torch.Size([4, 197, 768])
TTTViTBlock 3 input shape: torch.Size([4, 197, 768])
TTTViTBlock 3 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 3 output shape: torch.Size([4, 197, 768])
TTTViT after block 3 shape: torch.Size([4, 197, 768])
TTTViTBlock 4 input shape: torch.Size([4, 197, 768])
TTTViTBlock 4 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 4 output shape: torch.Size([4, 197, 768])
TTTViT after block 4 shape: torch.Size([4, 197, 768])
TTTViTBlock 5 input shape: torch.Size([4, 197, 768])
TTTViTBlock 5 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 5 output shape: torch.Size([4, 197, 768])
TTTViT after block 5 shape: torch.Size([4, 197, 768])
TTTViTBlock 6 input shape: torch.Size([4, 197, 768])
TTTViTBlock 6 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 6 output shape: torch.Size([4, 197, 768])
TTTViT after block 6 shape: torch.Size([4, 197, 768])
TTTViTBlock 7 input shape: torch.Size([4, 197, 768])
TTTViTBlock 7 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 7 output shape: torch.Size([4, 197, 768])
TTTViT after block 7 shape: torch.Size([4, 197, 768])
TTTViTBlock 8 input shape: torch.Size([4, 197, 768])
TTTViTBlock 8 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 8 output shape: torch.Size([4, 197, 768])
TTTViT after block 8 shape: torch.Size([4, 197, 768])
TTTViTBlock 9 input shape: torch.Size([4, 197, 768])
TTTViTBlock 9 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 9 output shape: torch.Size([4, 197, 768])
TTTViT after block 9 shape: torch.Size([4, 197, 768])
TTTViTBlock 10 input shape: torch.Size([4, 197, 768])
TTTViTBlock 10 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 10 output shape: torch.Size([4, 197, 768])
TTTViT after block 10 shape: torch.Size([4, 197, 768])
TTTViTBlock 11 input shape: torch.Size([4, 197, 768])
TTTViTBlock 11 after TTT shape: torch.Size([4, 197, 768])
TTTViTBlock 11 output shape: torch.Size([4, 197, 768])
TTTViT after block 11 shape: torch.Size([4, 197, 768])
TTTViT after final norm shape: torch.Size([4, 197, 768])
TTTViT after selecting CLS token shape: torch.Size([4, 768])
TTTViT output shape: torch.Size([4, 1000])
Input shape: torch.Size([4, 3, 224, 224])
Output shape: torch.Size([4, 1000])
```
