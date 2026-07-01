"""
32x32 tour — VISION TRANSFORMER (CLIP-style, reused as the CLIP image encoder).
                                    ***TASK: implement Attention.forward + Block.forward.***

A plain ViT built the way CLIP's vision tower is: patchify -> prepend a learned CLS token
-> add a LEARNED positional embedding -> pre-LN transformer blocks -> take the CLS token
-> project. You implement multi-head self-attention BY HAND (no nn.MultiheadAttention, no
fused SDPA) and the pre-norm block; the patch embed, CLS token, positional embedding, and
container are provided.

`VisionTransformer` is CLIP-compatible and is imported directly by the CLIP model
(clip/model.py) as its `vision="vit"` encoder. For the tour, `output_dim` is the number of
classes, so the projection doubles as the classifier.

    from expirements.task.model_vit import model_vit
    model = model_vit(size="resnet18")            # 32x32 -> 4x4 patches (8x8 grid + cls = 65)
"""
import torch
import torch.nn as nn

try:
    from train import NUM_CLASSES
except Exception:
    NUM_CLASSES = 100

# "resnet18/34" keys are only for interface parity with the rest of the tour.
CONFIGS = {
    "resnet18": dict(width=384, layers=9, heads=6),     # ~16M params (macro reference)
    "resnet34": dict(width=384, layers=12, heads=6),    # deeper variant
}
PATCH = 4
IMG = 32
MLP_RATIO = 4


class QuickGELU(nn.Module):
    """CLIP's activation: x * sigmoid(1.702 x)."""

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    """Multi-head self-attention, written out (softmax(QK^T / sqrt(d)) V)."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        # TODO: attention слои, q,k,v, out слои
         
    def forward(self, x):
        # x: [B, L, C]. TODO (implement by hand, можно SDPA):
        raise NotImplementedError("TODO: hand-written multi-head self-attention")


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, dim, heads, mlp_ratio=MLP_RATIO):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        # self.mlp = MLP слой

    def forward(self, x):
        # TODO: attn и mlp слои 
        raise NotImplementedError("TODO: pre-norm transformer block")


class VisionTransformer(nn.Module):
    """CLIP-style ViT. forward(image [N,3,H,W]) -> [N, output_dim]."""

    def __init__(self, image_size=IMG, patch_size=PATCH, width=384, layers=9, heads=6,
                 output_dim=NUM_CLASSES, mlp_ratio=MLP_RATIO):
        super().__init__()
        
        #self.patch_embed = ...
        #self.positional_embedding = ...# 
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads, mlp_ratio) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        # TODO: VIT forward
        # x: [b, c, h, w]
        # patchify
        # перевод в [b, tokens, h] формат
        # CLS token и pos embeddings
        # ...
        return ...

def model_vit(size="resnet18", num_classes=None):
    """Tour ViT: the projection maps the CLS token straight to `num_classes` logits."""
    assert size in CONFIGS, f"size must be one of {set(CONFIGS)}"
    return VisionTransformer(image_size=IMG, patch_size=PATCH, output_dim=num_classes, **CONFIGS[size])


if __name__ == "__main__":
    for s in ("resnet18", "resnet34"):
        m = model_vit(s)
        p = sum(x.numel() for x in m.parameters()) / 1e6
        y = m(torch.zeros(2, 3, 32, 32))
        print(f"{s:9s}: {p:5.2f}M params, out {tuple(y.shape)}")
