import torch
from torch import nn


class LearnedPrompt(nn.Module):
    """Learned prompt embeddings for SAM-VTP."""

    def __init__(self, embed_dim, image_embedding_size, input_image_size, num_tokens):
        super().__init__()
        self.embed_dim = embed_dim
        g = image_embedding_size[0]
        self.image_embedding_size = (g, g)
        self.input_image_size = input_image_size
        self.num_tokens = num_tokens
        self.sparse_tokens = nn.Parameter(
            torch.randn(num_tokens, embed_dim)
        )  # (4, 256)
        self.dense_emb = nn.Parameter(torch.randn(embed_dim, g, g))  # (256, 64, 64)
        self.learned_pe = nn.Parameter(torch.randn(embed_dim, g, g))  # (256, 64, 64)

    def forward(self, points=None, boxes=None, masks=None):
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif masks is not None:
            bs = masks.shape[0]
        else:
            bs = 1
        return (
            self.sparse_tokens.unsqueeze(0).expand(bs, -1, -1),
            self.dense_emb.unsqueeze(0).expand(bs, -1, -1, -1),
        )

    def get_dense_pe(self) -> torch.Tensor:
        return self.learned_pe.unsqueeze(0)
