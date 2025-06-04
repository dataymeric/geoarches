import torch
from torch import nn

from geoarches.backbones.weatherlearn_utils.pad import get_pad2d, get_pad3d


class PatchEmbed2d(nn.Module):
    """2D Tensor to Patch Embedding.

    Args:
        tensor_size (tuple[int, int]): Size of input tensor. (Lat, Lon)
        patch_size (tuple[int, int]): Patch token size. (Lat, Lon)
        in_chans (int): Number of input channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None

    Returns:
        torch.Tensor: Patch embeddings of shape (B, D, Lat', Lon')

    Shape:
        - Input: (B, C, Lat, Lon)
        - Output: (B, D, Lat', Lon')
        where Lat' = Lat // patch_size[0], Lon' = Lon // patch_size[1] and D = embed_dim.
    """

    def __init__(self, tensor_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.tensor_size = tensor_size

        self.pad = nn.ZeroPad2d(get_pad2d(tensor_size, patch_size))
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor):
        assert self.tensor_size == x.shape[2:], (
            f"Expected input size {self.tensor_size}, but got {x.shape[2:]}"
        )

        x = self.pad(x)
        x = self.proj(x)  # (B, D, Lat', Lon')
        if self.norm is not None:
            x = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return x


class PatchEmbed3d(nn.Module):
    """3D Tensor to Patch Embedding.

    Args:
        tensor_size (tuple[int, int, int]): Size of input tensor. (Pl, Lat, Lon)
        patch_size (tuple[int, int, int]): Patch token size. (Pl, Lat, Lon)
        in_chans (int): Number of input channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: None

    Returns:
        torch.Tensor: Patch embeddings of shape (B, D, Pl', Lat', Lon')

    Shape:
        - Input: (B, C, Pl, Lat, Lon)
        - Output: (B, D, Pl', Lat', Lon')
        where Pl' = Pl // patch_size[0], Lat' = Lat // patch_size[1], Lon' = Lon // patch_size[2]
        and D = embed_dim.
    """

    def __init__(self, tensor_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.tensor_size = tensor_size

        self.pad = nn.ZeroPad3d(get_pad3d(tensor_size, patch_size))
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x: torch.Tensor):
        assert self.tensor_size == x.shape[2:], (
            f"Expected input size {self.tensor_size}, but got {x.shape[2:]}"
        )

        x = self.pad(x)
        x = self.proj(x)  # (B, D, Pl', Lat', Lon')
        if self.norm:
            x = self.norm(x.movedim(1, -1)).movedim(-1, 1)
        return x
