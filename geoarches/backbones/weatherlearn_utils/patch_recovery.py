import torch
from torch import nn

from geoarches.backbones.weatherlearn_utils.crop import crop2d, crop3d


class PatchRecovery2D(nn.Module):
    """Patch Embedding Recovery to 2D Tensor.

    Args:
        tensor_size (tuple[int, int]): Size of input tensor.
        patch_size (tuple[int, int]): Patch token size
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.

    Returns:
        torch.Tensor: Recovered tensor of shape (B, C, Lat, Lon)
    """

    def __init__(self, tensor_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.tensor_size = tensor_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)  # (B, C, Lat, Lon)
        output = output.movedim(1, -1)  # (B, Lat, Lon, C)
        output = crop2d(output, self.tensor_size).movedim(-1, 1)  # (B, C, Lat, Lon)
        return output


class PatchRecovery3D(nn.Module):
    """Patch Embedding Recovery to 3D Tensor.

    Args:
        tensor_size (tuple[int, int, int]): Size of input tensor.
        patch_size (tuple[int, int, int]): Patch token size.
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.

    Returns:
        torch.Tensor: Recovered tensor of shape (B, C, Pl, Lat, Lon)
    """

    def __init__(self, tensor_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.tensor_size = tensor_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)  # (B, C, Pl, Lat, Lon)
        output = output.movedim(1, -1)  # (B, Pl, Lat, Lon, C)
        output = crop3d(output, self.tensor_size).movedim(-1, 1)  # (B, C, Pl, Lat, Lon)
        return output
