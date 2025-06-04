import torch


def crop2d(x: torch.Tensor, resolution):
    """Crops a 2D tensor to a specified resolution.

    Args:
        x (torch.Tensor): input tensor of shape (B, Lat, Lon, C)
        resolution (tuple[int, int]): output resolution (Lat, Lon)

    Returns:
        torch.Tensor: Cropped tensor of shape (B, Lat, Lon, C)
    """
    _, Lat, Lon, _ = x.shape

    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[:, padding_top : Lat - padding_bottom, padding_left : Lon - padding_right, :]


def crop3d(x: torch.Tensor, resolution):
    """Crops a 3D tensor to a specified resolution.

    Args:
        x (torch.Tensor): input tensor of shape (B, Pl, Lat, Lon, C)
        resolution (tuple[int, int, int]): output resolution (Pl, Lat, Lon)

    Returns:
        torch.Tensor: Cropped tensor of shape (B, Pl, Lat, Lon, C)
    """
    _, Pl, Lat, Lon, _ = x.shape

    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
        :,
    ]
