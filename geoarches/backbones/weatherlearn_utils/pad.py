def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int, int, int]): input resolution (Pl, Lat, Lon)
        window_size (tuple[int, int, int]): target window size (Pl, Lat, Lon)

    Returns:
        padding: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int, int]): input resolution (Lat, Lon)
        window_size (tuple[int, int]): target window size (Lat, Lon)

    Returns:
        padding: (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]
