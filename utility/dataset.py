from scipy import ndimage


def dilate_pixel(args, mask, label_y, label_x):
    if args.dilate == 0:
        return mask
    
    if label_y == 512: label_y -= 1
    if label_x == 512: label_x -= 1
    mask[label_y][label_x] = 1.0
    struct = ndimage.generate_binary_structure(rank=2, connectivity=args.connectivity)
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=args.dilate).astype(mask.dtype)

    return dilated_mask


def black_line():
    pass


def compress_pixel_values():
    pass