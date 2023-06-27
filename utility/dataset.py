from scipy import ndimage


def dilate_pixel(args, mask, label_y, label_x):
    if label_y == 512: label_y -= 1
    if label_x == 512: label_x -= 1
    mask[label_y][label_x] = 1.0
    struct = ndimage.generate_binary_structure(2, 1)
    dilated_mask = ndimage.binary_dilation(mask, structure=struct, iterations=args.dilate).astype(mask.dtype)

    return dilated_mask