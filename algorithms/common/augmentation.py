import numpy as np
from skimage.util.shape import view_as_windows


def random_crop(images, crop_size, r_vec=None, c_vec=None):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones
    args:
        imgs, batch images with shape (B,H,W,C)
    """
    # batch size
    batch_size, h, w, c = images.shape
    assert h == w
    crop_max = h - crop_size
    r_vec = np.random.randint(0, crop_max, batch_size) if r_vec is None else r_vec
    c_vec = np.random.randint(0, crop_max, batch_size) if c_vec is None else c_vec
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(images, (batch_size, crop_size, crop_size, c))[0, :, :, 0]
    # selects a random window for each batch element
    cropped_images = windows[r_vec, c_vec, np.arange(batch_size)]

    return cropped_images


def center_crop_image(images, crop_size):
    """
    :param images: (..., H, W, C)
    :param crop_size:
    :return:
    """
    h, w, c = images.shape[-3:]
    new_h, new_w = crop_size, crop_size
    top = (h - new_h) // 2
    left = (w - new_w) // 2

    cropped_images = images[..., top:top + new_h, left:left + new_w, :]

    return cropped_images


if __name__ == "__main__":
    images = np.arange(2 * 10 * 10 * 1).reshape(2, 10, 10, 1)
    images = np.tile(images, (1, 1, 1, 5))
    cropped_images = random_crop(images, 3)
    print(cropped_images.shape)
