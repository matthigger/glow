import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def imshow_rgb(y, mask_idx, label=None, mask=None):
    x = np.zeros((*mask_idx.shape, 3))
    x[mask_idx >= 0] = y.T
    x[x < 0] = 0
    x[x > 255] = 255
    x = x.astype(np.uint8)

    plt.imshow(x)
    plt.axis('off')

    if label is not None:
        plt.gca().set_title(label)

    if mask is not None:
        # set mask as alpha channel in green highlighted area
        im_mask = np.zeros((*mask_idx.shape, 4), dtype=np.uint8)
        # blue
        im_mask[:, :, 1] = 255
        im_mask[:, :, 3] = mask * 255
        plt.imshow(im_mask)
