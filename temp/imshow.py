plt.subplot(1, 2, 1) # have two plots in 1 row two columns, first plot
# assuming im is batch x channel x h x w and channel is RGB
plt.plot(im[0].detach().cpu().permute(1, 2, 0))
plt.subplot(1, 2, 2) # second plot
plt.plot(mask[0].detach().cpu())





plt.subplot(2, 2, 1);
plt.imshow(im[0].detach().cpu().permute(1,2,0));
plt.subplot(2, 2, 2);
plt.imshow(mask[0][0].detach().cpu());
plt.subplot(2, 2, 3);
plt.imshow(mask[1][0].detach().cpu());

plt.subplot(2, 2, 4);
plt.imshow(mask[2][0].detach().cpu());


plt.show();


import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


