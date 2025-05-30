import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2


image_files = sorted(glob.glob("../../DENSE/very_small/train_sequence_00_town01/rgb/frames/*.png"))
images = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files])

images = images.astype("float32") / 255     # normalize

device = "cuda:0"
prev = torch.log(torch.from_numpy(images[0]).to(device))
cur = torch.log(torch.from_numpy(images[1])).to(device)
diff = cur - prev

def compute_event_frame(diff):
    eps = 1e-4
    C = 0.2
    w = 100
    return ((diff + eps) / (torch.abs(diff) + eps)) * (1 / (1 + torch.exp(-w*torch.abs(diff)+w*C)))

event_frame = compute_event_frame(diff)

plt.imshow(event_frame.to('cpu'), cmap='gray')
plt.show()
plt.savefig("event_frame.png", dpi=300, bbox_inches='tight')