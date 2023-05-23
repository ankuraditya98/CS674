import torch

mask = torch.empty((3,3))
height, width = mask.shape

mask.fill_(1)

print(mask)

# # A
# mask[height//2, width//2:] = 0
# mask[height//2+1:, :] = 0

# print("A: ", mask)

# B
mask[height//2, width//2+1:] = 0
mask[height//2+1:, :] = 0

print("B: ", mask)