import cv2
import numpy as np
import matplotlib.pyplot as plt

file = "aug_0_7904.jpg"
image = cv2.imread(file, 0)
plt.imshow(image, cmap='gray')

# Read image to be aligned
imFilename = "scanned-form.jpg"
print("Reading image to align : ", imFilename)
im = cv2.imread(imFilename, 0)

print("Aligning images ...")
# Registered image will be resotred in imReg.
# The estimated homography will be stored in h.
imReg, h = alignImages(im, imReference)

# Write aligned image to disk.
outFilename = "aligned.jpg"
print("Saving aligned image : ", outFilename)
cv2.imwrite(outFilename, imReg)

# Print estimated homography
print("Estimated homography : \n",  h)
