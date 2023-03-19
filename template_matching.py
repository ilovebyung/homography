import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('rotated.jpg')
plt.imshow(img_rgb, cmap='gray')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.jpg', 0)
h, w = template.shape[::]

# methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# For all others it is exactly opposite, max value = good fit.


min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  # Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)
roi = img_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
plt.imshow(roi, cmap='gray')
# White rectangle with thickness 2.
cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)
croped_image = img_gray[max_loc[0]:min_loc[0]][min_loc[1]:max_loc[1]]

plt.imshow(img_gray, cmap='gray')

cv2.imshow("Matched image", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()


# Template matching - multiple objects

# For multiple occurances, cv2.minMaxLoc() won’t give all the locations
# So we need to set a threshold

img_rgb = cv2.imread('000.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('pen.jpg', 0)
h, w = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
plt.imshow(res, cmap='gray')

# Pick only values above 0.8. For TM_CCOEFF_NORMED, larger values = good fit.
threshold = 0.8

loc = np.where(res >= threshold)
# Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

# Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
# then the second item and then third, etc.

# -1 to swap the values as we assign x and y coordinate to draw the rectangle.
for pt in zip(*loc[::-1]):
    # Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
    # Red rectangles with thickness 2.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

#cv2.imwrite('images/template_matched.jpg', img_rgb)
cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()
