#Read an image with all color
import numpy as np
import cv2
img1 = cv2.imread('images//flower1.jpg')
cv2.imwrite('images//flower1.png',img1)
img2 = cv2.imread('images//flower1.png',0)
cv2.imshow('Original',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()