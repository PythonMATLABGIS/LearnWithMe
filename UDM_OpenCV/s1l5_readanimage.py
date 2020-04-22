import numpy as np
import cv2

img = cv2.imread('images//flower1.jpg')
cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()