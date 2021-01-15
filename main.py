import cv2 as cv
import numpy as np 

h_img = cv.imread('fotos/ab.jpg')
p_img = cv.imread('fotos/ac.jpg')

result = cv.matchTemplate(h_img, p_img, cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)



cv.imshow('resultado', result)
print(max_val, max_loc)
cv.waitKey()
