import cv2

img = cv2.imread("cinque_terre_small.jpg")



# Apply Gaussian blur
reference_image = cv2.GaussianBlur(img, (9, 9), 2)


cv2.imwrite('py_hw2_ref_cinque_terre_small.jpg', reference_image)