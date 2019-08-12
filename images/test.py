import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150) #sick
    return canny
sourceimage = cv2.imread('/data/yy/CenterNet/visDemo/2019-08-04 08:45:31.082ctdet.png')
#sourceimage = cv2.imread('/data/yy/CenterNet/images/9064748793_bb942deea1_k.jpg')
print(sourceimage)
img = np.copy(sourceimage)
print(np.shape)
canny = canny(img)
