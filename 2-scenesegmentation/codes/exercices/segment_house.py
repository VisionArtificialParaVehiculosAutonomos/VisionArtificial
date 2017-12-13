import numpy as np
import cv2
import matplotlib.pyplot as plt

def segmentLayer(imHSV,y1, y2, x1, x2):
    patchBGR = imBGR[y1:y2, x1:x2]
    patchHSV = imHSV[y1:y2, x1:x2]
    # plt.figure()
    # plt.imshow(patchBGR[..., ::-1])
    # plt.show()

    # Build a histogram:
    hist = cv2.calcHist([patchHSV], channels=[0], mask=None, histSize=[180], ranges=[0, 180])
    histNorma = hist / max(hist)

    # Compute the histogram back projection:
    return cv2.calcBackProject([imHSV], channels=[0], hist=histNorma, ranges=[0, 180], scale=255)


# Read image
imBGR = cv2.imread("../ressources/house.jpg")

# Convert to BGR to HSV
imHSV = cv2.cvtColor(imBGR, cv2.COLOR_BGR2HSV)

# Segment brick
brickMap = segmentLayer(imHSV, 300, 350, 300, 350)

# Segment grass
grassMap = segmentLayer(imHSV, 900, 950, 400, 450)

# Segment roof
roofMap = segmentLayer(imHSV, 160, 190, 820, 850)

# Segment pathway
pathwayMap = segmentLayer(imHSV, 820, 850, 630, 660)

# Don't forget to plot a BGR image
blurBrick = cv2.GaussianBlur(brickMap,(9,9),0)
_ ,thresh = cv2.threshold(brickMap, 50, 255, 0)
mask = cv2.merge([thresh , thresh , thresh ])


res = cv2.bitwise_and(imHSV,mask)

plt.figure()
plt.imshow(mask[..., ::-1])
plt.show()
