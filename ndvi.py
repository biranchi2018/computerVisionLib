import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--cspace", required=True, help="RGB, LBA, HSV, HSL, LUV color space")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
resizeDisplay = imutils.resize(image, width = 500)

if(args["cspace"] == "LAB"):
    transformORiginal = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    transformResized = cv2.cvtColor(resizeDisplay, cv2.COLOR_BGR2LAB)
elif(args["cspace"] == "HSV"):
    transformOriginal = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    transformResized = cv2.cvtColor(resizeDisplay, cv2.COLOR_BGR2HSV)
elif(args["cspace"] == "HSL"):
    transformOriginal = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    transformResized = v2.cvtColor(resizeDisplay, cv2.COLOR_BGR2HLS)


cv2.imshow("Original", resizeDisplay)
cv2.imshow("NDVI", cv2.imread("NDVI/b.jpg"))

channels = cv2.split(transformOriginal)
resizedChannels = cv2.split(transformResized)

for (i,channel) in enumerate(resizedChannels):
    winTitle = args["cspace"] + " - channel " + str(i)
    cv2.imshow(winTitle, channel)

#Use channel 0
ret,thresh1 = cv2.threshold(resizedChannels[1],90,255,cv2.THRESH_BINARY)
cv2.imshow("test1", thresh1)



cv2.waitKey(0)
