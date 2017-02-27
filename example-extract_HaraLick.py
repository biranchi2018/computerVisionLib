import argparse
import cv2
from libCH.descriptor import TextureFeatures

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

hara = TextureFeatures()
hara.loadImage(imgPath=args["image"], resize_w=400)
features = hara.HaralickFeatures1(colorSpace="RGB", channelSelect=(0,0,1), debug=True)
print features
