import argparse
import cv2
from libCH.extractShapes import Shapes
#from libCH.descriptor import colorDescriptor
#from libCH.descriptor import histogramFeatures
#from libCH.descriptor import Moments
#from libCH.descriptor import HaralicktextureFeatures

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

shapesExtract = Shapes()
shapesExtract.loadImage(imgPath=args["image"], resize_w=600)
shapesExtract.extractShapes1(colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=True, debug=True)
cv2.waitKey(0)
