import argparse
import cv2
from libCH.descriptor import colorDescriptor
#from libCH.descriptor import histogramFeatures
#from libCH.descriptor import Moments
#from libCH.descriptor import HaralicktextureFeatures

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

colorFeature = colorDescriptor(shapesBlur=11, shapesThreshold=(140,140,150), shapesInvert=False, debug=True)
colorFeature.loadImage(imgPath=args["image"], resize_w=500)
print colorFeature.histogramsFeatures1(colorSpace="LAB", onlyShapes=True, channelSelect=(0,1,0), bins=[8,8,8], range=[0, 256, 0, 256, 0, 256])



#print colorFeature.histogramsFeatures1(colorSpace="HSV", channelSelect=(1,1,0), bins=[8,8,8], range=[0, 256, 0, 256, 0, 256])
#print colorFeature.channelFeatures1(colorSpace="LAB", putRGBLayers=True, onlyShapes=True, channelSelect=(0,1,0))
#print colorFeature.channelFeatures1(colorSpace="LUV", channelSelect=(1,1,1))

#print colorFeature.channelFeatures1(colorSpace="LAB", putRGBLayers=True, onlyShapes=True, channelSelect=(0,1,0))
#print colorFeature.histogramsFeatures1(colorSpace="HSV", channelSelect=(1,1,0), bins=[8,8,8], range=[0, 256, 0, 256, 0, 256])


'''
colorFeature.displayImage(title="Featured")
colorFeature.waitWindowClose()

histFeature = histogramFeatures("LAB")
histFeature.loadImage(imgPath="test.jpg", resize_w=500)
histFeature.displayImage(title="Original")
print histFeature.describe(channelSelect=(1,0,1), bins=[8,8,8], range=[0, 256, 0, 256, 0, 256])

histFeature.displayImage(title="Featured")
histFeature.waitWindowClose()

hu = Moments()
hu.loadImage(imgPath=args["image"], resize_w=400)
#features = hu.extractObject(colorSpace="LAB", channelSelect=(0,1,1), blur=11, threshold=(120,140,150), bwInvert=False, debug=True)
features = hu.ZernikeFeatures(colorSpace="LAB", channelSelect=(1,1,1), blur=11, threshold=(120,140,150), bwInvert=False, zern=(21, 8), debug=True)
print features


hara = HaralicktextureFeatures("LAB")
hara.loadImage(imgPath=args["image"], resize_w=400)
hara.extactContours(colorSpace="LAB", channelSelect=(0,1,1), blur=7, threshold=(0,130,150), bwInvert=False, debug=True)
features = hara.describe(channelSelect=(0,0,1))
#print features
'''
