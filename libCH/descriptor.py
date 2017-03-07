import cv2
import numpy as np
import imutils
import mahotas
from common import Common

commlib = Common()

class colorDescriptor:
    def __init__(self, shapesBlur=11, shapesThreshold=(120,120,120), shapesInvert=False, debug=False):
        self.image = None
        self.colorSpace = "RGB"
        self.shapesBlur = shapesBlur
        self.shapesThreshold = shapesThreshold
        self.shapesInvert = shapesInvert
        self.debug = debug

    def getImage(self):
        return self.image

    def loadImage(self, objImage=None, imgPath=None, resize_w=None):
        if ((objImage is None) and (imgPath is None)):
            raise Exception('Please assign a value to objImage or imgPath.')
        else:
            if(objImage is None):
                objImage = cv2.imread(imgPath)
				
            objImage = imutils.resize(objImage, width = resize_w)            
            self.image = objImage

    def channelFeatures1(self, colorSpace="RGB", putRGBLayers=False, onlyShapes=False, channelSelect=(1,1,1)):  #example: describe(channelSelect=(1,0,0))
        if(self.debug==True):
            cv2.imshow("Original image", self.image)
            cv2.waitKey(0)		
			
        if(onlyShapes==True):
            self.image = commlib.onlyContours(self.image, colorSpace=colorSpace, channelSelect=channelSelect, blur=self.shapesBlur, 
                threshold=self.shapesThreshold, bwInvert=self.shapesInvert, debug=self.debug)

        self.image = commlib.toColorSpace(image=self.image, colorSpace=colorSpace, putRGBLayers=putRGBLayers, channelSelect=channelSelect, debug=self.debug)

        (means, stds) = cv2.meanStdDev(self.image)
        features = np.concatenate([means, stds]).flatten()

        return features

    def histogramsFeatures1(self, colorSpace="RGB", putRGBLayers=False, onlyShapes=False, channelSelect=(1,1,1), bins=[8,8,8], range=[0, 256, 0, 256, 0, 256]):  #example: describe(channelSelect=(1,0,0))
        if(self.debug==True):
            cv2.imshow("Original image", self.image)
            cv2.waitKey(0)	

        if(onlyShapes==True):
            self.image = commlib.onlyContours(self.image, colorSpace=colorSpace, channelSelect=channelSelect, blur=self.shapesBlur,
                threshold=self.shapesThreshold, bwInvert=self.shapesInvert, debug=self.debug)
			
        self.image = commlib.toColorSpace(image=self.image, colorSpace=colorSpace, putRGBLayers=putRGBLayers, channelSelect=channelSelect, debug=self.debug)

        hist = cv2.calcHist(self.image, [channelSelect[0], channelSelect[1], channelSelect[2]], None, bins, range)
        features = cv2.normalize(hist).flatten()

        return features

# How to use:
# hu = HuMoments(momentType="Hu")  momentType = Hu or Zernike
# hu.loadImage(imgPath="zernike_reference.jpg", resize_w=400)
# features = hu.extractObject(colorSpace="LAB", channelSelect=(0,1,1), blur=11, threshold=(120,140,150), bwInvert=False, debug=True)

class momentsDescriptor:
    def __init__(self):
        self.image = None
        self.colorSpace = "RGB"

    def getImage(self):
        return self.image

    def loadImage(self, objImage=None, imgPath=None, resize_w=None):
        if ((objImage is None) and (imgPath is None)):
            raise Exception('Please assign a value to objImage or imgPath.')
        else:
            if(objImage is None):
                objImage = cv2.imread(imgPath)
				
            objImage = imutils.resize(objImage, width = resize_w)            
            self.image = objImage
    
    #Easy Hu features, read image and get Hu moments
    def HuFeatures(self):
        #image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        features = cv2.HuMoments(cv2.moments(self.image)).flatten()
        return features

    #Extract contours and get the largest to caculate the moments
    def HuFeatures1(self, colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=False, debug=True):
        bwImage = commlib.binaryShapes(image=self.image, colorSpace=colorSpace, channelSelect=channelSelect, blur=blur, threshold=threshold, bwInvert=bwInvert, debug=debug)
        cnts = commlib.findContours(bwImage, debug=debug)
        print ('cnts number:{}'.format(len(cnts)))

        if(len(cnts)>1):
            print('Warning: There are more than 1 shape in the image, only the first shape moment will be returned.')
        elif(len(cnts)<1):
            print('Warning: There are no shape in the image!')

        if(len(cnts)>0):
            (x, y, w, h) = cv2.boundingRect(cnts[0])
            print (x, y, w, h)
            roi = bwImage[y:y + h, x:x + w]

            features = cv2.HuMoments(cv2.moments(roi)).flatten()
			
        else:
            features = []
            print('Warning: No Hu-Moments Features for the image!')
			
        return features

    def ZernikeFeatures1(self, colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=False, zern=(21, 8), debug=True):
        bwImage = commlib.binaryShapes(image=self.image, colorSpace=colorSpace, channelSelect=channelSelect, blur=blur, threshold=threshold, bwInvert=bwInvert, debug=debug)
        cnts = commlib.findContours(bwImage, debug=debug)
        print ('cnts number:{}'.format(len(cnts)))

        if(len(cnts)>1):
            print('Warning: There are more than 1 shape in the image, only the first shape moment will be returned.')
        elif(len(cnts)<1):
            print('Warning: There are no shape in the image!')

        if(len(cnts)>0):
            (x, y, w, h) = cv2.boundingRect(cnts[0])
            print (x, y, w, h)
            roi = bwImage[y:y + h, x:x + w]

            features = mahotas.features.zernike_moments(bwImage, zern[0], degree=zern[1])
        else:
            features = []
            print('Warning: No Zernike-Moments Features for the image!')
			
        return features

#Usage:
#hara = HaralicktextureFeatures("RGB")
#hara.loadImage(imgPath="zernike_reference.jpg", resize_w=400)
#hara.extactContours(colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=False, debug=True)
#features = hara.describe(channelSelect=(0,0,1))

class TextureFeatures:
    def __init__(self):
        self.image = None
        self.colorSpace = "RGB"

    def getImage(self):
        return self.image

    def loadImage(self, objImage=None, imgPath=None, resize_w=None):
        if ((objImage is None) and (imgPath is None)):
            raise Exception('Please assign a value to objImage or imgPath.')
        else:
            if(objImage is None):
                objImage = cv2.imread(imgPath)

            objImage = imutils.resize(objImage, width = resize_w)
            self.image = objImage

    def HaralickFeatures1(self, colorSpace="RGB", channelSelect=(1,1,1), debug=True):  #example: describe(channelSelect=(1,0,0))
        self.image = commlib.toColorSpace(image=self.image, colorSpace=colorSpace, channelSelect=channelSelect, debug=debug)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # extract Haralick texture features in 4 directions, then take the
        # mean of each direction
        features = mahotas.features.haralick(gray).mean(axis=0)
        return features

class imgDifference:
    def __init__(self):
        self.image1 = None
        self.image2 = None
        self.colorSpace = "RGB"

    def getImage(self):
        return self.image

    def loadImage(self, objImage1=None, imgPath1=None, objImage2=None, imgPath2=None, resize_w=None):
        if (((objImage1 is None) and (imgPath1 is None)) or ((objImage2 is None) and (imgPath2 is None))):
            raise Exception('Please assign a value to objImage or imgPath.')

        else:
            if(objImage1 is None):
                objImage1 = cv2.imread(imgPath1)
            if(objImage2 is None):
                objImage2 = cv2.imread(imgPath2)

            objImage1 = imutils.resize(objImage1, width = resize_w)
            objImage2 = imutils.resize(objImage2, width = resize_w)

            self.image1 = objImage1
            self.image2 = objImage2

    def channelDifference(self, colorSpace="RGB", channelSelect=(1,1,1)):
        if(len(channelSelect)) != 3:
            raise Exception('channelSelect parameter must be (1,1,1) format. (0 means unselect, 1 means selected')
        if(channelSelect[0]>1 or channelSelect[1]>1 or channelSelect[2]>1) or (channelSelect[0]<0 or channelSelect[1]<0 or channelSelect[2]<0):
            raise Exception('ChannelSelect value only be 0 or 1 (0-> unselected, 1-> selected)')

        if(colorSpace == "LAB"):
            imgTransform1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2LAB)
            imgTransform2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2LAB)
        elif(colorSpace == "HSV"):
            imgTransform1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2HSV)
            imgTransform2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2HSV)
        elif(colorSpace == "HSL"):
            imgTransform1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2HLS)
            imgTransform2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2HLS)
        elif(colorSpace == "LUV"):
            imgTransform1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2LUV)
            imgTransform2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2LUV)
        else:
            imgTransform1 = self.image1
            imgTransform2 = self.image2


        img1_Channels = cv2.split(imgTransform1)
        img2_Channels = cv2.split(imgTransform2)
        diffChannel = []
        print(len(img1_Channels))
        print(len(img2_Channels))

        for (i,channelNum) in enumerate(channelSelect):
            print ("i={}, channelNum={}".format(i, channelNum))
            if(channelNum == 1):
                diffChannel.append(cv2.bitwise_xor(img1_Channels[i], img2_Channels[i]))
                cv2.imshow("Channel"+str(i), diffChannel[i])
                cv2.waitKey(0)
