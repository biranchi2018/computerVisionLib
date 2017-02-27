import cv2
import numpy as np
import imutils
from common import Common
commlib = Common()

class Shapes:
    def __init(self):
        self.colorSpace = "RGB"

    def loadImage(self, objImage=None, imgPath=None, resize_w=None):
        if ((objImage is None) and (imgPath is None)):
            raise Exception('Please assign a value to objImage or imgPath.')
        else:
            if(objImage is None):
                objImage = cv2.imread(imgPath)

            objImage = imutils.resize(objImage, width = resize_w)
            self.image = objImage

    def extractShapes1(self, colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=False, debug=False):
        image = self.image
        shapeImages = [] 
        bwImage = commlib.binaryShapes(self.image, colorSpace, channelSelect, blur, threshold, bwInvert, debug)
        cnts = commlib.findContours(bwImage, debug=debug)
 
        if(len(cnts)>0):
            for (i, c) in enumerate(cnts):
                (x, y, w, h) = cv2.boundingRect(c)
                roi = image[y:y + h, x:x + w]
                shapeImages.append(roi)
                if(debug==True):
                    print (x, y, w, h)
                    cv2.imshow("Shape"+str(i), roi)           
        else:
            print('Warning: No shapes found for the image!')

        return shapeImages
        
