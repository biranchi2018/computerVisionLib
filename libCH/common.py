import cv2
import numpy as np
import imutils

class Common:
    def __init__(self):
        self.colorSpace = "RGB"

    def colorRatio(self, image=None, color=(255,255,255), sizeRatio=1):
        if(image is None):
            raise Exception('Error: No image.')

        image = imutils.resize(image, width = (image.shape[1] * sizeRatio))
        width = image.shape[1]
        height = image.shape[0]
        sameColor = 0.0

        for pixel_w in range(0, width, 1):
            for pixel_h in range(0, height, 1):
                (b, g, r) = image[pixel_h, pixel_w]
                if(b==color[0] and g==color[1] and r==color[2]):
                    sameColor += 1

        plantArea = greenArea / (width*height)
        return int(plantArea * 100)

    def displayImage(self, image=None, title="Image Display", sizeRatio=1):
        if (not image is None):
           image = imutils.resize(image, width = (image.shape[1] * sizeRatio))
           cv2.imshow(title, image)

    def waitWindowClose(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def findContours(self, image, debug=False):
        # detect contours in the edge map
        if(debug==True):
            cv2.imshow("Original-Contours", image)

        (cnts, _) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if(debug==True):
            for (i, c) in enumerate(cnts):
                orig = drawContours(image, c, i)

            cv2.imshow("Contours find", orig)
            waitWindowClose()

        return cnts

    def drawContours(self, image, cnts):
        for (i, c) in enumerate(cnts):
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            # compute the center of the contour area and draw a circle
            # representing the center
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"]+1))
            cY = int(M["m01"] / (M["m00"]+1))

            # draw the countour number on the image
            cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2)

            # return the image with the contour number drawn on it
            return image

    def toColorSpace(self, image, colorSpace="RGB", putRGBLayers=False, channelSelect=(1,1,1), debug=False):  #example: describe(channelSelect=(1,0,0))
        if(image is None):
            raise Exception('No image object!')

        else:
            if(len(channelSelect)) != 3:
                raise Exception('channelSelect parameter must be (1,1,1) format. (0 means unselect, 1 means selected')
            if(channelSelect[0]>1 or channelSelect[1]>1 or channelSelect[2]>1) or (channelSelect[0]<0 or channelSelect[1]<0 or channelSelect[2]<0):
                raise Exception('ChannelSelect value only be 0 or 1 (0-> unselected, 1-> selected)')

            if(colorSpace == "LAB"):
                imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif(colorSpace == "HSV"):
                imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif(colorSpace == "HSL"):
                imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif(colorSpace == "LUV"):
                imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            else:
                imgTransform = image
          
            zeros = np.zeros(image.shape[:2], dtype = "uint8")
            imgChannels = cv2.split(imgTransform)
            arrayChannel = []

            for (i,selected) in enumerate(channelSelect):
                if(selected==1):
                    arrayChannel.append(imgChannels[i])
                    if(debug==True):
                        print('Channel {}'.format(i))
                        if(colorSpace=="RGB" or putRGBLayers==True):
                            if(i==0): rgbChannelDisplay = cv2.merge([imgChannels[i], zeros, zeros])
                            if(i==1): rgbChannelDisplay = cv2.merge([zeros, imgChannels[i], zeros])
                            if(i==2): rgbChannelDisplay = cv2.merge([zeros, zeros, imgChannels[i]])
                        else:
                            rgbChannelDisplay = imgChannels[i]

                        cv2.imshow(colorSpace + " channel-"+str(i), rgbChannelDisplay)

                else:
                    arrayChannel.append(zeros)

            img = cv2.merge(arrayChannel)

            if(debug==True):
                cv2.imshow(colorSpace+" ColorSpace", img)
                waitWindowClose()

            return img


    #Parameters: 
    #Image: RGB image obj
    #colorSpace: RGB/LAB/HSV
    #channelSelect: Color-space channels, 0--> don't select, 1--> select, format is 3 dimentions
    #blur: blur size
    #threshold: threshold intensity for each channel
    #bwInvert: Black & Whilte (0 & 1) invert for the image
    def binaryShapes(self, image, colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=False, debug=False):
        if(debug==True):
            cv2.imshow("Original", image)
		
        accumLayers = np.zeros(image.shape[:2], dtype = "uint8")
        zeros = np.zeros(image.shape[:2], dtype = "uint8")

        if(colorSpace == "LAB"):
            imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif(colorSpace == "HSV"):
            imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif(colorSpace == "HSL"):
            imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif(colorSpace == "LUV"):
            imgTransform = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        else:
            imgTransform = image

        i=0
        for chan in cv2.split(imgTransform):
            print('i:{} selected:{}'.format(i,channelSelect[i]))
            if(channelSelect[i]==1):
                if(debug==True):
                    cv2.imshow("Channel", chan)

                chan = cv2.medianBlur(chan, blur)
                if(bwInvert==True):
                    (T, threshed) = cv2.threshold(chan, threshold[i], 255, cv2.THRESH_BINARY_INV)
                else:
                    (T, threshed) = cv2.threshold(chan, threshold[i], 255, cv2.THRESH_BINARY)

                cv2.imshow("Threshold-"+str(i), threshed)

                #edged = cv2.Canny(threshed, canny[0], canny[1])
                accumLayers = cv2.bitwise_or(accumLayers, threshed)

            i += 1

        if(debug==True):
            cv2.imshow("Edge Map", accumLayers)
            cv2.waitKey(0)

        return accumLayers

	
    def onlyContours(self, orgImage, colorSpace="RGB", channelSelect=(1,1,1), blur=11, threshold=(50,50,50), bwInvert=False, debug=False):
        mask = self.binaryShapes(orgImage, colorSpace, channelSelect, blur, threshold, bwInvert, debug)

        if(debug==True):
            print(orgImage.shape)
            print(mask.shape)

        bitwiseAnd = cv2.bitwise_and(orgImage, orgImage, mask=mask)
        if(debug==True):
            displayImage(title="Original", image=orgImage)
            displayImage(title="Mask", image=mask)
            displayImage(title="Masked", image=bitwiseAnd)
            waitWindowClose()	
			
        return bitwiseAnd        
