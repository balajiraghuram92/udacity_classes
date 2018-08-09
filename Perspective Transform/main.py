import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def convertGray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def loadimage(path):
    return mpimg.imread(path)

imgpaths = glob.glob('calibration_wide/GOPR*.jpg')

objpoints = []
imgpoints = []


for path in imgpaths:
    img = loadimage(path)
    imgobj = np.zeros((6*8,3),np.float32)
    imgobj[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    gray = convertGray(img)
    ret,corners = cv2.findChessboardCorners(gray,(8,6),None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(imgobj)
        
def cameracalib(image):
    gimage = convertGray(image)
    ret,mtx,dist,rvecs,tvercs = cv2.calibrateCamera(objpoints,imgpoints,gimage.shape[::-1],None,None)
    dstimg = cv2.undistort(image,mtx,dist,None,mtx)
    return dstimg

testimage = loadimage('calibration_wide/test_image.jpg')
undistimage = cameracalib(testimage)
plt.imshow(undistimage)
    