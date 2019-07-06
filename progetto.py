import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#casa13 = cv2.imread("casa13.jpg")
#casa14 = cv2.imread("casa14.jpg")
#casa15 = cv2.imread("casa15.jpg")
boat1 = cv2.imread("boat1.jpg")
boat2 = cv2.imread("boat2.jpg")
boat3 = cv2.imread("boat3.jpg")
boat4 = cv2.imread("boat4.jpg")
#im1=cv2.imread("im1.jpg")
#im2=cv2.imread("im2.jpg")
#im3=cv2.imread("im3.jpg")

images=[]

#images.append(casa13)
#images.append(casa14)
#images.append(casa15)
images.append(boat1)
images.append(boat2)
images.append(boat3)
images.append(boat4)
#images.append(im1)
#images.append(im2)
#images.append(im3)

result=images[0]

for x in range(len(images)-1):
  images[x]=result
  grayImage1 = cv2.cvtColor(images[x], cv2.COLOR_BGR2GRAY)
  grayImage2 = cv2.cvtColor(images[x+1], cv2.COLOR_BGR2GRAY)
  

  sift = cv2.xfeatures2d.SIFT_create()
  keyp1, desc1 = sift.detectAndCompute(grayImage1, None)
  keyp2, desc2 = sift.detectAndCompute(grayImage2, None)
  keyImage1 = cv2.drawKeypoints(grayImage1, keyp1, np.array([]), (255, 0, 0))
  keyImage2 = cv2.drawKeypoints(grayImage2, keyp2, np.array([]), (255, 0, 0))

  brutef = cv2.BFMatcher()
  matches = brutef.knnMatch(desc1, desc2, k=2)

  goodMatch = []
  for m, n in matches:
	  if m.distance < 0.7*n.distance:
		  goodMatch.append(m)

  matchImage = cv2.drawMatches(images[x], keyp1, images[x+1], keyp2, goodMatch, np.array([]), (255, 0, 0), flags=2)

  cv2.imwrite('img_matches_knn'+str(x)+'.jpg', matchImage)

  
  srce_pts = np.float32([ keyp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
  dest_pts = np.float32([ keyp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
  homographyMat, mask = cv2.findHomography(srce_pts, dest_pts, cv2.RANSAC, 5.0)
  matchesMask = mask.ravel().tolist()
    
  draw_params = dict(matchColor = (255,0,0), 
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)

  imgMatch = cv2.drawMatches(images[x],keyp1,images[x+1],keyp2,goodMatch,None,**draw_params)
  cv2.imwrite('img_matches'+str(x)+'.jpg', imgMatch)
  
  h1, w1 = images[x+1].shape[:2]
  h2, w2 = images[x].shape[:2]
  pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
  pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
  pts2_ = cv2.perspectiveTransform(pts2, homographyMat)
  pts = np.concatenate((pts1, pts2_), axis=0)
  
  [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
  [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
  t = [-xmin, -ymin]
  Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
  
  result = cv2.warpPerspective(images[x], Ht.dot(homographyMat), (xmax-xmin, ymax-ymin))
  result[t[1]:h1+t[1], t[0]:w1+t[0]] = images[x+1]
 
  cv2.imwrite('img_pano'+str(x)+'.jpg', result)
