import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

MIN_MATCH_COUNT = 6

q1 = cv2.imread('images/1.png',0)
q2 = cv2.imread('images/2.png',0)
q3 = cv2.imread('images/3.png',0)
scene = cv2.imread('scene.png',0) # trainImage

def do_transform(q1, scene):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(q1,None)
    kp2, des2 = sift.detectAndCompute(scene,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    
    dst = []
    
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = q1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        sceneTransform = copy.deepcopy(scene)

        # plt.imshow(cv2.polylines(sceneTransform,[np.int32(dst)],True,255,3, cv2.LINE_AA), 'gray'),plt.show()
        
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    return dst

q1dst = do_transform(q1, scene)
q2dst = do_transform(q2, scene)
q3dst = do_transform(q3, scene)