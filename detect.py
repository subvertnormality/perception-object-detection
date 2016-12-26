import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy

MIN_MATCH_COUNT = 6

q1 = cv2.imread('images/1.png',0)
q2 = cv2.imread('images/2.png',0)
q3 = cv2.imread('images/3.png',0)
q1l = cv2.imread('images/1l.png',0)
q2l = cv2.imread('images/2l.png',0)
q3l = cv2.imread('images/3l.png',0)
q1r = cv2.imread('images/1r.png',0)
q2r = cv2.imread('images/2r.png',0)
q3r = cv2.imread('images/3r.png',0)
scene = cv2.imread('scene2r.png',0) # trainImage


def do_transform(ref, scene):

    objects = [ref, cv2.flip(ref,0), cv2.flip(ref,1), cv2.flip(cv2.flip(ref,0),1)]

    for o in objects:

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(o,None)
        kp2, des2 = sift.detectAndCompute(scene,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        
        dst = np.array([])
        
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = o.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            sceneTransform = copy.deepcopy(scene)

        else:
            # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
        if (dst.size == 0):
            continue
        else:
            return dst
    return np.array([])

q1dst = np.array([])
q1dst = do_transform(q1, scene)
if (q1dst.size == 0):
    q1dst = do_transform(q1l, scene)
if (q1dst.size == 0):
    q1dst = do_transform(q1r, scene)
q2dst = np.array([])
q2dst = do_transform(q2, scene)
if (q2dst.size == 0):
    q2dst = do_transform(q2l, scene)
if (q2dst.size == 0):
    q2dst = do_transform(q2r, scene)
q3dst = np.array([])
q3dst = do_transform(q3, scene)
if (q3dst.size == 0):
    q3dst = do_transform(q3l, scene)
if (q3dst.size == 0):
    q3dst = do_transform(q3r, scene)


truth = np.array([[[True,True],[True,False],[False,False],[False,True]],[[True,True],[True,False],[False,False],[False,True]],[[True,True],[True,False],[False,False],[False,True]],[[True,True],[True,False],[False,False],[False,True]]])
one_bounds = np.array([[459, 378], [459, 444], [604, 444], [604, 378]])
two_bounds = np.array([[459, 378], [459, 444], [604, 444], [604, 378]])
three_bounds = np.array([[459, 378], [459, 444], [604, 444], [604, 378]])
four_bounds = np.array([[459, 378], [459, 444], [604, 444], [604, 378]])
five_bounds = np.array([[459, 378], [459, 444], [604, 444], [604, 378]])
six_bounds = np.array([[459, 378], [459, 444], [604, 444], [604, 378]])


dsts = [q1dst, q2dst, q3dst]
bounds = [one_bounds, two_bounds, three_bounds, four_bounds, five_bounds, six_bounds]

result = []

for bound in bounds:
    found = False
    block = 1
    for dst in dsts:
        if np.equal(np.greater_equal(dst, bound), truth).all():
            result.append(block)
            found = True
        block = block + 1
    if not found:
        result.append(0)
        
print(result)

# print(np.equal(np.greater_equal(q3dst, one_bounds), truth).all())
plt.imshow(cv2.polylines(scene,[np.int32(one_bounds)],True,255,3, cv2.LINE_AA), 'gray'),plt.show()