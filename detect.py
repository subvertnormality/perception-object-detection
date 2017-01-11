import numpy as np
import cv2
from matplotlib import pyplot as plt
import copy
import time
import requests


cert_file_path = "certs/client.crt"
key_file_path = "certs/client.key"
cert = (cert_file_path, key_file_path)

url = 'https://secure.playperception.com/game/playround/'
url_local = 'https://secure.localhost/game/playround/'

MIN_MATCH_COUNT = 4

q1 = cv2.imread('images/1.png',0)
q2 = cv2.imread('images/2.png',0)
q3 = cv2.imread('images/3.png',0)
q1l = cv2.imread('images/1l.png',0)
q2l = cv2.imread('images/2l.png',0)
q3l = cv2.imread('images/3l.png',0)
q1r = cv2.imread('images/1r.png',0)
q2r = cv2.imread('images/2r.png',0)
q3r = cv2.imread('images/3r.png',0)

truth = np.array([[[True,True],[True,False],[False,False],[False,True]],[[True,True],[True,False],[False,False],[False,True]],[[True,True],[True,False],[False,False],[False,True]],[[True,True],[True,False],[False,False],[False,True]]])
one_bounds = np.array([[635, 0], [635, 163], [890, 163], [890, 0]])
two_bounds = np.array([[890, 0], [890, 163], [1190, 163], [1190, 0]])
three_bounds = np.array([[1190, 0], [1190, 163], [1445, 163], [1445, 0]])
four_bounds = np.array([[610, 700], [610, 1080], [920, 1080], [920, 700]])
five_bounds = np.array([[920, 700], [920, 1080], [1250, 1080], [1250, 700]])
six_bounds = np.array([[1250, 700], [1250, 1080], [1650, 1080], [1650, 700]])

bounds = [one_bounds, two_bounds, three_bounds, four_bounds, five_bounds, six_bounds]

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
            try:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = o.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                sceneTransform = copy.deepcopy(scene)
            except:
                print('Error finding match')
        else:
            # print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
        if (dst.size == 0):
            continue
        else:
            return dst

    return np.array([])




while True:
    try:
        cap = cv2.VideoCapture(0)
        cap.set(3,1920)
        cap.set(4,1080)
        ret, frame = cap.read()
        cv2.imwrite("frame.jpg", frame)

        # Display the resulting frame

        q1dst = np.array([])
        q1dst = do_transform(q1, frame)
        if (q1dst.size == 0):
            q1dst = do_transform(q1l, frame)
        if (q1dst.size == 0):
            q1dst = do_transform(q1r, frame)
        if (q1dst.size == 0):
            print('skipping because of 1')
            continue
        q2dst = np.array([])
        q2dst = do_transform(q2, frame)
        if (q2dst.size == 0):
            q2dst = do_transform(q2l, frame)
        if (q2dst.size == 0):
            q2dst = do_transform(q2r, frame)
        if (q2dst.size == 0):
            print('skipping because of 2')
            continue
        q3dst = np.array([])
        q3dst = do_transform(q3, frame)
        if (q3dst.size == 0):
            q3dst = do_transform(q3l, frame)
        if (q3dst.size == 0):
            q3dst = do_transform(q3r, frame)
        if (q3dst.size == 0):
            print('skipping because of 3')
            continue
        dsts = [q1dst, q2dst, q3dst]


        result = []

        for bound in bounds:
            found = False
            block = 1
            for dst in dsts:
                try:
                    if np.equal(np.greater_equal(dst, bound), truth).all():
                        result.append(block)
                        found = True
                except:
                    print('No valid bounds found') 
                block = block + 1
            if not found:
                result.append(0)

        print(result)

        if (1 in result and 2 in result and 3 in result and len(result) == 6):
            print('posting')
            r = requests.post(url, json={'attempt': result}, cert=cert, verify=False)
        else:
            print('ígnoring')
    except Exception as e:
        print(e)
    time.sleep(1)