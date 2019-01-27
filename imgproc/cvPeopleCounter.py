import cv2
import imutils as iu
import numpy as np
import argparse
import random as rd
from numpy.linalg import norm


VIDEO_WIDTH = 640
GAUSS_KERNEL = (15, 15)
THRESH_MIN_THRESH = 140
THRESH_MAX_VAL = 255
DILATE_KERNEL = (75, 75)
DILATE_ITER = 4
AREA_OVER = 30
PERIMETER_OVER = 150
ASPECTRATIO_OVER = 0.4
ASPECTRATIO_BELOW = 1.9
SOLIDITY_OVER = 0.02
EXTENT_OVER = 0.4
AREA_PER_OVER = 8.00
RECT_COLOR = (255, 255, 0)
RECT_WIDTH = 2
SOLIDITY_EQ = 1.0
CIRCLE_RADIUS = 3
CIRCLE_COLOR = (0, 255, 0)
CIRCLE_FILL = -1
EUCLIDEAN_DIST_MIN = 150
TRACKER_TIMEOUT = 15 # number of frames to wait to recover
HALF_LINE_HORIZONTAL = 0
HALF_LINE_VERTICAL = 1



# basic image pre-processing operations
def imgproc(frame, model):
    frame = iu.resize(frame, width=VIDEO_WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray.copy(), GAUSS_KERNEL, 0)
    fgmask = model.apply(blur)
    thr = cv2.threshold(fgmask, THRESH_MIN_THRESH,
                        THRESH_MAX_VAL, cv2.THRESH_BINARY)[1]
    binary_frame = cv2.dilate(thr, DILATE_KERNEL, iterations=DILATE_ITER)
    cnts = cv2.findContours(binary_frame.copy(),
                            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    return (frame, binary_frame, cnts)

# argument input and video capture
def arginput():
    # argument parser for video input
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Path to the video")
    args = vars(ap.parse_args())
    # if no video use camera
    if not args.get("video", False):
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(args["video"])
    return args, cam

def finishproc():
    if cv2.waitKey(1) & 0xFF is ord("q"):
        quit()
    return

def getframe(cam):
    captured, frame = cam.read()
    if args.get("video") and not captured:
        cam.release()
        cv2.destroyAllWindows()
        quit()
    return frame

def calcAreaPerimeter(rect):
    area = cv2.contourArea(rect)
    perimeter = cv2.arcLength(rect, True)
    return area, perimeter

def calcAdvProperties(rect):
    x, y, w, h = cv2.boundingRect(rect)
    bbox = (x, y, w, h)
    aspectRatio = w / float(h)
    extent = area / float(w * h)
    hull = cv2.convexHull(rect)
    hullArea = cv2.contourArea(hull)
    solidity = area / float(hullArea)
    ap = float(area / perimeter)
    return bbox, aspectRatio, extent, solidity, ap

# calculate the midpoint of the region of interest
def roimidpoint(rect):
    M = cv2.moments(rect)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

# filter centroids (simple elimination)
def filterCentroidsAndBoxes(centroids, bboxes):
    if(len(centroids) > 1):
        for c2 in centroids:
            for i, c1 in enumerate(centroids[1:]):
                cf = c2 - c1
                L2 = float("{:.4f}".format(norm(cf)))
                if L2 < EUCLIDEAN_DIST_MIN:
                    centroids.pop(i)
                    bboxes.pop(i)
    return centroids, bboxes

# contours data
centroids = []
bboxes = []

# detection method
dfltLineMethod = HALF_LINE_HORIZONTAL

# stored object data
objid = []
objcent = []
objbbox = []
objtimeout = []
objbboxcolor = []
objtrackpts = []
objIn = 0
objOut = 0
id = 1


def colorize():
    B = rd.randint(0, 255)
    G = rd.randint(0, 255)
    R = rd.randint(0, 255)
    return (B, G, R)

# process frames
args, cam = arginput()
model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
write = False
while True:
    frame = getframe(cam)
    frame, bframe, cnts = imgproc(frame, model)
    if len(cnts):
        # print("Found {} contours".format(len(cnts)))
        for (i, rect) in enumerate(cnts):
            area, perimeter = calcAreaPerimeter(rect)
            if (area > AREA_OVER and perimeter > PERIMETER_OVER):
                bbox, aspectRatio, extent, \
                solidity, ap = calcAdvProperties(
                    rect)
                if aspectRatio < ASPECTRATIO_BELOW \
                        and aspectRatio > ASPECTRATIO_OVER \
                        and solidity > SOLIDITY_OVER \
                        and extent > EXTENT_OVER \
                        and solidity != SOLIDITY_EQ \
                        and ap > AREA_PER_OVER:

                    centroid = roimidpoint(rect)
                    centroids.append(np.array(centroid))
                    bboxes.append(np.array(bbox))

        centroids, bboxes = filterCentroidsAndBoxes(centroids, bboxes)
        for (i, to) in enumerate(objtimeout):
            objtimeout[i] = to + 1
            if objtimeout[i] > TRACKER_TIMEOUT:
                objcent.pop(i)
                objid.pop(i)
                objtimeout.pop(i)
                objbbox.pop(i)
                objbboxcolor.pop(i)

        if len(centroids): # if detected a centroid in mage
            if len(objcent): # and the table of objects has at least one centroid
                for ix in range(len(objid)): # for al centroids in table
                    c1 = objcent[ix] # get the centroid
                    for (i, c2) in enumerate(centroids): # for all detected centroids
                        L2 = float("{:.4f}".format(norm(c2 - c1))) # calculate euclidean distance
                        if L2 < EUCLIDEAN_DIST_MIN: # if is the actual object
                            objcent[ix] = c2 # update coordinates
                            objbbox[ix] = bboxes[i] # update bounding box
                            objtimeout[ix] = 0 # reset timeout
                            centroids.pop(i)
                            bboxes.pop(i)

                            if dfltLineMethod == HALF_LINE_HORIZONTAL:
                                mid_line = frame.shape[0] // 2
                                if c2[1] < mid_line and c1[1] >= mid_line:
                                    objIn +=1
                                    print("{} people in".format(objIn))
                                elif c2[1] > mid_line and c1[1] <= mid_line:
                                    objOut +=1
                                    print("{} people out".format(objOut))
                            elif dfltLineMethod == HALF_LINE_VERTICAL:
                                mid_line = frame.shape[1] // 2
                                if c2[0] < mid_line and c1[0] >= mid_line:
                                    objIn +=1
                                    print("{} people in".format(objIn))
                                elif c2[0] > mid_line and c1[0] <= mid_line:
                                    objOut +=1
                                    print("{} people out".format(objOut))

                if len(centroids): # if there are still centroids are new ones
                    objid.append(id)
                    objcent.append(centroids.pop(0))
                    objbbox.append(bboxes.pop(0))
                    objtimeout.append(0)
                    objbboxcolor.append(colorize())
                    id += 1
            else: # the table is empty, list the current detected object
                objid.append(id)
                objcent.append(centroids.pop(0))
                objbbox.append(bboxes.pop(0))
                objtimeout.append(0)
                objbboxcolor.append(colorize())
                id += 1

        for i in range(len(objcent)):
            x, y, w, h = objbbox[i]
            cx, cy = objcent[i]
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(frame, p1, p2, objbboxcolor[i], RECT_WIDTH)
            cv2.rectangle(frame, p1, (x + 40, y - 20), objbboxcolor[i], -1)
            cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, \
                       CIRCLE_COLOR, CIRCLE_FILL)
            cv2.putText(frame, "#{}".format(objid[i]), p1, \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), \
                        thickness=2, lineType=cv2.LINE_AA)

        centroids = []
        bboxes = []

    cv2.putText(frame, "People In  = {}".format(objIn), (10, 40), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), \
                thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "People Out = {}".format(objOut), (10, 70), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), \
                thickness=2, lineType=cv2.LINE_AA)
    cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (0, 0, 255))

    cv2.imshow("frame", frame)
    (h, w) = frame.shape[:2]
    if (write):
        write = False
        out = cv2.VideoWriter("out.avi", fourcc, 10, (w,  h), True)
#    out.write(frame)
    finishproc()

# release memory
cam.release()
#out.release()
cv2.destroyAllWindows()
