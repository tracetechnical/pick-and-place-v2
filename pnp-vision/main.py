import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt


import cv2
import requests
import numpy as np
template = cv.imread('C:\\Users\\Admin\\Downloads\\Fid.png')
h, w = template.shape[:2]
cap = cv.VideoCapture('http://192.168.30.189:8000/stream.mjpg')
a = np.empty([40,2])
b = np.empty([40,2])
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # img_rgb = frame[300:800,205:600]
    img_rgb = frame
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    # img_gray2 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
    ret, edge = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
    edge=cv.bitwise_not(edge)
    edge_c = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    edges = cv.Canny(edge,100,200)

    k = 0
    avg = 0
    lines = cv.HoughLines(edges, 1, np.pi / 1800, 400)
    if lines is not None:
        for line in lines:
            k += 1
            rho, theta = line[0]
            ga = np.cos(theta)
            gb = np.sin(theta)
            x0 = ga * rho
            y0 = gb * rho
            x1 = int(x0 + 1000 * (-gb))
            y1 = int(y0 + 1000 * (ga))
            x2 = int(x0 - 1000 * (-gb))
            y2 = int(y0 - 1000 * (ga))
            deg = theta * 57.2958
            if deg > 150:
                t2 = 180 - deg
            else:
                t2 = deg
            avg += t2
            cv.putText(edge_c, str(t2), (20, 20*k), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.line(edge_c, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.putText(edge_c, str(avg/k), (20, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    else:
        cv.putText(edge_c, ':(', (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('edges', edges)
    cv.imshow('edge', edge_c)

    res = cv.matchTemplate(img_rgb, template, cv.TM_CCOEFF_NORMED)

    threshold = 0.85
    loc = np.where( res >= threshold)

    tree_count = 0
    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    i = 0

    cv2.namedWindow('dc1')  # Create a named window
    cv2.namedWindow('dc2')  # Create a named window
    cv2.namedWindow('dc3')  # Create a named window
    cv2.namedWindow('dc4')  # Create a named window

    cv2.namedWindow('cc1')  # Create a named window
    cv2.namedWindow('cc2')  # Create a named window
    cv2.namedWindow('cc3')  # Create a named window
    cv2.namedWindow('cc4')  # Create a named window

    cv2.moveWindow('dc1', 40, 100)  # Move it to (40,30)
    cv2.moveWindow('cc1', 150, 100)  # Move it to (40,30)
    cv2.moveWindow('dc2', 40, 200)  # Move it to (40,30)
    cv2.moveWindow('cc2', 150, 200)  # Move it to (40,30)
    cv2.moveWindow('dc3', 40, 300)  # Move it to (40,30)
    cv2.moveWindow('cc3', 150, 300)  # Move it to (40,30)
    cv2.moveWindow('dc4', 40, 400)  # Move it to (40,30)
    cv2.moveWindow('cc4', 150, 400)  # Move it to (40,30)

    for pt in zip(*loc[::-1]):
        if mask[pt[1] + int(round(h/2)), pt[0] + int(round(w/2))] != 255:
            mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
            tree_count += 1
            i += 1
            cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
            crop = img_rgb[pt[1]:pt[1] + h, pt[0]:pt[0] + w]
            cc = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            cv.imshow('cc'+str(i), cc)
            #

            ret, thresh1 = cv.threshold(cc, 130, 255, cv.THRESH_BINARY)
            thresh1 = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
            img = cv2.GaussianBlur(thresh1, (0,0), sigmaX=1, sigmaY=1, borderType = cv2.BORDER_DEFAULT)
            cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20,
                                      param1=50, param2=25, minRadius=0, maxRadius=0) #p2 20
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for k in circles[0, :]:
                    cv.circle(cimg, (k[0], k[1]), k[2], (0, 255, 0), 1)
                    cv.drawMarker(cimg, (k[0], k[1]), color=(0, 255, 0), markerType=cv.MARKER_CROSS, thickness=1)
                    x = k[0]
                    y = k[1]
                    radius = k[2]
                    cv.drawMarker(img_rgb, (pt[0] + int(x), pt[1] + int(y)), color=(255, 0, 255), markerType=cv.MARKER_CROSS, thickness=1)
                    cv.circle(img_rgb, (pt[0] + int(x), pt[1] + int(y)), radius, (255, 0, 255), 1)

                    a[i][0] = pt[0] + int(x)
                    a[i][1] = pt[1] + int(y)
                    xa=a[2][0]-a[1][0]
                    xb=-22* 26.63
                    ya=a[2][1]-a[1][0]
                    yb=0.3* 26.63
                    delta_x = xa - xb
                    delta_y = ya - yb
                    theta_radians = math.degrees(math.atan2(delta_y, delta_x))
                    cv.putText(img_rgb, str(round(((a[i][0]-a[1][0])/26.63),2)),  (pt[0] + int(x), pt[1] + int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    cv.putText(img_rgb, str(round(((a[i][1]-a[1][1])/26.63),2)), (pt[0] + int(x), pt[1] + int(y) + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                    cv.putText(img_rgb,'d:'+str(round(theta_radians, 2)), (pt[0] + int(x), pt[1] + int(y) + 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            cv.imshow('dc'+str(i), cimg)

            #
            # cc = cv.bitwise_not(cc)
            # # cv.imshow('A' + str(i), cc)
            # cc2 = cv.cvtColor(thresh1, cv.COLOR_GRAY2BGR)
            # kernel = np.ones((2,2),np.uint8)
            # thresh1 = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
            # thresh1 = cv.morphologyEx(thresh1, cv.MORPH_OPEN, kernel)
        REF=26.63 #/21.2
        NDP=2
        x1=round(a[1][0]/REF,NDP)
        x2=round(a[2][0]/REF,NDP)
        y1=round(a[1][1]/REF,NDP)
        y2=round(a[2][1]/REF,NDP)
        myradians = math.atan2(y2 - y1, x2 - x1)
        mydegrees = math.degrees(myradians)
        # cv.putText(img_rgb, str(round(a[1][0]/REF,NDP)) + ',' + str(round(a[1][1]/REF,NDP)), (20, 130), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        # cv.putText(img_rgb, str(round(a[2][0]/REF,NDP)) + ',' + str(round(a[2][1]/REF,NDP)), (20, 160), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        # cv.putText(img_rgb, str(round((a[2][0]-a[1][0])/REF,NDP)) + ',' + str(round((a[2][1]-a[1][1])/REF,NDP)), (20, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        # cv.putText(img_rgb, str(mydegrees) + 'degs', (20, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        # circles = np.uint16(np.around(circles))
            # for i in circles[0, :]:
            #     # draw the outer circle
            #     cv.circle(imagem, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #     # draw the center of the circle
            #     cv.circle(imagem, (i[0], i[1]), 2, (0, 0, 255), 3)
            # cv.imshow(str(i), imagem)

    cv.imshow('detected circles', img_rgb)
    # print("Found {} trees in total!".format(tree_count))

    cv.waitKey(1)