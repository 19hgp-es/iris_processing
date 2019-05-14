

import cv2 as cv
import math
import numpy as np
import os

#input_img_path = "C:\\Users\\ANTICODE\\Documents\\Iris\\iris_data\\MMUIrisDatabase\\MMU_Iris_Database"
#output_path = "C:\\Users\\ANTICODE\\Documents\\Iris\\iris_recognition\\images\\MMUIris_norm"
input_img_path = "/home/metaljsw2/iris_processing/CASIA-IrisV4(JPG)/CASIA-Iris-Interval"
output_path = "/home/metaljsw2/CASIA_Iris_interval_norm"
iris_circle = [0, 0, 0]

def bottom_hat_median_blurr(image):

    cimg = image
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    blackhat = cv.morphologyEx(cimg, cv.MORPH_BLACKHAT, kernel)
    bottom_hat_filtered = cv.add(blackhat, cimg)
    
    return cv.medianBlur(bottom_hat_filtered, 17)

def adjust_gamma(image, gamma=1.0):
  
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
	for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)

def detect_circles(img, canny_param=20, hough_param=20):

    filtered = bottom_hat_median_blurr(img)
    adjusted = adjust_gamma(filtered, 10)
     #case mmu => min_rad = 15, max_rad = 40
    circles = cv.HoughCircles(adjusted, cv.HOUGH_GRADIENT, 1, 20,
    							param1=canny_param,
    							param2=hough_param,
    							minRadius=20,
    							maxRadius=100)

    inner_circle = [0, 0, 0]
    if circles is not None:
    	inner_circle = np.uint16(np.around(circles[0][0])).tolist()
    circle_frame = cv.circle(img, (inner_circle[0], inner_circle[1]), inner_circle[2], (0,0,0), cv.FILLED)
    
    
    #case mmu => min_rad = inner_circle[2]+20, max_rad = 100
    circles = cv.HoughCircles(adjusted, cv.HOUGH_GRADIENT, 1, 20,
								param1=canny_param,
								param2=hough_param,
								minRadius=inner_circle[2]+20,
								maxRadius=140)
    outer_circle = [0, 0, 0]

    if circles is not None:
    	for circle in circles[0]:
    		
    		
    		outer_circle = np.uint16(np.around(circle)).tolist()
    		if (abs(outer_circle[0] - inner_circle[0]) < 15) and (abs(outer_circle[1] - inner_circle[1]) < 15):
    	
    			break
    else:
    	#case mmu =>  int(inner_circle[2] * 2.4)
    	outer_circle[2] = int(inner_circle[2] * 2.4)
    outer_circle[0], outer_circle[1] = inner_circle[0], inner_circle[1]

    global iris_circle
    iris_circle = outer_circle

    return circle_frame

def detect_iris_frame(frame):
	
	global iris_circle
#for casia_dabase
	if iris_circle[0] < iris_circle[2]:
		iris_circle[2] = iris_circle[0]
	if iris_circle[1] < iris_circle[2]:
		iris_circle[2] = iris_circle[1]
	if frame.shape[0] - iris_circle[1] < iris_circle[2]:
		iris_circle[2] = frame.shape[0] - iris_circle[1]
	if frame.shape[1] - iris_circle[0] < iris_circle[2]:
		iris_circle[2] = frame.shape[1] - iris_circle[0]
#	

	mask = cv.bitwise_not(
				cv.circle(np.zeros((np.size(frame,0),np.size(frame,1),1), np.uint8)
					, (iris_circle[0], iris_circle[1]), iris_circle[2], (255,255,255), cv.FILLED))
	iris_frame = frame.copy()
	iris_frame = cv.subtract(frame, frame, iris_frame, mask, -1)
	
	return iris_frame[(iris_circle[1] - iris_circle[2]):
						(iris_circle[1] + iris_circle[2]),
						(iris_circle[0] - iris_circle[2]):
						(iris_circle[0] + iris_circle[2])]

def getPolar2CartImg(image, rad):
	
	c = (float(np.size(image, 0)/2.0), float(np.size(image, 1)/2.0))
	
	imgRes = cv.warpPolar(image, (rad*3,360), c, np.size(image, 0)/2, cv.WARP_POLAR_LOG)
	
	for valid_width in reversed(range(rad*3)):
		blank_cnt = 0
		for h in range(360):
			if (imgRes[h][valid_width] != 0):
				blank_cnt+=1
		if(blank_cnt == 0):

			
			imgRes = imgRes[0:360, valid_width:rad*3]
			break

	imgRes = cv.resize(imgRes, (80, 360), interpolation=cv.INTER_CUBIC)
	

	return (imgRes)



key = 0
print("start image processing")
for (path, dir, files) in os.walk(input_img_path):
	
#	if not(os.path.isdir(output_path+ path.split("MMU_Iris_Database")[1])):
#		os.mkdir(output_path +path.split("MMU_Iris_Database")[1])
	if not(os.path.isdir(output_path+ path.split("CASIA-Iris-Interval")[1])):
		os.mkdir(output_path +path.split("CASIA-Iris-Interval")[1])
	for filename in files:
		ext = os.path.splitext(filename)[-1]
		if ((ext == '.bmp') or (ext == '.jpg')):

			print(filename)

			frame = cv.imread(path + "/" + filename, cv.CV_8UC1)
			#cv.imshow("input", frame)
			
			circle = detect_circles(frame)
			#cv.imshow("iris",circle)
			iris = detect_iris_frame(circle)
			
			#cv.imshow("iris",iris)
			try:
				norm_frame = getPolar2CartImg(iris,iris_circle[2])
			except cv.error:
				print("cv2 error detected..")
				continue
			#print(frame.shape)
			#cv.imshow("normalized", norm_frame)
			cv.imwrite(output_path + path.split("CASIA-Iris-Interval")[1] + "/" + filename, norm_frame)

			key = cv.waitKey(1000)
			if (key == 27 or key == 1048603):
				break	

cv.destroyAllWindows()
