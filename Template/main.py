import cv2
import numpy as np

class Digit:
  def __init__(self, value, x):
    self.value = value
    self.x = x



img_rgb = cv2.imread('meterImages/meterRot2.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('meterImages/CE.jpg',0)
w, h = template.shape[::-1]

smallNuberStartX = 0

result = []

#Turn image to match CE Marking
rows,cols = img_gray.shape
for x in range(0, 35):
	M = cv2.getRotationMatrix2D((cols/2,rows/2),x*10,1)
	dst = cv2.warpAffine(img_gray,M,(cols,rows))    
	#cv2.imshow('rotaded ' + str(x),dst)

	res = cv2.matchTemplate(dst,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.7
	loc = np.where( res >= threshold)

	if len(loc[0]) > 0:
		for pt in zip(*loc[::-1]):
			cv2.rectangle(dst, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)
		cv2.imshow('rotaded ' + str(x),dst)
		img_gray = dst
		img_rgb_rot = cv2.warpAffine(img_rgb,M,(cols,rows)) 
		break

#Match large numbers

templates = ["meterImages/0.jpg","meterImages/7.jpg","meterImages/9.jpg"]
values = [0,7,9]

recStart = [10000,1000]
recEnd = [0,0]

for i in range(len(templates)):

	template = cv2.imread(templates[i],0)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where( res >= threshold)

	for pt in zip(*loc[::-1]):

		digit = Digit(values[i],pt[0])
		append = True

		for dig in result:
			if dig.value == digit.value and   dig.x - 3 <  digit.x < dig.x + 3:  
				append = False
				break
		
		if append == True:
			cv2.rectangle(img_rgb_rot, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)

			result.append(digit)

			
			if pt[0] < recStart[0]:
				recStart[0] = pt[0]
			if pt[1] < recStart[1]:
				recStart[1] = pt[1]
			if pt[0] + w > recEnd[0]:
				recEnd[0] = pt[0] + w
			if pt[1] + h > recEnd[1]:
				recEnd[1] = pt[1] + h

cv2.rectangle(img_rgb_rot, tuple(recStart), tuple(recEnd), (0,0,255), 1)
smallNuberStartX = recEnd[0]

#Match small numbers
templates = ["meterImages/2s.jpg", "meterImages/5s.jpg", "meterImages/7s.jpg"]

recStart = [10000,1000]
recEnd = [0,0]

for template in templates:

	template = cv2.imread(template,0)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where( res >= threshold)

	for pt in zip(*loc[::-1]):

		if pt[0] > smallNuberStartX:
			cv2.rectangle(img_rgb_rot, pt, (pt[0] + w, pt[1] + h), (0,255,255), 1)

			if pt[0] < recStart[0]:
				recStart[0] = pt[0]
			if pt[1] < recStart[1]:
				recStart[1] = pt[1]
			if pt[0] + w > recEnd[0]:
				recEnd[0] = pt[0] + w
			if pt[1] + h > recEnd[1]:
				recEnd[1] = pt[1] + h

cv2.rectangle(img_rgb_rot, tuple(recStart), tuple(recEnd), (0,0,255), 1)

result.sort(key=lambda x: x.x, reverse=True)

reading = 0

for i in range(len(result)):
	reading += result[i].value * 10**i

print (reading)

cv2.imshow('Original',img_rgb)
cv2.imshow('Detected',img_rgb_rot)
cv2.waitKey(0)
cv2.destroyAllWindows()


