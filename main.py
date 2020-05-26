
import cv2
import numpy as np
import argparse
import time
import dlib
def load_yolo():
	net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
	classes=[]
	with open ("yolo.names","r") as f:
		classes=[line.strip() for line in f.readlines()]
	layers_names=net.getLayerNames()
	output_layers=[layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors=np.random.uniform(0,255,size=(len(classes),3))
	return net,classes,colors,output_layers
  
def load_image(img_path):
	img=cv2.imread(img_path)
	
	img=cv2.resize(img,None,fx=0.4,fy=0.4)
	height,width,chanels=img.shape
	return img,height,width,chanels
	
def detect_objects(img,net,output_layers):
	blob=cv2.dnn.blobFromImage(img,scalefactor=0.000392,size=(320,320),mean=(0,0,0),swapRB=True,crop=False)
	net.setInput(blob)
	outputs=net.forward(output_layers)
	print(len(outputs))
	return blob,outputs

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			#print(scores)
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
	
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	#print(len(indexes))
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			face=img[y:y+h, x:x+w] 
			cv2.imwrite('quoc1.jpg',face)
			cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255), 2)
			cv2.putText(img, label, (x, y - 5), font, 1,(0,255,0), 1)
			
	cv2.imshow("Image", img)  
	
	#print(face)
def image_detect(img_path):
	model,classes,colors,output_layers=load_yolo()
	image,height,width,chanels=load_image(img_path)
	img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	mask=np.zeros_like(img_gray)
	blob, outputs = detect_objects(image, model, output_layers)
	
	boxes,confs,class_ids=get_box_dimensions(outputs,height,width)
	#print('boxes',(boxes))
	predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	landmarks=predictor(image,dlib.rectangle(219,129,109,102))
	landmarks_points = []
	for n in range(2, 15):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		landmarks_points.append((x, y))
		points = np.array(landmarks_points, np.int32)
		convexhull = cv2.convexHull(points)
	#cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
		cv2.fillConvexPoly(image, convexhull, (0,0,0))
		face_image_1 = cv2.bitwise_and(image,image, mask=mask)
	#print(landmarks_points)
	cv2.imshow("quoc",image)
	draw_labels(boxes,confs,colors,class_ids,classes,image)
	
	while True:
		key=cv2.waitKey(1)
		if key==27:
			break
def start_video(video_path):
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
		
		_, frame = cap.read()
	   
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		fps = cap.get(cv2.CAP_PROP_FPS)
		
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()		   
	   
def webcam_detect():

	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(0)
	while True:
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
		
		draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()	  
	
def main():
	
	img_path="capture.png"
	
	
	image_detect(img_path)
	
	#webcam_detect()
	
	
	#start_video("quoc.mp4")
if __name__=='__main__':
	main()
	
	