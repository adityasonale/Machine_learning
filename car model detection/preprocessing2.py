import cv2
from ultralytics import YOLO
import numpy as np 
import os 
from tensorflow import keras
import shutil
from PIL import Image

# Here I have used YOLOv8 model for detecting car from the image

model = YOLO("D:\\vs code\python\DeepLearning\Projects\car\yolo_weights\yolov8n.pt")

#------------------------------------------------------------------------------------------------------------------------------------------

def create_folder(x):
    if os.path.exists(x):
        print("Folder Already Exists !!!")

    else:
        os.mkdir(x)
        print("{x} created Successfully !!")

#-------------------------------------------------------------------------------------------------------------------------------------------

def crop_image(input_image, coordinates, output_path):
    img = Image.open(input_image)

    # Crop the image using the provided coordinates
    cropped_img = img.crop(coordinates)
    if cropped_img.mode == 'RGBA' or cropped_img.mode == 'P':
        cropped_img = cropped_img.convert("RGB")

    # Save the cropped image
    cropped_img.save(output_path)

#-----------------------------------------------------------------------------------------------------------------------------------------

def calc_area(box):
    max_ar = 0
    for j,i in enumerate(box):
        b = int(i[2]) - int(i[0])
        l = int(i[3]) - int(i[1])
        if b*l > max_ar:
            max_ar = b*l
            x_min,y_min,x_max,y_max = int(box[j][0]),int(box[j][1]),int(box[j][2]),int(box[j][3])

    return  x_min,y_min,x_max,y_max

#-----------------------------------------------------------------------------------------------------------------------------------------

directories = []

new_path = "D:\Datasets\cars_new\\test"
crop_path = "E:\cars_cropped\\test_cropped"

for entry in os.listdir(new_path):
    entry_path = os.path.join(new_path,entry)
    if os.path.isdir(entry_path):
        directories.append(entry_path.split("\\")[-1])


try:

    for folder in directories:
        cropped_path = os.path.join(crop_path,folder)
        create_folder(cropped_path)
        source_path = os.path.join(new_path,folder)
        for file in os.listdir(source_path):
            if file.lower().endswith((('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'))):
                img_path = os.path.join(source_path,file)
                try:
                    result = model(img_path)
                    for i in result:
                        x1,y1,x2,y2 = calc_area(i.boxes.xyxy)
                        x1,y1,x2,y2 = int(x1), int(x2), int(y1), int(y2)

                    destination_path = os.path.join(crop_path,folder)
                    destination_path = os.path.join(destination_path,file)
                    rectangular_coordinates = (x1,x2,y1,y2)

                    crop_image(img_path,rectangular_coordinates,destination_path)
                except Exception as e:
                    print(e)

except Exception as e:
    print(e)



'''
YOLOv8 was detecting everything that was present in the image like person, traffic light etc. So what i did was that i collected the coordinates
all the detected objects, and found the areas for each of the object, since my car was the mainly shown in the image it would have the maximum area.
After detecting the area i used the coordinates of cars to crop the region in which my car was present.
'''