from ultralytics import YOLO
from os.path import exists
import cv2
import numpy as np
import os

image_path = "/home/anubhav/datasets/invoice_1/infer/images/demo.jpg"
# files = os.listdir(f"/home/anubhav/datasets/invoice_1/train/images/demo.jpg")

# print(files)

# image_path = f"{img_path}/{file}"
original_image = cv2.imread(image_path)
model = YOLO('best.pt')
bounding = None
save_dir = "/home/anubhav/datasets/invoice_1/infer/yolo_res"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
results = model.predict(source=image_path, conf=0.5, classes=[1], save_dir=save_dir)
for result in results:
    bounding = (result.boxes.xyxy)
box = bounding.cpu().numpy()

# if the boundinig box is found
if (len(box) != 0):
    x1, y1, x2, y2 = map(int, box[0])
    # with open(f"bounding_box.txt", "+a") as f:
    #     f.write(f"{"demo:"} {x1} {y1} {x2} {y2}\n")
    print(x1, y1, x2, y2)
    cropped_image = original_image[y1:y2, x1:x2]
    # if not os.path.exists('./cropped_image'):
    #     os.makedirs('./cropped_image')
    cv2.imwrite(f"/home/anubhav/datasets/invoice_1/infer/images/demo_table.jpg", cropped_image)
# if the bounding box is not found
# else:
#     with open(f"bounding_box.txt","+a") as f:
#         f.write(f"{demo:} 0 0 0 0\n")
#     cv2.imwrite(f'./cropped_image/{file}_cropped.png', original_image)