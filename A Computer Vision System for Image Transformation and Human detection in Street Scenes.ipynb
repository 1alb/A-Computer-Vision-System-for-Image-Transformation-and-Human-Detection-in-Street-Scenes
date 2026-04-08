!pip install opencv-python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread('/content/download.jpg')
type(image)
np.ndarray
image.shape
plt.imshow(image)
new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(new_image)
plt.imshow(new_image)
"""
1.    Splitting image channels
"""
r,g,b =cv2.split(new_image)
"""print('r', r.shape)
print('g', g.shape)
print('b', b.shape)
"""
new_image = cv2.merge((r,g,b))
"""
Resize of images
"""

s = 10
w = int(new_image.shape[1]*s/100)
h = int(new_image.shape[0]*s/100)
dim = (w,h)
re_size = cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)
re_size.shape
""""
3. rotate operation
"""

(h,w) = new_image.shape[:2]

c = (w/2 , h/2)

angle = 90

m = cv2.getRotationMatrix2D(c, angle, 1.0)
rotate_90 = cv2.warpAffine(new_image, m, (h,w))
plt.imshow(rotate_90)
yolo = cv2.dnn.readNet("/content/yolov3-tiny.weights","/content/yolov3-tiny.cfg" )
classes = []
with open("/content/coco.names", 'r') as f:
  classes = f.read().splitlines()
len(classes)
img = cv2.imread("/content/download.jpg")
blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop = False)
blob.shape
# to print img
i = blob[0].reshape(320,320,3)
plt.imshow(i)
yolo.setInput(blob)
output_layes_names = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layes_names)
boxes = []
confidences = []
class_ids = []

height, width, channels = image.shape   # ✅ FIX ADDED

for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]

        if confidence > 0.7:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)   # ❗ you had detection[0] twice
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
len(boxes)
value = min(len(boxes), 2)
print(value)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))
for i in indexes.flatten():
    x, y, w, h = boxes[i]

    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i], 2))
    color = colors[i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    cv2.putText(img, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 2)
plt.imshow(img)
cv2.imwrite("/content/download.jpg", img)
import cv2

# Load image
img = cv2.imread("/content/download2.jpg")

# Resize for better detection speed
img = cv2.resize(img, (800, 600))

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect people
boxes, weights = hog.detectMultiScale(
    img,
    winStride=(8,8),
    padding=(8,8),
    scale=1.05
)

# Draw rectangles
for (x, y, w, h) in boxes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show result (Colab)
from google.colab.patches import cv2_imshow
cv2_imshow(img)

