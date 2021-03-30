# Object detection with Yolov5

## Basics for Object Detection
Object detection involves classifying localized bounding boxes in the image, that is classifying the objects and searching on the position for the bounding box. Through object detection mechanisms and algorithms, we are able to understand what's in an image, while being able to describe both what is in an image and the locations of those objects in the image.
<br>
On a predefined set of class labels (e.g. people and cars), object detection helps us describe the locations of each detected object in the image using a bounding box. The training data for object detection models include images and corresponding bounding box coordinates. 

---

## Input data format types for object detection tasks: VOC and MS-COCO
Input data for Pascal VOC is an **XML file**, whereas COCO dataset uses a **JSON file**.

Bounding boxes in VOC and COCO challenges are differently represented and they are as follows:
1. PASCAL VOC: (xmin-top left, ymin-top left,xmax-bottom right, ymax-bottom right)
2. COCO: (x-top left, y-top left, width, height)

The JSON file for MS COCO contains the images and their corresponding labels.
Sample image and annotations:
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/coco_data.png?token=AGCG5WBV27UP34ALLJGRSN3AMLNCS)
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/coco_data_format.png?token=AGCG5WEKPOCZ3HZZLAD7L33AMLNDY)
Example PASCAL VOC bounding box<br>
The format of bounding box in the provided json is specified by upper left coordinates and respective dimensions (width, height) of each box.<br>
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/voc_data.png?token=AGCG5WB43U72YUCSBTCCCITAMLNF6)

---
# Detection Mechanisms:
 
One stage detector algorithms such as models belonging to the SSD, YOLO classes of models and RetinaNet (as opposed to two stage detectors for instance: RCNN, Fast RCNN or Faster RCNN which have a region proposal network which could be learnable) do not have an intermediate stage which must be performed to produce the output, leading to accelerated training and inference.


## Two State Detection:
Framework for two stage object detection networks (RCNN, Fast RCNN or Faster RCNN) is as follows:
1. Extracting regions of interest which are then warped to a fix size image, as is required by the CNN (with or without the RoI pooling layer)
2. Feature extraction is extracted by running a pretrained convolutional network on top of the region proposals.
3. A classifier such as SVM makes classification decisions based on extracted features.
4. Bounding box regression to predict location and size of the bounding box surrounding the object (using coordinates for box origin with dimensions of the bounding boxes)
<br>
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/two_stage_detectors.JPG?token=AGCG5WC6QQ6LUIZRMQKAMR3AMLNI2)
<br>
### Issues with the two stage detectors:
1. Slow inference and expensive training policy
2. Multi-stage pipeline training (CNN, classifier and regressor)

## Single State Detection
In single stage detectors, the convolutional layers make predictions in one shot, with the approach being based on a feed forward network that creates a fix sized collection of bounding boxes, the objectness of each bounding box being then predicted by the logic regression as indicative of the level of overlap with the ground truth.<br>
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/one_stage.JPG?token=AGCG5WB2PV3752FKLOBLFVDAMLNKQ)

### SSD:
Working of the Single Shot Multibox Detection networks is based on these components:
1. Feature extractor convolutional network
2. Multi scaled feature layers which decrease in size progressively to allow for prediction of detections at multiple scales.
3. Non Maximum Suppression for elimination of overlapping bounding boxes, keeping only one box per each object detected.
4. In an effort to reduce objects belonging to the background class, we use hard negative mining which helps filter out anchor boxes that do not contain an object.


### YOLO family:
You Only Look Once (YOLO) family of detection frameworks aim to build a real time object detector, which what they lack in small differences of accuracy when compared to the two stage detectors, are able to provide faster inferences.

YOLO does not go through a regional proposal phase (as was the case with two stage detectors), instead predicts over limited bounding boxes generated by splitting image into a grid of cells, with each cell being responsible for classification and generation of bounding boxes, which are then consolidated by NMS.

#### Steps:
1. Prediction of bounding box coordinates (cell location offsets: [x, y] and dimensions of bounding box: [width, height])
2. Objectness score which indicates the probability of the cell contains an object. (probability that box contains an object x IoU of prediction and ground truth)
3. Class Prediction using sigmoid/softmax: if bounding box contains an object- network predicts probability for K number of classes.
---

## YOLOv5 architecture
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/yolov5.png?token=AGCG5WBMOXOC4YGWJCF2QATAMLLU4)

### New ideas:

1. **Data Augmentations**
Possibly one of the game changing aspects of YOLO is the use of mosaic augmentation, which helps detect smaller objects in the image by combining four images into one in arbitrary ratios. It encourages localization of a variety of images in different portions of the image.
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/Result-of-a-mosaic-data-augmentation-example-from-four-input-images-best-viewed-in.jpg?token=AGCG5WFM2NHWPDD26U3CASLAMLL4O)

2. **Anchor Boxes**
Usually learnt based on distribution of bounding boxes in a custom dataset through clustering. Yolov3 introduced the importance of learning anchor boxes through the custom dataset yields better accuracy. A YAML specified must contain number of input channels to network, depth and width multipliers and anchor box dimensions.
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/Anchorboxes.png?token=AGCG5WH5VMDRPEYKFZTMSNLAMLLZK)

3. **Backbone**
Cross Stage Partial networks, with close connection to the densenet layer, solve the gradient problem in large convolutional nets which leads to lesser parameters thus lesser computational overhead, helping conncect layers which solve issues of vanishing gradients and assisting the ability of network to learn mappings of reused features.
YOLOv5 uses CSPDarknet and CSPResNext which help eradicate bottlenecks during computations in CSP DenseNet, which then help improve learning process with the passage of an unedited version of the feature map.
![](https://raw.githubusercontent.com/TanyaChutani/YoloV5/main/assets/CSP.png?token=AGCG5WHBZFBMSUTNTY2UYXDAMLL6W)
4. **Size and inference speeds**
**YOLOv5 is faster** and 90% smaller than YOLOv4 in size which makes it a winner to be used for real time inference. (27 vs 244 MB)

---


## Stages of YOLOv5 model

1. **DataLoader** class is used to iterate through the data, returning a tuple with the batch being passed through a deep neural network.
2. A **loss function** lets know the model of its inability to fit the data, with idea being to converge on an optimum set of parameters.
3. **Splitting into training and testing data** : The issue may then be that the model "**overfits**" the training data and may fail when generalizing to a different subset. So we need for separate training, validation and testing steps which help combat overfitting. They help us have the idea of how well our model does on unseen data.
In an effort to increase model's performance on validation data, we can tune **training hyperparameters**, **model architecture** and make use of **data augmentation techniques**.
3. The **metric** used to determine model performance is **Mean Average Precision**. **Precision**, which is the measure of the percentage of correctly predicted labels, and **recall**, which is the measure of how well the model was able to fit the datapoints corresponding to the positive class, are along with IoU (Intersection over Union) which is the area of the overlap between our predictions and ground truth. A threshold is usually chosen to classify whether the prediction is a true positive or a false negative. **Average precision** is the area under the precision-recall curve and follows precision and recall in having a value between 0 and 1. Interpolation of the precision value for a recall by the maximum precision which makes the curve between precision and recall be less susceptible to small changes in ranking of the points. **Mean Average Precision (or mAP)** is calculated by average precision values for each class label.



---


## Cloning yolov5 repository
This step helps clone the YoloV5 repository while installing all the required libraries into our environment.
* git clone https://github.com/ultralytics/yolov5.git  
* pip3 install -r yolov5/requirements.txt  


---
## Input Pipeline

1. Load images and annotations
2. Extract relevant fields
3. Creating consolidated data frame of images and annotations
4. Split data into training and validation sets
5. Convert input format to one that is supported by YOLO
6. Transform and save respective images and bounding boxes based on YOLO formatted labels

---

## Training yolov5
The training on the model is done by the train.py. parameters of note, include:
1. **--cfg**: Path to the YAML file for configuration of the YOLOv5 model, which contains our preferences for everything from number of channels, model depth and width multipliers, choices for anchor boxes to our prefereces for the architecture (backbone or head) of the model.
2. **--epochs**: Number of times the training and validation steps are accomplished by forward and backward passes through network to reach a state of mininma on loss function.
3. **--batch**: Size of the batch which is passed through model on training and validation for steps across the network.
4. **--data**: Path to the YAML file which contains the training and validation directories and class names followed by the respective counts.
5. **--img**: Image size of the training images. Upon experimentation with this parameter, 1024 was chosen to be a good parameter value.


*  python yolov5/train.py --img 1024 --batch 4 --epochs 50 --data yolo.yml --cfg yolov5/models/yolov5s.yaml


---


## Detection with yolov5
In the final step, we should be able to detect objects on unseen images and label them into respective categories (localization+classification), which is accomplished by running the command detect.py.
It has the following arguments most commonly used as per their usage in the model:
1. **--source**: Path to the test image
2. **--weights**: Path to saved weights
3. **--img**: Size of the test image
4. **--save_txt**: either have predicted bounding boxes drawn on the localized object in the image or predict predictions to a text file

*   python yolov5/detect.py --source /content/yolov5_data/images/validation --weights /content/runs/train/exp/weights/best.pt

---


## Docker
Docker helps in execution of processes in isolated containers (lightweight execution environments), with a container being any process which runs on a host, which could be be local or remote.

### Commands in use:

1.
**Command**: **docker build**

**Usage**: docker build . -t yolov5:0.1

**Description**: Helps builds Docker images from a Dockerfile and a context, which are files located on a specified path.


2.
**Command**: **docker run**

Usage to detect objects in a user specified image: docker run --name yolov5 yolov5:v1.0 python3 detect.py --source image_000000068.jpg --weights best.pt --project /root/yolov5

Description: This command executes the containerised application of YOLOv5 with parameters used:
1. --name: name/identifier of the container.
2. --tag: used to specify version of the docker image we want to run on the container.
3. file specific params: source image (detect.py), weights (best.pt) and save directory (--project)


3.
**Command**: **docker cp**

**Usage**: docker cp yolo:/root/yolov5 C:\Users\HP\Downloads\docker\yolov5

**Description**: This command copies the contents of a source path to the destination path, from the container's file system to a local machine or vice versa.


**Dockerfile**: It is a text file containing all instructions and commands which could be called by the user to build an image.

