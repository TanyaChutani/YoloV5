# Object detection with Yolov5

## Basics for Object Detection
Object detection involves classifying localized bounding boxes in the image, that is classifying the objects and searching on the position for the bounding box. The training data for object detection models include images and corresponding bounding box coordinates.On a predefined set of class lanels (e.g. people, and cars), object detection helps us describe the locations of each detected object in the image using a bounding box. 
<br>
Through object detection mechanisms and algorithms, we are able to understand what's in an image, while being able to describe both what is in an image and the locations of those objects in the image.

---

## Input data format types for object detection tasks: VOC and MS-COCO
Input data for Pascal VOC is an **XML file**, whereas COCO dataset uses a **JSON file**.

Bounding boxes in VOC and COCO challenges are differently represented and they are as follows:
1. PASCAL VOC: (xmin-top left, ymin-top left,xmax-bottom right, ymax-bottom right)
2. COCO: (x-top left, y-top left, width, height)

The JSON file for MS COCO contains the images and their corresponding labels.
Sample image and annotations:
https://miro.medium.com/max/700/1*MzOK-eQMb0R2OuM7FdZFhg.png
https://miro.medium.com/max/700/1*-X2_9_vgQorX6J9BF8webQ.png

Example PASCAL VOC bounding box
https://miro.medium.com/max/565/1*J84PBv70HWVW_tJ2zQwd4g.png


The format of bounding box in the provided json is specified by upper left coordinates and respective dimensions (width, height) of each box.


---


## Key components of YOLOv5 model

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
