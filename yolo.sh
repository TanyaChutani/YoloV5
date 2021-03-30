git clone https://github.com/ultralytics/yolov5.git
pip3 install -r yolov5/requirements.txt

python main_input.py

python yolov5/train.py --img 1024 --batch 4 --epochs 1 --data yolo.yml --cfg yolov5/models/yolov5s.yaml

python yolov5/detect.py --source yolov5_data/images/validation --weights best.pt
