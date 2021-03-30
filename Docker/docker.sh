docker build . -t yolov5:v1.0
docker run --name yolov5 yolov5:v1.0 python3 detect.py --source image_000000068.jpg --weights best.pt --project /root/yolov5
docker cp yolo:/root/yolov5 C:\Users\HP\Downloads\docker\yolov5
