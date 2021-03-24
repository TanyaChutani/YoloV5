from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import json
import os
import cv2

class InputPipeline:
    def __init__(self,input_file
                 ,width=1024.0
                 ,height=1024.0):
      self.file = input_file
      self.width = width
      self.height = height
      self._main()

    def _load_json(self):      
      annotate = open(self.file)
      annotation_file = json.load(annotate)
      return annotation_file
    
    def _extract_column(self):
      annotation_file = self._load_json()
      images = pd.json_normalize(annotation_file['images'])
      boxes = pd.json_normalize(annotation_file['annotations'])
      return images, boxes
    
    def _clean_data(self):
      images, boxes = self._extract_column()
      images.drop('license',axis=1,inplace=True)
      boxes.drop(['segmentation','license','id','area','iscrowd']
                       ,axis=1,inplace=True)
      images.rename(columns = {'id':'image_id'}, inplace = True)
      return images, boxes
      
    def _merge_data(self):
      images, boxes = self._clean_data()
      boxes = boxes.groupby('image_id').\
            aggregate(lambda tdf: tdf.tolist())
      final_data = pd.merge(images,boxes,on="image_id")
      return final_data
    
    def _spliting_data(self):
      final_data = self._merge_data()
      train_data, test_data = train_test_split(final_data,test_size=0.1,
                                         shuffle=True)
      train_data.reset_index(drop=True,inplace=True)
      test_data.reset_index(drop=True,inplace=True)
      return train_data, test_data

    @staticmethod
    def convert_format_xywh(out):
       return np.stack([
        (out[...,0]+out[...,2])/2.0,
        (out[...,1]+out[...,3])/2.0,
        out[...,2]-out[...,0],
        out[...,3]-out[...,1]],axis=-1)
       
    def _transform(self,data):
      for id,bboxes in enumerate(data.bbox):
        for idx,box in enumerate(bboxes):
          box.insert(0,(data["category_id"][id][idx]-1))
          box[3] = box[3]+box[1]
          box[4] = box[4]+box[2]
          box[1] = (box[1]*self.width)/data.width[id]
          box[2] = (box[2]*self.height)/data.height[id]
          box[3] = (box[3]*self.width)/data.width[id]
          box[4] = (box[4]*self.height)/data.height[id]
          box[1:] = self.convert_format_xywh(np.array(box[1:]))
          box[1] = box[1]/self.width
          box[2] = box[2]/self.height
          box[3] = box[3]/self.width
          box[4] = box[4]/self.height
    
    def _creating_data(self):
      train_data, test_data = self._spliting_data()
      self._transform(train_data)
      self._transform(test_data)
      return train_data, test_data
    
    @staticmethod
    def _make_dir(path):
      if not os.path.exists(path):
        os.makedirs(path)

    def _saving_images_labels(self,data,choose_train_val="train"):
      self._make_dir(f"yolov5_data/labels/{choose_train_val}/")
      self._make_dir(f"yolov5_data/images/{choose_train_val}/")
      for idx,row in enumerate(data.values):
        filename = "yolov5_data/labels/{}/{}".format(choose_train_val,
                                             (data['file_name'][idx]).split(".")[0]+".txt")
        np.savetxt(filename,data["bbox"][idx],
                    fmt=["%d","%f","%f","%f","%f"])
        shutil.copyfile(os.path.join("trainval/images/",data["file_name"][idx]),
                          os.path.join(f"yolov5_data/images/{choose_train_val}/",data["file_name"][idx]))

    def _main(self):
      train_data, test_data = self._creating_data()
      self._saving_images_labels(train_data)
      self._saving_images_labels(test_data,choose_train_val="validation")
