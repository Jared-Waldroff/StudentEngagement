"""
A simple script to extract the HOG features from DAiSEE. This scripts assumes that user has already extracted out the frames from videos
and saved them as shown in below directory tree:

Nate:
-> Make sure that the script and dataset folder are in same directory. 
-> If not, then please adjust "data_path" and "target" variable declared at the beginning of the script according to the path.

dataset/
  Training/
    person1/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
    person2/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
        .
        .
    .
    .
  Testing/
    person1/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
    person2/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
        .
        .
    .
    .
  Validation/
    person1/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
    person2/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
        .
        .
    .
    .


The script will make a new histogram folder and put the plots insode the folder in the same pattern as in dataset:

histogram/
  Training/
    person1/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
    person2/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
        .
        .
    .
    .
  Testing/
    person1/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
    person2/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
        .
        .
    .
    .
  Validation/
    person1/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
    person2/
        video1/
          frame1.png
          frame2.png
          .
          .
        video2/
          frame1.png
          frame2.png
          .
          .
        .
        .
    .
    .

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

data_path = 'dataset'
target = 'histogram'
if not os.path.exists(target):
  os.makedirs(target)

sub_data = os.listdir(data_path)

for folder in sub_data:
  set_path = os.path.join(target,folder)
  if not os.path.exists(set_path):
    os.makedirs(set_path)
  person_folder_list = os.listdir(os.path.join(data_path,folder))
  for sub_person_folder in person_folder_list:
    person_videos_list = os.listdir(os.path.join(data_path,folder,sub_person_folder))
    sub_person_video_path = os.path.join(target,folder,sub_person_folder)
    if not os.path.exists(sub_person_video_path):
      os.makedirs(sub_person_video_path)

    video_list = os.listdir(os.path.join(data_path,folder,sub_person_folder))
    for video in video_list:
      folder_path = os.path.join(target,folder,sub_person_folder,video)
      if not os.path.exists(folder_path):
        os.makedirs(folder_path)

#now let us save the histograms of each frame in histogram folder.
hist_folder_list = os.listdir(data_path)

for folder in hist_folder_list:
  sub_folder_list = os.listdir(os.path.join(data_path, folder))
  for sub_folder in sub_folder_list:
    person_video_list = os.listdir(os.path.join(data_path, folder,sub_folder))
    for video in person_videos_list[0:1]:
      frame_list= os.listdir(os.path.join(data_path, folder,sub_folder,video))
      for frame in frame_list:
        img = cv2.imread(os.path.join(data_path, folder,sub_folder,video,frame))
        hist = cv2.calcHist([img],[0],None,[256],[0,256])
        print(img.ravel(),256,[0,256])
        plt.hist(img.ravel(),256,[0,256]);
        print("saving the histogram of",os.path.join(data_path, folder,sub_folder,video,frame),"to",os.path.join(target, folder
          ,sub_folder,video,frame))
        plt.savefig(os.path.join(target, folder,sub_folder,video,frame))
