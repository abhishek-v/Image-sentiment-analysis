import pickle
import os
import cv2
from skimage import io


os.chdir("./data")

files = os.listdir()
for file in files:
  os.chdir(file)
  sub_files = os.listdir()
  for sub_file in sub_files:
    try:
      img = io.imread(sub_file)
    except:
      print(sub_file)
      exit(1)
  os.chdir("..")
