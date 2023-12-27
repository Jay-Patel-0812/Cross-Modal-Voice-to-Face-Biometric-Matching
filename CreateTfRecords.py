# CreateTfRecords
# This file contains the functions to create tensorflow records
# from the input data for static and dynamic architectures

# importing required libraries
import cv2
import tensorflow as tf
import sys
import pandas as pd
import os
import numpy as np
from AudioHelpers import extractAudioFeatures
from VideoHelpers import frames_from_video_files

# function to serailze and convert a ndarray to a bytelist tensorflow feature
def _float_array_feature(array):
    tensor = tf.convert_to_tensor(array)
    value =  tf.io.serialize_tensor(tensor).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# function to convert an integer to tensorflow feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  
# required output image dimensions
ImageDimensions = (224,224)

# paths to the folders contains audio video and image data
# change this to the folder containing the data files
rootAudioPath = r'D:\BiometricProject\Mini_Data\Audio\Dev'
rootVideoPath = r'D:\BiometricProject\Mini_Data\Video\Dev'
rootImgPath = r'D:\BiometricProject\Mini_Data\Images\Dev'

def load_imag(imgPath):
     # read the image and resize into a fixed size
     
     img = cv2.imread(os.path.join(rootImgPath, imgPath))
     
     if img is None:
       return None
     
     # perform resizing 
     img = cv2.resize(img, ImageDimensions)
     
     # rescaling to normalize
     img = img/255.0
     return img

def createDataRecord_static1(outfileName, img_addrs, audio_addrs, labels):
    
    writer = tf.io.TFRecordWriter(outfileName)
    
    for i in range(len(img_addrs)):
      
      if i%1000 == 0:
        print('Created {}/{} records'.format(i, len(img_addrs)))
        sys.stdout.flush()
        
      img = load_imag(img_addrs[i])
      
      # load audio file image
      audio = extractAudioFeatures(os.path.join(rootAudioPath,audio_addrs[i]), 'mfcc', n_mfcc=20)
      audio = np.array(audio, dtype=np.float64)
      label = labels[i]
      
      if img is None:
        continue
      feature={
        'image_raw': _float_array_feature(img),
        'audio_raw': _float_array_feature(audio),
        'label': _int64_feature(label)
      }
      
      example = tf.train.Example(features = tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
      
    writer.close()
    sys.stdout.flush()
    

def createDataRecord_static2(outfileName, img1_addrs, img2_addrs, audio_addrs, labels):
    writer = tf.io.TFRecordWriter(outfileName)
    
    for i in range(len(audio_addrs)):
      
      if i%1000 == 0:
        print('Created {}/{} records'.format(i, len(audio_addrs)))
        sys.stdout.flush()
        
      img1 = load_imag(img1_addrs[i])
      img2 = load_imag(img2_addrs[i])
      
      # load audio file image
      audio = extractAudioFeatures(os.path.join(rootAudioPath,audio_addrs[i]), 'mfcc', n_mfcc=20)
      audio = np.array(audio, dtype=np.float64)
      label = labels[i]
      
      if img1 is None or img2 is None:
        continue
      feature={
        'image1_raw': _float_array_feature(img1),
        'image2_raw': _float_array_feature(img2),
        'audio_raw': _float_array_feature(audio),
        'label': _int64_feature(label)
      }
      
      example = tf.train.Example(features = tf.train.Features(feature=feature))
      writer.write(example.SerializeToString())
      
    writer.close()
    sys.stdout.flush()

# creates Tf Records with the inputs as follows
# x = (face, video, audio) 
# y = label = 0 or 1
# y = 0 - indicates face,video are of not of same person that of audio
# y = 1 - indicate face,video are of same person that of audio
def create_Data_Record_Dynamic1(outfileName, img_addrs, audio_addrs, video_addrs, labels):
    writer = tf.io.TFRecordWriter(outfileName)

    for i in range(len(audio_addrs)):
       
       # logging for progress indication
       if i%1000 == 0:
         print('Created {}/{} records'.format(i, len(img_addrs)))
         sys.stdout.flush()
         
       # extract audio, video and image inputs
       img = load_imag(img_addrs[i])
       
       audio = extractAudioFeatures(os.path.join(rootAudioPath, audio_addrs[i]), 'mfcc', n_mfcc=20)
       audio = np.array(audio, dtype=np.float64)
       video_frames = frames_from_video_files(video_addrs[i],n_frames=10, output_size=ImageDimensions)
       
       label = labels[i]
       
       if img is None or video_frames is None or audio is None:
         continue
       
       feature = {
          'image_raw':_float_array_feature(img), 
          'video_raw':_float_array_feature(video_frames), 
          'audio_raw':_float_array_feature(audio),
          'label': _int64_feature(label)
       }
       
       example = tf.train.Example(features = tf.train.Features(feature = feature))
       writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


# creates Tf Records with the inputs as follows
# x = (face1, video1, face2, vidoe2, audio) 
# y = label = 0 or 1
# y = 0 - indicates audio matches with face1, video1
# y = 1 - indicate audio matches with face2, video2
def create_Data_Record_Dynamic2(outfileName, img1_addrs, img2_addrs, video_addrs1, video_addrs2, audio_addrs, labels):
    writer = tf.io.TFRecordWriter(outfileName)
    
    for i in range(len(audio_addrs)):
         # logging for progress indication
         if i%1000 == 0:
           print('Created {}/{} records'.format(i, len(audio_addrs)))
           sys.stdout.flush()
           
         # extract audio, video and image inputs
         img1 = load_imag(img1_addrs[i])
         img2 = load_imag(img2_addrs[i])
         
         audio = extractAudioFeatures(os.path.join(rootAudioPath, audio_addrs[i]), 'mfcc', n_mfcc=20)
         audio = np.array(audio, dtype=np.float64)
         video_frames1 = frames_from_video_files(video_addrs1[i],n_frames=10, output_size=ImageDimensions)
         video_frames2 = frames_from_video_files(video_addrs2[i],n_frames=10, output_size=ImageDimensions)
         
         label = labels[i]
         
         if img1 is None or img2 is None or video_frames1 is None or video_frames2 is None or audio is None:
           continue
         
         feature = {
            'image_raw1':_float_array_feature(img1), 
            'image_raw2':_float_array_feature(img1), 
            'video_raw1':_float_array_feature(video_frames1), 
            'video_raw2':_float_array_feature(video_frames2), 
            'audio_raw':_float_array_feature(audio),
            'label': _int64_feature(label)
         }
         
         example = tf.train.Example(features = tf.train.Features(feature = feature))
         writer.write(example.SerializeToString())

    writer.close() 
    sys.stdout.flush()