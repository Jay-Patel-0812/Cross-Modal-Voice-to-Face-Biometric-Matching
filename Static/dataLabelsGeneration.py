# to randomly generate input and output labels using audio and video files

import random
import os
import pandas as pd

AudioFolder = r'D:\BiometricProject\Mini_Data\Audio\Dev'
ImageFolder = r'D:\BiometricProject\Mini_Data\Images\Dev'


AudioFilesList = []
ImageFilesList = []

for subject in os.listdir(AudioFolder):
    subjectAudioFiles = []
    subjectImageFiles = []
    for audio in os.listdir(os.path.join(AudioFolder, subject)):
        audioFileName = os.path.join(subject, audio)
        subjectAudioFiles.append(audioFileName)
    for img in os.listdir(os.path.join(ImageFolder, subject)):
        imgFileName = os.path.join(subject, img)
        subjectImageFiles.append(imgFileName)
    AudioFilesList.append(subjectAudioFiles)
    ImageFilesList.append(subjectImageFiles)
    


# static architecture -1 
# Data = []
# N_2=100000
# instancesPerSubject = 30

# # 50,000 instances from different users
# # 50,000 instances of same users
# for _ in range(N_2//2):
#    [i,j]=random.sample(range(50),2)
#    ind1 = random.randint(0,instancesPerSubject-1)
#    ind2 = random.randint(0,instancesPerSubject-1)
#    Data.append([AudioFilesList[i][ind1],ImageFilesList[j][ind2],0])

# for _ in range(N_2//2):
#    i=random.randint(0,50-1)
#    ind1 = random.randint(0,instancesPerSubject-1)
#    ind2 = random.randint(0,instancesPerSubject-1)
#    Data.append([AudioFilesList[i][ind1],ImageFilesList[i][ind2],1])
   
# DataFrame = pd.DataFrame(Data, columns=['voice','face','label'])
# DataFrame.head()
# DataFrame.sample(frac=1).to_csv('labelledDatasetStatic1.csv',index=False) 
 

# static architecture - 2
# # Generate 1 lakh instances
# Data = []
# N_2=100000
# instancesPerSubject = 30

# # 50,000 instances from different users
# # 50,000 instances of same users
# for _ in range(N_2//2):
#    [i,j]=random.sample(range(len(AudioFilesList)),2)
#    ind1 = random.randint(0,instancesPerSubject-1)
#    ind2 = random.randint(0,instancesPerSubject-1)
#    ind3 = random.randint(0, instancesPerSubject-1)
#    Data.append([AudioFilesList[i][ind1],ImageFilesList[i][ind2],ImageFilesList[j][ind3],0])
#    # 0 indicating that audio corresponds to image 1

# for _ in range(N_2//2):
#    [i,j]=random.sample(range(len(AudioFilesList)),2)
#    ind1 = random.randint(0,instancesPerSubject-1)
#    ind2 = random.randint(0,instancesPerSubject-1)
#    ind3 = random.randint(0, instancesPerSubject-1)
#    Data.append([AudioFilesList[i][ind1],ImageFilesList[j][ind2],ImageFilesList[i][ind3],1])
#    # 1 indicating that audio corresponds to image 2
   
# DataFrame = pd.DataFrame(Data, columns=['voice','face1','face2','label'])
# DataFrame.head()
# DataFrame.sample(frac=1).to_csv('labelledDatasetStatic2.csv',index=False)

# dynamic architecture -1
# dynamic architecture -2 