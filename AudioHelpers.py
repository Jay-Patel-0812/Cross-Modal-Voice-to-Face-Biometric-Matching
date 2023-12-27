# AudioHelpers.py
# This file contains the required functions used for 
# processing audio files and to extract spectrogram 
# and mfcc co-efficents from the 

# importing requried libraries
import librosa
import noisereduce as nr

# rate at which audio should be sampled
SamplingRate = 16000

# function that trims the given audio to targetLength seconds
# audio - the input audio signal
# sr - sampling rate of the audio
# targetLength - in seconds to trim
def trimAudio(audio, sr, targetLength=3):
  
  if audio is None:
    return None
  
  # insufficent length
  if len(audio) <= targetLength*sr:
    return audio
 
  result = audio[0:targetLength*sr]
  return result
   

# perform pre-processing steps including 
# background noise removal, trimming audio to 3seconds length
def preProcessAudio(audio, sr):
  
  # trim the audio to a 3 second length
  trimmed_audio  = trimAudio(audio, sr, targetLength=3)
  
  if trimmed_audio is None:
    return None
  
  # remove background noise
  S = nr.reduce_noise(y=trimmed_audio, sr=sr)
  return S

# function to calculate Mel-frequency Cepstral co-efficients from audio signal
# n_mfcc : number of mfcc co-efficients to consider
def mfcc(audio, sr, n_mfcc=20):
  if audio is None:
    return None
  
  # using in-built functions from librosa to calculate mfcc
  S = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
  return S

# function that calcluates mel-spectrogram representation of input audio
# n_fft: length of the window used in fast fourier transform
def melSpectrogram(audio, sr, n_fft=300):
  if audio is None:
    return None
  
  S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft)
  return S

# function that takes audio file path and returns the extracted features
# features: mfcc or mel-spectrogram
def extractAudioFeatures(audioPath, feature='mfcc', n_mfcc=20, n_fft=300):
  audio, sr = librosa.load(audioPath, sr=SamplingRate)
  
  # perform preprocessing
  audio = preProcessAudio(audio=audio, sr=sr)
  
  if feature == 'mfcc':
    return mfcc(audio, sr, n_mfcc)
  elif feature == "mel-sepctrogram":
    return melSpectrogram(audio, sr, n_fft)
  
  return None