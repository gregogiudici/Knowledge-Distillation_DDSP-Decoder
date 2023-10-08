
import os
import os.path as path

import soundfile as sf
import librosa
import numpy as np
import sys

'''
Use to recursively resample all the dataset files to 16kHz to compute FAD. 
'''
if len(sys.argv) != 4:
    print ("Usage: resample.py [data directory] [saving directory] [sample rate]")
    quit()
    
if not (path.isdir(sys.argv[1])):
    print("no data directory")
    quit()
    
if not (path.isdir(sys.argv[2])):
    os.mkdir(sys.argv[2])

#SAMPLE_RATE=16000
SAMPLE_RATE = int(sys.argv[3])
list_of_files = {}
for (dirpath, dirnames, filenames) in os.walk(sys.argv[1]):
    for filename in filenames:
        if filename.endswith('.wav'): 
            list_of_files[filename] = [dirpath, filename]
            
for k in list_of_files.keys():
    f = os.sep.join([list_of_files[k][0], list_of_files[k][1]])
    print("Processing {}".format(f))
    data, sr = librosa.load(f, sr=SAMPLE_RATE)
    sf.write(
        os.sep.join([sys.argv[2], list_of_files[k][1]]),
        data,
        SAMPLE_RATE,
    )    


