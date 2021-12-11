import os
from scipy.io import wavfile
import scipy.signal as sps
import numpy as np
import sys

if __name__ == '__main__':

    if not len(sys.argv) == 2:
        print('Usage: python3 upsampler.py directory/')
        sys.exit(0)

    dir = sys.argv[1]
    files = os.listdir(dir)
    # Your new sampling rate
    new_rate = 22050

    counter = 0
    for f in files:

        # Read file
        sampling_rate, data = wavfile.read(dir+f)

        # Resample data
        number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
        data = sps.resample(data, number_of_samples).astype(np.int16)

        wavfile.write(dir+f, new_rate, data)

        counter += 1
        print('Count: {}, File: {} | Rate: {} | Samples: {}'.format(counter, f, new_rate, len(data)))
