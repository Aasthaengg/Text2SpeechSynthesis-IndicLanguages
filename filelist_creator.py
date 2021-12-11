import os
import sys

def create_filelist(transcription_path):
    
    # Read file
    f = open(transcription_path + 'transcription.txt', 'r+')

    # Write file
    writepath = 'filelists/openslr_hindi_{}.txt'

    if 'train' in transcription_path:
        writepath = writepath.format('train')
    else:
        writepath = writepath.format('test')
    
    w = open(writepath, 'a+')

    lines = f.readlines()
    for l in lines:
        l = l[:-1]
        filename, text = l.split(' ', 1)
        filename = transcription_path + 'audio/' + filename + '.wav'

        data = '{}|{}\n'.format(filename, text)
        w.write(data)

    f.close()
    w.close()

if __name__ == '__main__':
    
    if not len(sys.argv) == 2:
        print('Usage: python3 filelist_creator.py dataset_path/')
        sys.exit(0)
    
    dataset_path = sys.argv[1]
    folders = ['train/', 'test/']

    for f in folders:
        transcription_path = dataset_path + f
        create_filelist(transcription_path)
