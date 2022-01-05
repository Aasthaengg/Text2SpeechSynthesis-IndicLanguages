import os
import sys
from multiprocessing import Pool
import ffmpeg
import tqdm

def convert(f):
    
    dir = sys.argv[1]
    newdir = sys.argv[2]
    
    try:
        stream = ffmpeg.input(dir+f)
        audio = stream.audio
        stream = ffmpeg.output(audio, newdir+f, **{'ar': '22050','acodec':'pcm_s16le','ac':1})
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)

    except:
        print(dir+f)

if __name__ == '__main__':

    if not len(sys.argv) == 3:
        print('Usage: python3 format_changer.py directory/ directory/')
        sys.exit(0)

    dir = sys.argv[1]
    newdir = sys.argv[2]

    files = os.listdir(dir)

    with Pool(16) as p:
        r = list(tqdm.tqdm(p.imap(convert, files), total=len(files)))

