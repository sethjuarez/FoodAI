import os
import argparse
from bing import Bing
from pathlib import Path

def download(query, target):
    # Free to share and use commercially
    filter = '+filterui:license-L2_L3_L4'

    t = Path(target).resolve()
    if not t.exists():
        os.makedirs(str(t))

    image_dir = (t / query).resolve()
    if not image_dir.exists():
        os.makedirs(str(image_dir))

    bing = Bing(query, limit=500, output_dir=str(image_dir), 
                adult='on', timeout=100, filters=filter)
    bing.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hot Dog vs Pizza')
    parser.add_argument('-q', '--query', help='image query', default='tacos')
    parser.add_argument('-t', '--target', help='output directory', default='../data/test')


    args = parser.parse_args()

    download(args.query, args.target)