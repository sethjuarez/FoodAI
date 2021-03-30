import os
import re
import sys
import uuid
import imghdr
import urllib
import posixpath
import urllib.request
from pathlib import Path

'''
Python api to download image form Bing.
Author: Guru Prasad (g.gaurav541@gmail.com)
(Seth Juarez) ADDENDUM:
Original code found here: 
https://github.com/gurugaurav/bing_image_downloader

Made adjustments to file names and paths for 
downloaded images
'''


class Bing:
    def __init__(self, query, limit, output_dir, adult, timeout, filters=''):
        self.download_count = 0
        self.query = query
        self.output_dir = output_dir
        self.adult = adult
        self.filters = filters

        assert type(limit) == int, "limit must be integer"
        self.limit = limit
        assert type(timeout) == int, "timeout must be integer"
        self.timeout = timeout

        self.headers = {'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        self.page_counter = 0

    def save_image(self, link, file_path):
        request = urllib.request.Request(link, None, self.headers)
        image = urllib.request.urlopen(request, timeout=self.timeout).read()
        if not imghdr.what(None, image):
            print('[Error]Invalid image, not saving {}\n'.format(link))
            raise Exception()
        with open(file_path, 'wb') as f:
            f.write(image)

    def download_image(self, link):
        self.download_count += 1

        # Get the image link
        try:
            path = urllib.parse.urlsplit(link).path
            filename = posixpath.basename(path).split('?')[0]
            file_type = filename.split(".")[-1]
            if file_type.lower() not in ["jpe", "jpeg", "jfif", "exif", "tiff", "gif", "bmp", "png", "webp", "jpg"]:
                file_type = "jpg"

            # Download the image
            print("[%] Downloading Image #{} from {}".format(self.download_count, link))

            img_path = (Path(self.output_dir) / f'{str(uuid.uuid1()).lower()}.{file_type}').resolve()
            self.save_image(link, str(img_path))
            print("[%] File Downloaded !\n")
        except Exception as e:
            self.download_count -= 1
            print("[!] Issue getting: {}\n[!] Error:: {}".format(link, e))

    def run(self):
        while self.download_count < self.limit:
            print('\n\n[!!]Indexing page: {}\n'.format(self.page_counter + 1))
            # Parse the page source and download pics
            request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(self.query) \
                          + '&first=' + str(self.page_counter) + '&count=' + str(self.limit) \
                          + '&adlt=' + self.adult + '&qft=' + self.filters
            request = urllib.request.Request(request_url, None, headers=self.headers)
            response = urllib.request.urlopen(request)
            html = response.read().decode('utf8')
            links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)

            print("[%] Indexed {} Images on Page {}.".format(len(links), self.page_counter + 1))
            print("\n===============================================\n")

            for link in links:
                if self.download_count < self.limit:
                    self.download_image(link)
                else:
                    print("\n\n[%] Done. Downloaded {} images.".format(self.download_count))
                    print("\n===============================================\n")
                    break

            self.page_counter += 1