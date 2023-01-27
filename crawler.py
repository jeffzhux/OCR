import io
import requests
import urllib.request
import json
import lmdb
import numpy as np
import cv2
from PIL import Image, ImageFilter


def img2byte(img):
    is_success, img_arr = cv2.imencode('.jpg', img)
    return img_arr.tobytes()

path = './data/test'
is_lmdb = False
num_loop = 1
if is_lmdb:    
    env = lmdb.open(path)
    with env.begin(write=True) as txn:
        index = 0
        for j in range(num_loop):
            re_json = json.loads(requests.get('https://gen.caca01.com/rd/test').text)
            for i in re_json['codelist']:
                req_img = urllib.request.urlopen(f'data:image/gif;base64,{i["code"]}')
                arr = np.asarray(bytearray(req_img.read()), dtype=np.uint8)

                # high resolution
                img = cv2.imdecode(arr, -1)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = cv2.resize(img, (100, 32))
                txn.put(b'image-%09d' % index, img2byte(img))

                # label
                txn.put(b'label-%09d' % index, str(i['ans']).encode())
                # print(i['ans'])
                index += 1
                
        # print(re.text)
        print(index)
        txn.put(b'num-samples', str(index-1).encode())
else:
    index = 0
    for j in range(num_loop):
        re_json = json.loads(requests.get('https://gen.caca01.com/rd/test').text)
        for i in re_json['codelist']:
            urllib.request.urlretrieve(f'data:image/gif;base64,{i["code"]}', f'{path}/{index}.jpg')
            index += 1