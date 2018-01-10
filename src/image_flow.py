from darkflow.net.build import TFNet
import cv2

from io import BytesIO
import time
import requests
from PIL import Image
import numpy as np

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

birdsSeen = 0
def handleBird():
    pass

while True:
    r = requests.get('http://192.168.1.111:5001/image.jpg') # a bird yo
    curr_img = Image.open(BytesIO(r.content))
    curr_img_cv2 = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)

    result = tfnet.return_predict(curr_img_cv2)
    print(result)
    for detection in result:
        if detection['label'] == 'bird':
            print("bird detected")
            birdsSeen += 1
            curr_img.save('img/%i.jpg' % birdsSeen)
    print('running again')
    time.sleep(4)
