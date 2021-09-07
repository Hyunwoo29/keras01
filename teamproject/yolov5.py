from glob import glob
import shutil
import os

data = ['train', 'valid', 'test']

for i  in data:
    source = './yolov5/data/images/' + i + '/'
    images = './yolov5/data/images/' + i
    labels = './yolov5/data/images/' + i
    mydict = {
        images: ['jpg', 'png', 'gif','jpeg','JPG'],
        labels: ['txt','json']
    }
    for destination, extensions in mydict.items():
        for ext in extensions:
            for file in glob(source + '*.' + ext):
                shutil.move(file, os.path.join(destination,file.split('/')[-1]))
