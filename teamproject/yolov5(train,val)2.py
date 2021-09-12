import yaml

with open('./teamproject/yolov5/data/movie.yaml', 'r') as f:
    data = yaml.load(f)

# print(data)
# {'train': '/data/images/train', 'val': '/data/images/valid', 'nc': 1, 'names': ['person']}
data['train'] = './teamproject/yolov5/train.txt'
data['val'] = './teamproject/yolov5/val.txt'

with open('./teamproject/yolov5/data/movie.yaml', 'w') as f:
    yaml.dump(data, f)

# print(data)
# {'train': './teamproject/yolov5/train.txt', 'val': './teamproject/yolov5/val.txt', 'nc': 1, 'names': ['person']}

