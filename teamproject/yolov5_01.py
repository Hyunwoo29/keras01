import glob
input_dir='C:\\ProgramData\\Anaconda3\\envs\\yolov5\\Yolo_mark\\x64\\Release\\data\\img\\'
f = open('./teamproject/yolov5/train_list.txt', 'rt', encoding="UTF8")

input_file=glob.glob(input_dir+'*.txt')

for file in input_file:
    f.write(file[:200]+file[200:-4]+'\n')
f.close()

f = open('./teamproject/yolov5/val_list.txt', 'rt', encoding="UTF8")

for file in input_file:
    f.write(file[:200]+file[200:-4]+'\n')
f.close()
