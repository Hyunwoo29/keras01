import cv2

vidcap = cv2.VideoCapture('./_data/[일본영화][예고편] 도쿄 리벤저스 (한글자막).mp4')

count = 0

while(vidcap.isOpened()):
    ret, image = vidcap.read()

    if(int(vidcap.get(1)) % 10 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("./_data/img/movie_%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()