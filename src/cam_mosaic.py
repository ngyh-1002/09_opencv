import cv2
import dlib

# 얼굴 검출기 생성 --- ①
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
#cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 320)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 영역 검출 --- ②
    faces = detector(gray)
    for rect in faces:
        # 얼굴 영역을 좌표로 변환 후 사각형 표시 --- ③
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y
        roi = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi = cv2.resize(roi, (w//15, h//15)) # 1/rate 비율로 축소
        # 원래 크기로 확대
        roi = cv2.resize(roi, (w,h), interpolation=cv2.INTER_AREA)  
        img[y:y+h, x:x+w] = roi   # 원본 이미지에 적용
    
    cv2.imshow("face landmark", img)
    if cv2.waitKey(1)== 27:
        break
cap.release()