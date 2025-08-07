import cv2
import numpy as np
import os, glob
import csv
from datetime import datetime

# 미등록자 얼굴 저장 함수
def save_unknown_face(img):
    if not os.path.exists('./unsigned'):
        os.makedirs('./unsigned')
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'./unsigned/unknown_{now}.jpg'
    cv2.imwrite(filename, img)
    print(f"[저장됨] 미등록자 얼굴: {filename}")

# 변수 설정 ---①
base_dir = './faces'
min_accuracy = 85
log_file = 'visitor_log.csv'

# LBP 얼굴 인식기 및 케스케이드 얼굴 검출기 생성 및 훈련 모델 읽기 ---②
face_classifier = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(base_dir, 'all_face.xml'))

# 디렉토리 이름으로 사용자 이름과 아이디 매핑 정보 생성 ---③
dirs = [d for d in glob.glob(base_dir + "/*") if os.path.isdir(d)]
names = dict([])
for dir in dirs:
    dir = os.path.basename(dir)
    name, id = dir.split('_')
    names[int(id)] = name

# 기록된 사람 확인용 딕셔너리 (오늘 날짜 기준)
today = datetime.now().strftime('%Y-%m-%d')
recorded_today = set()

# CSV 파일에 방문자 기록 저장 함수
def log_visitor(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    entry = (name, date_str, time_str)
    
    if name not in recorded_today:
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(entry)
        recorded_today.add(name)
        print(f"[LOG] 방문자 기록됨: {entry}")

# CSV 파일 없으면 헤더 만들기
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Time'])

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("no frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (200, 200))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        label, confidence = model.predict(face_gray)
        allow_access = False

        if confidence < 400:
            accuracy = int(100 * (1 - confidence / 400))
            if accuracy >= min_accuracy:
                person_name = names[label]
                msg = f'{person_name}({accuracy}%)'
                log_visitor(person_name)
                allow_access = True
            else:
                msg = 'Unknown'
                log_visitor('Unknown')
                save_unknown_face(face)
        else:
            msg = 'Unknown'
            log_visitor('Unknown')
            save_unknown_face(face)

        # 텍스트 출력 처리
        if allow_access:
            txt, base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, 1, 3)
            cv2.rectangle(frame, (x, y - base - txt[1]), (x + txt[0], y + txt[1]), (0, 255, 0), -1)
            cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            warning_msg = "접근 허가가 없습니다"
            txt, base = cv2.getTextSize(warning_msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
            center_x = x + w // 2 - txt[0] // 2
            center_y = y - 10
            cv2.rectangle(frame, (center_x - 5, center_y - txt[1] - 5),
                          (center_x + txt[0] + 5, center_y + txt[1] + 5),
                          (0, 0, 255), -1)
            cv2.putText(frame, warning_msg, (center_x, center_y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Visitor Recognition', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
