import cv2
import numpy as np
import os, glob
import csv
import webbrowser
from datetime import datetime

# 미등록자 얼굴 저장 함수
def save_unknown_face(img):
    if not os.path.exists('./unsigned'):
        os.makedirs('./unsigned')
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'./unsigned/unknown_{now}.jpg'
    cv2.imwrite(filename, img)
    print(f"[저장됨] 미등록자 얼굴: {filename}")

# 변수 설정
base_dir = './faces'
min_accuracy = 85
log_file = 'visitor_log.csv'
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1cFewCY-mAs0gRMiv9RikLlRmFap6Mfl_yvUAFb6sfxE/edit?pli=1&gid=1649354833#gid=1649354833'

# LBP 얼굴 인식기 및 학습 모델
face_classifier = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
model = cv2.face.LBPHFaceRecognizer_create()
model.read(os.path.join(base_dir, 'all_face.xml'))

# 이름과 ID 매핑
dirs = [d for d in glob.glob(base_dir + "/*") if os.path.isdir(d)]
names = dict([])
for dir in dirs:
    dir = os.path.basename(dir)
    name, id = dir.split('_')
    names[int(id)] = name

# 오늘 날짜 기준으로 방문자 기록 중복 방지
today = datetime.now().strftime('%Y-%m-%d')
recorded_today = set()

# 방문자 기록 함수
def log_visitor(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    entry = (name, date_str, time_str)

    if name == 'Unknown':
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(entry)
        print(f"[LOG] 미등록자 기록됨: {entry}")
    elif name not in recorded_today:
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(entry)
        recorded_today.add(name)
        print(f"[LOG] 등록자 기록됨: {entry}")
        webbrowser.open(spreadsheet_url, new=2)

# 로그 파일이 없으면 헤더 작성
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Time'])

# 카메라 시작
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
                # 카메라 종료
                cap.release()
                cv2.destroyAllWindows()
                exit()
            else:
                msg = 'Unknown'
                log_visitor('Unknown')
                save_unknown_face(face)
                # 카메라 종료
                cap.release()
                cv2.destroyAllWindows()
                exit()
        else:
            msg = 'Unknown'
            log_visitor('Unknown')
            save_unknown_face(face)
            # 카메라 종료
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow('Visitor Recognition', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
