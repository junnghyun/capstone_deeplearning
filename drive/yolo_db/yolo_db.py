import datetime
import cv2
import numpy as np
import json
import psycopg2
from datetime import datetime
from ultralytics import YOLO

# 데이터베이스 연결 및 테이블 생성
def init_db():
    conn = psycopg2.connect(host="localhost", port=5432, database="nsu_db", user="postgres", password="postgis")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS detection_info
                      (id SERIAL PRIMARY KEY, image_file BYTEA, detected_objects JSON, rule_type TEXT, detected_count INTEGER, detection_date DATE, coordinates TEXT)''')
    conn.commit()
    conn.close()

# 탐지된 정보를 데이터베이스에 저장
def insert_detection_info(image_name, detected_objects, rule_type, detected_count, detection_date, coordinates):
    conn = psycopg2.connect(host="localhost", port=5432 database="nsu_db", user="postgres", password="postgis")
    cursor = conn.cursor()
    with open(image_path, 'rb') as file:
        binary_image = file.read()
    cursor.execute("INSERT INTO detection_info (image_file, detected_objects, rule_type, detected_count, detection_date, coordinates) VALUES (%s, %s, %s, %s, %s, %s)",
                   (image_name, json.dumps(detected_objects), rule_type, detected_count, detection_date, coordinates))
    conn.commit()
    conn.close()

# YOLO를 이용한 객체 탐지 및 데이터베이스 저장
def detect_and_store(image_path, rule_type):
    model = YOLO('yolov8')  # 균열 탐지 모델

    # 이미지 로드 및 객체 탐지
    img =cv2.imread(image_path)
    results = model(img)
    # ...
    detected_objects = []  # 탐지된 객체 정보를 저장하는 리스트
    coordinates = ""  # 좌표 정보 문자열

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                w = int(detection[2] * img.shape[1])
                h = int(detection[3] * img.shape[0])

                detected_objects.append({"class_id": class_id, "confidence": float(confidence), "x": center_x, "y": center_y, "width": w, "height": h})
                coordinates += f"({center_x},{center_y}),"

    # 데이터베이스에 저장
    detected_count = len(detected_objects)  # 탐지된 객체 수
    detection_date = datetime.now().date()  # 현재 날짜
    # 데이터베이스에 정보 저장
    insert_detection_info(image_path.split('/')[-1], detected_objects, rule_type, detected_count, detection_date, coordinates[:-1])

if __name__ == "__main__":
    init_db()
    detect_and_store("your_image.jpg", "rule_type_here")  # 이미지 경로와 규칙 종류를 지정하세요.
