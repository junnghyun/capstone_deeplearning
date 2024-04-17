import cv2
import numpy as np
import psycopg2
from ultralytics import YOLO

# 데이터베이스 연결 설정
conn_db1 = psycopg2.connect("dbname=db1 user=your_username password=your_password")
conn_db2 = psycopg2.connect("dbname=db2 user=your_username password=your_password")
cur_db1 = conn_db1.cursor()
cur_db2 = conn_db2.cursor()

# YOLO 모델 로드
model = YOLO("./pretrained/yolov8n.pt")

def process_and_move_images():
    # DB1에서 모든 이미지 레코드를 가져옴
    cur_db1.execute("SELECT id, image_file, coordinates FROM db1_table")
    images = cur_db1.fetchall()

    for img_record in images:
        id, image_data, coordinates = img_record
        # 바이너리 데이터로부터 이미지 읽기
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # YOLO를 사용하여 차량 탐지
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        outs = model.forward(output_layers)

        # 차량 탐지 여부 확인
        vehicle_detected = False
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 2 and confidence > 0.5:  # 차량 클래스 ID가 2라고 가정
                    vehicle_detected = True
                    break

        if vehicle_detected:
            # 차량이 탐지된 경우: 차량 삭제 및 이미지 편집 로직 구현
            pass  # 이미지 편집 로직 추가
        else:
            # 차량이 탐지되지 않은 경우: DB2로 이미지 이동
            cur_db2.execute("INSERT INTO db2_table (id, image_file, coordinates) VALUES (%s, %s, %s)",
                            (id, image_data, coordinates))
            conn_db2.commit()
            # 같은 좌표의 이미지를 DB1에서 삭제
            cur_db1.execute("DELETE FROM db1_table WHERE coordinates = %s", (coordinates,))
            conn_db1.commit()

process_and_move_images()

# 데이터베이스 연결 종료
cur_db1.close()
conn_db1.close()
cur_db2.close()
conn_db2.close()
