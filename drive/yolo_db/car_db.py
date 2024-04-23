import cv2
import numpy as np
import psycopg2
from ultralytics import YOLO

# 데이터베이스 연결 설정
conn_db1 = psycopg2.connect(host="localhost", port=5432, database="db1", user="postgres", password="postgis")
conn_db2 = psycopg2.connect(host="localhost", port=5432, database="db2", user="postgres", password="postgis")
cur_db1 = conn_db1.cursor()
cur_db2 = conn_db2.cursor()

# YOLO 모델 로드
model = YOLO("./pretrained/yolov8n.pt")

def process_and_move_images():
    vehicle_detected = True  # 초기값으로 차량이 탐지되었다고 설정
    while vehicle_detected:  # 루프 시작
        # DB1에서 모든 이미지 레코드를 가져옴
        cur_db1.execute("SELECT id, image_file, coordinates FROM db1_table")
        images = cur_db1.fetchall()
        vehicle_detected = False  # 이미지 처리를 시작하기 전에 변수를 False로 설정

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
            vehicle_detected_image = False  # 이미지에 차량이 탐지되었는지 여부
            vehicle_boxes = []  # 탐지된 차량의 박스 좌표를 저장할 리스트
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 2 and confidence > 0.5:  # 차량 클래스 ID가 2라고 가정
                        vehicle_detected_image = True
                        vehicle_detected = True  # 루프를 계속하기 위해 전체 이미지에 대한 탐지 여부를 업데이트
                        # 탐지된 차량의 박스 좌표 저장
                        center_x = int(detection[0] * img.shape[1])
                        center_y = int(detection[1] * img.shape[0])
                        w = int(detection[2] * img.shape[1])
                        h = int(detection[3] * img.shape[0])
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        vehicle_boxes.append((x, y, x + w, y + h))

            if vehicle_detected_image:
                # 차량이 탐지된 경우: 차량 삭제 및 이미지 편집
                for box in vehicle_boxes:
                    x1, y1, x2, y2 = box
                    # 차량 박스 삭제
                    img[y1:y2, x1:x2] = 0  # 삭제된 부분을 검정색으로 채움

                    # DB1에서 같은 좌표에 해당하는 이미지 가져오기
                    cur_db1.execute("SELECT image_file FROM db1_table WHERE coordinates = %s", (coordinates,))
                    other_image_data = cur_db1.fetchone()[0]
                    nparr_other = np.frombuffer(other_image_data, np.uint8)
                    other_image = cv2.imdecode(nparr_other, cv2.IMREAD_COLOR)

                    # 삭제된 부분에 다른 이미지 삽입
                    img[y1:y2, x1:x2] = other_image[y1:y2, x1:x2]

                    # DB1에서 해당 이미지 삭제
                    cur_db1.execute("DELETE FROM db1_table WHERE id = %s", (id,))
                    conn_db1.commit()

            else:
                # 차량이 탐지되지 않은 경우: DB2로 이미지 이동
                cur_db2.execute("INSERT INTO db2_table (id, image_file, coordinates) VALUES (%s, %s, %s)",
                                (id, image_data, coordinates))
                conn_db2.commit()
                # 같은 좌표의 이미지를 DB1에서 삭제
                cur_db1.execute("DELETE FROM db1_table WHERE coordinates = %s", (coordinates,))
                conn_db1.commit()

                # 이미지를 DB2로 이동했으므로 DB1에서 해당 이미지 삭제
                cur_db1.execute("DELETE FROM db1_table WHERE id = %s", (id,))
                conn_db1.commit()

                # 이미지를 편집하고 남은 0 값을 다시 채워넣기 위해 처음 이미지를 가져와서 사용
                cur_db1.execute("SELECT image_file FROM db1_table WHERE coordinates = %s ORDER BY id ASC LIMIT 1", (coordinates,))
                initial_image_data = cur_db1.fetchone()[0]
                nparr_initial = np.frombuffer(initial_image_data, np.uint8)
                initial_image = cv2.imdecode(nparr_initial, cv2.IMREAD_COLOR)

                # 편집된 이미지에서 0 값의 위치를 처음 이미지에서 해당 위치로 채움
                zero_indices = np.where((img == [0, 0, 0]).all(axis=2))  # 편집된 이미지에서 0 값의 위치 찾기
                img[zero_indices] = initial_image[zero_indices]  # 처음 이미지에서 해당 위치의 값으로 채움

        # 이미지 처리가 끝나면 다음 이미지를 가져오기 위해 루프를 반복

# 이미지 처리 함수 호출
process_and_move_images()

# 데이터베이스 연결 종료
cur_db1.close()
conn_db1.close()
cur_db2.close()
conn_db2.close()
