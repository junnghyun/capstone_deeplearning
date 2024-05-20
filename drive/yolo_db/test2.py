import os
import cv2
import numpy as np
from pathlib import Path
from shutil import move
from ultralytics import YOLO

# 경로 설정
input_folder = "./ex/input_images"
processed_folder = "./ex/processed_images"
output_folder = "./ex/output_images"
no_vehicle_folder = "./ex/no_vehicle_images"

# 폴더가 존재하지 않으면 생성
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(no_vehicle_folder, exist_ok=True)

# YOLO 모델 로드
model = YOLO("./yolov8n.pt")

def process_and_move_images():
    vehicle_detected = True
    while vehicle_detected:
        vehicle_detected = False

        # 입력 폴더에서 모든 이미지 파일 가져오기
        images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in images:
            image_path = os.path.join(input_folder, image_file)
            img = cv2.imread(image_path)

            # YOLO를 사용하여 차량 탐지
            results = model(img)
            vehicle_boxes = []
            for detection in results.xyxy[0]:
                if int(detection[5]) == 2 and detection[4] > 0.5:  # 클래스 ID와 신뢰도 점수
                    vehicle_detected = True
                    vehicle_boxes.append(detection[:4].astype(int))  # 탐지된 차량의 박스 좌표 저장

            if vehicle_boxes:
                # 차량이 탐지된 경우: 차량 삭제 및 이미지 편집
                for box in vehicle_boxes:
                    x1, y1, x2, y2 = box
                    img[y1:y2, x1:x2] = 0  # 삭제된 부분을 검정색으로 채움

                # 처리된 이미지를 저장
                processed_image_path = os.path.join(processed_folder, image_file)
                cv2.imwrite(processed_image_path, img)

            else:
                # 차량이 탐지되지 않은 경우: 이미지 이동
                no_vehicle_image_path = os.path.join(no_vehicle_folder, image_file)
                move(image_path, no_vehicle_image_path)  # 원본 이미지를 이동

        # 모든 이미지가 처리되었거나 탐지된 차량이 없으면 루프 종료
        if not vehicle_detected or not images:
            break

# 이미지 처리 함수 호출
process_and_move_images()
