import torch
from ultralytics import YOLO
import os

# 학습 설정
model = YOLO('yolov8n.pt')

# 체크포인트와 로그를 저장할 디렉토리 생성
checkpoint_dir = './checkpoints'
log_dir = './logs'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 학습 설정
try:
    model.train(data='./tld.yaml',
                epochs=50,  # 에폭 수
                batch_size=16,  # 배치 크기
                imgsz=640,  # 이미지 크기
                device='cuda',  # 학습에 사용할 디바이스 ('cuda' or "cpu")
                workers=8,  # 데이터 로딩을 위한 워커 수
                project=checkpoint_dir, # 체크 포인트 저장 경로
                name='yolov8_training', # 학습 세션 이름
                exist_ok=True)  # 이전 학습 세션 덮어쓰기 허용
    torch.save(model.state_dict(), 'yolov8_model.pt')
    print("모델 학습 및 저장이 성공적으로 완료되었습니다.")
except Exception as e:
    print(f"학습 중 오류가 발생했습니다: {e}")

    