import yaml
from pathlib import Path
import glob

data = {
    "train": [],
    "val": [],
    "test": [],
    "names": {0: 'pothole', 1: 'crack'}
}

base_path = Path('/tld_sample')

# 데이터 검증 및 텍스트 파일 내용 읽기
for key in ['train', 'val', 'test']:
    dir_path = base_path / key
    if not dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    # 해당 디렉토리의 모든 .txt 파일 읽기
    text_files = glob.glob(str(dir_path / '*.txt'))
    for filename in text_files:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            data[key].append(content)  # 파일 내용을 리스트에 추가

if len(data['names']) != len(set(data['names'].values())):
    raise ValueError("Duplicate entries found in 'names'")

file_path = Path('.tld.yaml')

# YAML 파일로 저장
with file_path.open('w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# YAML 파일 읽기
with file_path.open('r') as f:
    loaded_data = yaml.safe_load(f)
    print(loaded_data)
