import librosa  # 오디오 및 음악 신호 분석을 위한 패키지, 오디오 파일 로드, 특성 추출 등의 기능 제공

from sklearn.model_selection import train_test_split
import numpy as np  # 고성능의 다차원 배열 객체와 이를 다루기 위한 도구 제공, 수학적.과학적 연산을 위한 광범위한 함수 포함, 데이터 전처리, 배열 변환 등에 사용됨
import pandas as pd  # 데이터 조작 및 분석을 위한 강력한 데이터 구조 제공, 다양한 데이터 포맷의 읽기 및 쓰기를 지원함 (CSV, Excel 등), 데이터 클리닝, 탐색, 필터링, 집계 등의 작업에 이상적
import random

from torch import nn  # 신경망 모듈로 딥러닝 모델을 구축하는 데 필요한 다양한 레이어와 함수를 포함
import torch.nn.functional as F  # 활성화 함수, 손실 함수 등 다양한 연산을 함수 형태로 제공
from torch.utils.data import Dataset, DataLoader  # 데이터 로딩 및 전처리를 위한 유틸리티 제공, 데이터 셋을 정의하고, 배치 단위로 불러오고, 이터레이션하는 데 사용
from tqdm import tqdm  # 반복문에서 진행률을 표현해주며, 코드 실행 상태를 쉽게 확인 가능

import torch  # 다차원 배열은 tensor를 조작하기 위한 함수와 도구 제공
import torchmetrics  # pytorch 모델의 성능을 평가하기 위함
import os  # 파일 시스템을 탐색하고, 파일 및 디렉토리를 관리하며, 운영체제의 환경 변수를 접근하는 등의 작업을 수행 가능
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class Config:
    SR = 32000  # 오디오 데이터의 샘플링 레이트를 설정, 오디오 데이터의 초당 샘플 수를 정의
    N_MFCC = 13  # 멜 주파수 캡스트럼 계수의 개수를 의미, 오디오 신호의 주파수 특성을 인간의 청각 특성에 맞게 변환한 것
    # Dataset
    ROOT_FOLDER = './'  # 데이터셋의 루트 폴더 경로를 설정
    #Training
    N_CLASSES = 2  # 분류할 클래스의 수를 설정, 모델의 출력 차원을 설정할 때 사용
    BATCH_SIZE = 96  # 배치 크기를 설정, 학습 시 한 번에 처리할 데이터 샘플의 수를 정의
    N_EPOCHS = 5  # 학습 에폭 수를 설정, 전체 데이터셋을 학습 횟수를 정의, 너무 적으면 과소적합, 많으면 과적합이 발생할 수 있음
    LR = 3e-4  # 학습률을 설정, 모델의 가중치를 업데이트할 때 사용되는 학습 속도를 정의, 너무 크면 학습이 불안정, 작으면 학습 속도가 느려짐
    #Others
    SEED = 42  # 재현성을 위해 SEED 값을 고정하는 SEED를 설정해줌

CONFIG = Config()



# 머신러닝이나 딥러닝 모델을 훈련할 때, 결과의 재현성을 보장하기 위해 사용되는 함수
# 다양한 랜덤 시드를 고정하여, 실행할 때마다 동일한 결과를 얻기 위해 사용
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)  # Seed 고정