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



# 데이터 전처리 수행 과정들 아래로 쭉

# 모델을 훈련하기 전에 전체 데이터 세트를 두 개의 서브셋으로 나눠준다. 학습 데이터 세트와 검증 데이터 세트
df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)


# MRCC 특징을 추출하고, 이를 학습에 사용할 형식으로 변환하는 함수
def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        # Librosa 패키지를 사용하여 wav 파일 Load
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)  # row['path']에 해당하는 오디오 파일 로드, 샘플링 레이트는 CONFIG.SR로 지정됨

        # Librosa 패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)  # 오디오 신호 y로부터 MFXX 특징을 추출함, CONFIG.N_MFCC는 추출할 MFCC 계수를 지정함
        # 추출된 MFCC는 프레임별로 계산되므로, 각 프레임의 평균값을 구하여 전체 파일에 대한 MFCC 특징을 대표하는 벡터를 얻는다.
        mfcc = np.mean(mfcc.T, axis=0)
        features.append(mfcc)

        # train_mode = True인 경우, 현재 행의 레이블을 읽어와 CONFIG.N_CLASSES 길이의 벡터로 변환한다.
        # 레이블이 'fake'이면 첫 번째 원소를 1로, 'real'이면 두 번째 원소를 1로 설정한다.
        # 이 벡터를 labels 리스트에 추가한다.
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features

train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)