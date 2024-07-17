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



# PyTorch의 Dataset 클래스를 상속받아 데이터 로딩과 관련된 과정을 관리하는 클래스
class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc  # 오디오 파일에서 추출한 MFCC 특징들의 리스트
        self.label = label  # 해당 오디오 파일의 레이블 리스트, 레이블이 없는 경우 None이 될 수 있음

    def __len__(self):
        return len(self.mfcc)  # 데이터셋의 총 데이터 개수를 반환, 주어진 MFCC 리스트의 길이와 동일함
    
    # 주어진 인덱스에 해당하는 데이터를 데이터셋에서 불러와 반환, 레이블이 있는 경우 MFCC와 레이블을 함께 반환하고, 없는 경우 MFCC만 반환
    def __getitem__(self, index):
        if self.label is not None:
            return self.mfcc[index], self.label[index]
        return self.mfcc[index]
    

# DataLoader는 구출된 데이터셋에서 배치크기(batch_size)에 맞게 데이터를 추출하고, 필요에 따라 섞거나(shuffle=True) 순서대로 반환(shuffle=False)하는 역할을 한다.
train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)  # 훈련 데이터
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)  # 검증 데이터



# MLP 모델은 여러 층의 노드로 구성되며 각 노드를 입력값과 가중치를 통해 계산된 값을 출력으로 전달한다. 일반적으로 입력층, 은닉층, 출력층으로 구성된다.
class MLP(nn.Module):
    # input_dim : MFCC의 개수
    # hidden_dim : 은닉층의 차원 수
    # output_dim : 분류할 클래스의 수
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    # 입력 데이터를 순차적으로 세 개의 선형 계층과 ReLU 활성화 함수를 거쳐 최종적으로 시그모이드 함수를 적용하여 출력 확률을 계산
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    


from sklearn.metrics import roc_auc_score

# 모델을 훈련하는 주요 함수
def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(features)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val AUC : [{_val_score:.5f}]')
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    
    return best_model

# 각 클래스에 대해 AUC 점수의 평균을 계산하여 반환하는 함수
def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score

# 모델의 검증 과정을 처리하는 함수
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score



