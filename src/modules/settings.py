import os
from src.data.brain import BrainDataset
from src.data.ham10000 import HAM10000Dataset
from src.data.isic2019 import ISICDataset
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = '../../dataset/'
DATA_PATH2 = '../dataset/'
IMG_PATH = os.getenv('IMG_PATH')
LABEL_PATH = os.getenv('LBL_PATH')

chosen_dataset = os.getenv('DATASET')
if chosen_dataset == 'brain':
    DATASET = BrainDataset(IMG_PATH, LABEL_PATH)
elif chosen_dataset == 'isic2019':
    DATASET = ISICDataset(IMG_PATH, LABEL_PATH)
elif chosen_dataset == 'ham10000':
    DATASET = HAM10000Dataset(IMG_PATH, LABEL_PATH)
else:
    DATASET = BrainDataset(IMG_PATH, LABEL_PATH)
