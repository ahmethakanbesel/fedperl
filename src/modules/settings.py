import os
from src.data.brain import BrainDataset

DATA_PATH = '../dataset/'
IMG_PATH = os.path.abspath(DATA_PATH + 'brain_dataset_img.npy')
LABEL_PATH = os.path.abspath(DATA_PATH + 'brain_dataset_lbl.npy')
DATASET = BrainDataset(IMG_PATH, LABEL_PATH)
