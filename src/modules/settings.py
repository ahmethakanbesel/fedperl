from src.data.brain import BrainDataset

DATA_PATH = '../../dataset/'
DATASET = BrainDataset(DATA_PATH + '/dataset_img.npy', DATA_PATH + '/dataset_lbl.npy')
