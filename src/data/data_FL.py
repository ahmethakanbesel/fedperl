import numpy as np
from torch.utils.data import Dataset
from src.modules.settings import DATASET


class MyDataset(Dataset):
    def __init__(self, x, y, transform=None, rand_aug=None):
        self.X = x
        self.y = y
        self.transform = transform
        self.RandAug = rand_aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load data and get label
        img = self.X[index]
        label = self.y[index]
        img2 = self.X[index]
        if self.RandAug:
            img2 = self.RandAug(img2)
        if self.transform:
            img = self.transform(img)
        return img, label, img2


class Data:
    def __init__(self, data_path, clients_path):
        self.data_path = data_path
        self.clients = clients_path
        self.dataset = DATASET
        self.dataset.clients_path = self.data_path + f'/clients/'

    def load_server(self):
        img_tr_l, lbl_tr_l = self.dataset.get_server_data()
        missing_class_weight = 100
        id, counts = np.unique(lbl_tr_l, return_counts=True)
        med_freq = np.median(counts)
        lbl_weights = {}
        cK = 0
        for k in id:
            lbl_weights[k] = med_freq / counts[cK]
            cK += 1

        # max_class_id = np.max(lbl_weights.keys())+1
        weights = {}
        for k in range(self.dataset.num_classes):
            if k in lbl_weights.keys():
                weights[k] = lbl_weights[k]
            else:
                weights[k] = missing_class_weight

        val_ds = MyDataset(img_tr_l, lbl_tr_l, self.dataset.get_validation_transform())
        return val_ds, list(weights.values())

    def load_clients_test_val(self, client_id):
        img_t, lbl_t, img_v, lbl_v = self.dataset.get_client_test_val_data(client_id)

        test_ds = MyDataset(img_t, lbl_t, self.dataset.get_validation_transform())
        val_ds = MyDataset(img_v, lbl_v, self.dataset.get_validation_transform())
        return test_ds, val_ds

    def load_clients_ssl(self, client_id):
        images, labels, idx_l, idx_u, idx_v = self.dataset.get_client_data(client_id)

        img_tr_l = images[idx_l]
        lbl_tr_l = labels[idx_l]

        img_tr_u = images[idx_u]
        lbl_tr_u = labels[idx_u]

        img_tr_v = images[idx_v]
        lbl_tr_v = labels[idx_v]

        missing_class_weight = 100
        id, counts = np.unique(lbl_tr_l, return_counts=True)
        med_freq = np.median(counts)
        lbl_weights = {}
        cK = 0
        for k in id:
            lbl_weights[k] = med_freq / counts[cK]
            cK += 1

        # max_class_id = np.max(lbl_weights.keys())+1
        weights = {}
        for k in range(self.dataset.num_classes):
            if k in lbl_weights.keys():
                weights[k] = lbl_weights[k]
            else:
                weights[k] = missing_class_weight

        train_ds = MyDataset(img_tr_l, lbl_tr_l, self.dataset.get_labeled_transform())
        train_uds = MyDataset(img_tr_u, lbl_tr_u, self.dataset.get_labeled_transform(),
                              self.dataset.get_unlabeled_transform())
        val_ds = MyDataset(img_tr_v, lbl_tr_v, self.dataset.get_validation_transform())
        return train_ds, train_uds, val_ds, list(weights.values())

    def load_clients_lower(self, client_id):
        if client_id in (0, 1, 2, 3, 4):
            img = np.load(self.data_path + 'ISIC19_IMG_224.npy')
            lbl = np.load(self.data_path + 'ISIC19_LBL_224.npy')
        elif client_id in (6, 7, 5):
            img = np.load(self.data_path + 'HAM_IMG_224.npy')
            lbl = np.load(self.data_path + 'HAM_LBL_224.npy')
        elif client_id == 8:
            img = np.load(self.data_path + 'PAD_IMG_224.npy')
            lbl = np.load(self.data_path + 'PAD_LBL_224.npy')
        elif client_id == 9:
            img = np.load(self.data_path + 'DERM_IMG_224.npy')
            lbl = np.load(self.data_path + 'DERM_LBL_224.npy')

        idx_l = np.load(self.clients + 'client' + str(client_id) + 'L.npy')
        idx_v = np.load(self.clients + 'client' + str(client_id) + 'V.npy')

        img_tr_l = img[idx_l]
        lbl_tr_l = lbl[idx_l]

        img_v = img[idx_v]
        lbl_v = lbl[idx_v]

        missing_class_weight = 100
        id, counts = np.unique(lbl_tr_l, return_counts=True)
        med_freq = np.median(counts)
        lbl_weights = {}
        cK = 0
        for k in id:
            lbl_weights[k] = med_freq / counts[cK]
            cK += 1

        weights = {}
        for k in range(8):
            if k in lbl_weights.keys():
                weights[k] = lbl_weights[k]
            else:
                weights[k] = missing_class_weight

        train_ds = MyDataset(img_tr_l, lbl_tr_l, self.dataset.get_labeled_transform())
        val_ds = MyDataset(img_v, lbl_v, self.dataset.get_validation_transform())
        return train_ds, val_ds, list(weights.values())

    def load_clients_upper(self, client_id):
        if client_id in (0, 1, 2, 3, 4):
            img = np.load(self.data_path + 'ISIC19_IMG_224.npy')
            lbl = np.load(self.data_path + 'ISIC19_LBL_224.npy')
        elif client_id in (6, 7, 5):
            img = np.load(self.data_path + 'HAM_IMG_224.npy')
            lbl = np.load(self.data_path + 'HAM_LBL_224.npy')
        elif client_id == 8:
            img = np.load(self.data_path + 'PAD_IMG_224.npy')
            lbl = np.load(self.data_path + 'PAD_LBL_224.npy')
        elif client_id == 9:
            img = np.load(self.data_path + 'DERM_IMG_224.npy')
            lbl = np.load(self.data_path + 'DERM_LBL_224.npy')

        idx_l = np.load(self.clients + 'client' + str(client_id) + 'L.npy')
        idx_u = np.load(self.clients + 'client' + str(client_id) + 'U.npy')
        idx_v = np.load(self.clients + 'client' + str(client_id) + 'V.npy')
        idx_l = np.concatenate((idx_l, idx_u))

        img_tr_l = img[idx_l]
        lbl_tr_l = lbl[idx_l]
        img_v = img[idx_v]
        lbl_v = lbl[idx_v]

        missing_class_weight = 100
        id, counts = np.unique(lbl_tr_l, return_counts=True)
        med_freq = np.median(counts)
        lbl_weights = {}
        cK = 0
        for k in id:
            lbl_weights[k] = med_freq / counts[cK]
            cK += 1

        # max_class_id = np.max(lbl_weights.keys())+1
        weights = {}
        for k in range(8):
            if k in lbl_weights.keys():
                weights[k] = lbl_weights[k]
            else:
                weights[k] = missing_class_weight

        train_ds = MyDataset(img_tr_l, lbl_tr_l, self.dataset.get_labeled_transform())
        val_ds = MyDataset(img_v, lbl_v, self.dataset.get_validation_transform())
        return train_ds, val_ds, list(weights.values())
