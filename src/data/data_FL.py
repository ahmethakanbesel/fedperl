import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from utils.randaug import RandAugment

# Define transforms for dataset
train_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomRotation(20),
     transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
     ])

trainU_transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3),
     RandAugment(),
     transforms.ToTensor(),
     transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
     ])

val_transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])


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


class SkinData:
    def __init__(self, data_path, clients_path):

        self.data_path = data_path  # '/home/tariq/code/UsedData/SKIN_IMG_Numpy/'
        self.clients = clients_path  # '/home/tariq/code/UsedData/SKIN_IMG_Numpy/clients/'

    def load_isic20(self):
        img = np.load(self.data_path + 'dataset_img.npy')
        lbl = np.load(self.data_path + 'dataset_lbl.npy')
        test_ds = MyDataset(img, lbl, val_transform)
        return test_ds

    def load_server(self):
        img = np.load(self.data_path + 'dataset_img.npy')
        lbl = np.load(self.data_path + 'dataset_lbl.npy')
        # idx = np.load(self.clients + 'server.npy')
        idx = np.array([])
        img_tr_l = img[idx]
        lbl_tr_l = lbl[idx]

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

        val_ds = MyDataset(img_tr_l, lbl_tr_l, val_transform)
        return val_ds, list(weights.values())

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

        train_ds = MyDataset(img_tr_l, lbl_tr_l, train_transform)
        val_ds = MyDataset(img_v, lbl_v, val_transform)
        return train_ds, val_ds, list(weights.values())

    # TODO: Load dataset properly
    def load_clients_test_val(self, client_id):
        img_t = np.load(self.data_path + f'/clients/client-{str(client_id)}-U_img.npy')
        lbl_t = np.load(self.data_path + f'/clients/client-{str(client_id)}-U_lbl.npy')

        img_v = np.load(self.data_path + f'/clients/client-{str(client_id)}-V_img.npy')
        lbl_v = np.load(self.data_path + f'/clients/client-{str(client_id)}-V_lbl.npy')

        test_ds = MyDataset(img_t, lbl_t, val_transform)
        val_ds = MyDataset(img_v, lbl_v, val_transform)
        return test_ds, val_ds

    def get_client_image_ids(self, client_id):
        labeled, unlabeled, validation = [], [], []
        # 100 val, 2000 training = 2100
        start_idx = 2100 * client_id
        # Pick first 100 as validation
        validation = [i for i in range(start_idx, start_idx + 101)]
        start_idx = start_idx + 100
        if client_id < 2:  # 2 labeled clients
            labeled = [i for i in range(start_idx, start_idx + 2001)]
        else:
            unlabeled = [i for i in range(start_idx, start_idx + 2001)]
        return labeled, unlabeled, validation

    # TODO: Load dataset properly
    def load_clients_ssl(self, client_id):
        images = np.load(self.data_path + f'/dataset_img.npy')
        labels = np.load(self.data_path + f'/dataset_lbl.npy')

        idx_l, idx_u, idx_v = self.get_client_image_ids(client_id)

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
        for k in range(8):
            if k in lbl_weights.keys():
                weights[k] = lbl_weights[k]
            else:
                weights[k] = missing_class_weight

        train_ds = MyDataset(img_tr_l, lbl_tr_l, train_transform)
        train_uds = MyDataset(img_tr_u, lbl_tr_u, train_transform, trainU_transform)
        val_ds = MyDataset(img_tr_v, lbl_tr_v, val_transform)
        return train_ds, train_uds, val_ds, list(weights.values())

    def load_clients_ssl_client(self, client_id):
        img_tr_l = np.load(self.data_path + f'/clients/client-{str(client_id)}-L_img.npy')
        lbl_tr_l = np.load(self.data_path + f'/clients/client-{str(client_id)}-L_lbl.npy')

        img_tr_u = np.load(self.data_path + f'/clients/client-{str(client_id)}-U_img.npy')
        lbl_tr_u = np.load(self.data_path + f'/clients/client-{str(client_id)}-U_lbl.npy')

        img_tr_v = np.load(self.data_path + f'/clients/client-{str(client_id)}-V_img.npy')
        lbl_tr_v = np.load(self.data_path + f'/clients/client-{str(client_id)}-V_lbl.npy')

        print(len(lbl_tr_v), len(lbl_tr_l), len(lbl_tr_u))

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

        train_ds = MyDataset(img_tr_l, lbl_tr_l, train_transform)
        train_uds = MyDataset(img_tr_u, lbl_tr_u, train_transform, trainU_transform)
        val_ds = MyDataset(img_tr_v, lbl_tr_v, val_transform)
        return train_ds, train_uds, val_ds, list(weights.values())

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

        train_ds = MyDataset(img_tr_l, lbl_tr_l, train_transform)
        val_ds = MyDataset(img_v, lbl_v, val_transform)
        return train_ds, val_ds, list(weights.values())
