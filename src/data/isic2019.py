import numpy as np

from src.data.dataset import Dataset
from src.utils.randaug import RandAugment
from torchvision import transforms


class ISICDataset(Dataset):
    def __init__(self, image_file, label_file):
        self.image_file = image_file
        self.label_file = label_file
        self.classes = ['mel', 'nv', 'bcc', 'ak', 'bkl', 'df', 'vasc', 'scc']
        self.num_classes = len(self.classes)
        self.class_idx = [i for i in range(self.num_classes)]
        self.label_map = {}
        for i in range(self.num_classes):
            self.label_map[self.classes[i]] = i
        self.clients_path = None
        self.images = np.load(self.image_file)
        self.labels = np.load(self.label_file)
        self.distribution = self.__get_client_image_ids_20L_80U

    def get_classes(self):
        return self.classes

    def get_server_data(self):
        images, labels = self.__get_images_labels()
        idx = np.array([0])
        return images[idx], labels[idx]

    def get_client_data(self, client_id):
        images, labels = self.__get_images_labels()
        idx_l, idx_u, idx_v = self.distribution(client_id)

        return images, labels, idx_l, idx_u, idx_v

    def get_client_data_counts(self, client_id):
        idx_l, idx_u, idx_v = self.distribution(client_id)

        return len(idx_l), len(idx_u), len(idx_v)

    def get_client_class_distribution(self, client_id):
        client_classes = {}
        for c in range(len(self.classes)):
            client_classes[c] = 0

        images, labels, idx_l, idx_u, idx_v = self.get_client_data(client_id)

        for l in idx_l:
            client_classes[labels[l]] += 1

        for u in idx_u:
            client_classes[labels[u]] += 1

        return client_classes

    def get_client_test_val_data(self, client_id):
        img_t = np.load(self.clients_path + f'client-{str(client_id)}-U_img.npy')
        lbl_t = np.load(self.clients_path + f'client-{str(client_id)}-U_lbl.npy')

        img_v = np.load(self.clients_path + f'client-{str(client_id)}-V_img.npy')
        lbl_v = np.load(self.clients_path + f'client-{str(client_id)}-V_lbl.npy')
        return img_t, lbl_t, img_v, lbl_v

    def get_global_test_data(self):
        images, labels = self.__get_images_labels()

        start_idx = 20000
        test = [i for i in range(start_idx, start_idx + 5000)]

        start_idx = 25000
        validation = [i for i in range(start_idx, start_idx + 330)]

        return images, labels, test, validation

    def get_global_test_data_distribution(self):
        classes = {}
        for c in range(len(self.classes)):
            classes[c] = 0

        images, labels, idx_t, idx_v = self.get_global_test_data()

        for t in idx_t:
            classes[labels[t]] += 1

        return classes

    def __get_images_labels(self):
        images, labels = self.images, self.labels
        if images is None:
            images = np.load(self.image_file)
        if labels is None:
            labels = np.load(self.label_file)
        return images, labels

    def __get_client_image_ids_20L_80U(self, client_id):
        labeled, unlabeled, validation = [], [], []
        # 100 val, 2000 training = 2100
        start_idx = 2100 * client_id
        # Pick first 100 as validation
        # validation = [i for i in range(start_idx, start_idx + 101)]
        # start_idx = start_idx + 100
        unlabeled = [i for i in range(start_idx, start_idx + 1601)]
        start_idx = start_idx + 1600
        labeled = [i for i in range(start_idx, start_idx + 401)]
        start_idx = 25000
        validation = [i for i in range(start_idx, start_idx + 331)]
        return labeled, unlabeled, validation

    def __get_client_image_ids_80L_20U(self, client_id):
        labeled, unlabeled, validation = [], [], []
        # 100 val, 2000 training = 2100
        start_idx = 2100 * client_id
        # Pick first 100 as validation
        validation = [i for i in range(start_idx, start_idx + 101)]
        start_idx = start_idx + 100
        labeled = [i for i in range(start_idx, start_idx + 1601)]
        start_idx = start_idx + 1600
        unlabeled = [i for i in range(start_idx, start_idx + 401)]
        return labeled, unlabeled, validation

    def __get_client_image_ids_2labeled(self, client_id):
        labeled, unlabeled, validation = [], [], []
        # 100 val, 2000 training = 2100
        start_idx = 2100 * client_id
        # Pick first 100 as validation
        validation = [i for i in range(start_idx, start_idx + 101)]
        start_idx = start_idx + 100
        if client_id < 2:  # 2 labeled clients
            labeled = [i for i in range(start_idx, start_idx + 2001)]
            unlabeled = [start_idx]
        else:
            labeled = [start_idx]
            unlabeled = [i for i in range(start_idx, start_idx + 2001)]
        return labeled, unlabeled, validation

    def get_labeled_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        ])

    def get_unlabeled_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        ])

    def get_validation_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        ])
