import numpy as np

from src.data.dataset import Dataset
from src.utils.randaug import RandAugment
from torchvision import transforms


class ISICDataset(Dataset):
    def __init__(self, image_file, label_file):
        # Define paths of image and label files
        self.image_file = image_file
        self.label_file = label_file
        # Define classes of the dataset
        self.classes = ['mel', 'nv', 'bcc', 'ak', 'bkl', 'df', 'vasc', 'scc']
        # Count num of classes
        self.num_classes = len(self.classes)
        # Map each class with an integer between [0, n-1] where n is the number of classes
        self.class_idx = [i for i in range(self.num_classes)]
        # Create a dictionary to do quick look up for learning id of label when labels are strings
        self.label_map = {}
        for i in range(self.num_classes):
            self.label_map[self.classes[i]] = i
        # Set client dataset path if you are using separate .npy files for clients
        # By default this is unnecessary
        self.clients_path = None
        # Load images and labels to memory
        # You need to have enough free space in your ram to fit all data
        self.images = np.load(self.image_file)
        self.labels = np.load(self.label_file)
        # Set the distribution function
        # Distribution function defines how the data will be distributed among clients
        # It returns indexes of images
        # By default this dataset uses %20 labeled %80 unlabeled data distribution for each client
        # There are different functions in this file
        # for %80 labeled, %20 unlabeled and 2 labeled 8 unlabeled clients scenarios
        self.distribution = self.__get_client_image_ids_20L_80U

    def get_classes(self):
        return self.classes

    def get_server_data(self):
        """
        This function returns indexes of images located at server.
        By default the server does not store any images.
        But if you set an empty error the code won't work this is why we put one images
        @return:
        """
        images, labels = self.__get_images_labels()
        idx = np.array([0])
        return images[idx], labels[idx]

    def get_client_data(self, client_id):
        """
        This methods gets images and labels of the dataset first
        Then it gets indexes of images which will be used for labeled, unlabeled and validation
        @param client_id: integer value starting from zero
        @return:
        """
        images, labels = self.__get_images_labels()
        idx_l, idx_u, idx_v = self.distribution(client_id)

        return images, labels, idx_l, idx_u, idx_v

    def get_client_data_counts(self, client_id):
        """
        Returns number of labeled, unlabeled and validation images for the given client_id
        This is used for creating plots
        @param client_id: integer value starting from zero
        @return:
        """
        idx_l, idx_u, idx_v = self.distribution(client_id)

        return len(idx_l), len(idx_u), len(idx_v)

    def get_client_class_distribution(self, client_id):
        """
        Returns number of images belongs for each class for the given client_id
        This is used for creating plots
        @param client_id: integer value starting from zero
        @return:
        """
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
        """
        Return client data for the given client_id
        This can be used if you would like to store data of each client in separate files
        Not necessary by default
        @param client_id: integer value starting from zero
        @return:
        """
        img_t = np.load(self.clients_path + f'client-{str(client_id)}-U_img.npy')
        lbl_t = np.load(self.clients_path + f'client-{str(client_id)}-U_lbl.npy')

        img_v = np.load(self.clients_path + f'client-{str(client_id)}-V_img.npy')
        lbl_v = np.load(self.clients_path + f'client-{str(client_id)}-V_lbl.npy')
        return img_t, lbl_t, img_v, lbl_v

    def get_global_test_data(self):
        """
        Returns all images and labels of the dataset with indexes of images
        which will be used for testing and validation of the global model
        Eg. k = 100 (your dataset has 100 images)
        images[image[0], image[1], ..., image[k-1]]
        labels[label[0], label[1], ..., label[k-1]]
        You would like to use images 32 45 87 for validation and images after 87 for testing
        It should return like that
        images, labels, [32, 45, 87], [87, 88, 89, ..., 99]
        @return:
        """
        images, labels = self.__get_images_labels()

        start_idx = 20000
        test = [i for i in range(start_idx, start_idx + 5000)]

        start_idx = 25000
        validation = [i for i in range(start_idx, start_idx + 330)]

        return images, labels, test, validation

    def get_global_test_data_distribution(self):
        """
        Returns number of images for each client in global test dataset
        Used for plotting
        @return:
        """
        classes = {}
        for c in range(len(self.classes)):
            classes[c] = 0

        images, labels, idx_t, idx_v = self.get_global_test_data()

        for t in idx_t:
            classes[labels[t]] += 1

        return classes

    def __get_images_labels(self):
        """
        Checks self.images and self.labels
        If they are empty load images and labels from files
        Then returns images and labels
        @return:
        """
        images, labels = self.images, self.labels
        if images is None:
            images = np.load(self.image_file)
        if labels is None:
            labels = np.load(self.label_file)
        return images, labels

    def __get_client_image_ids_20L_80U(self, client_id):
        """
        Returns indexes of images for each client

        ISIC2019 dataset has 25,330 images
        20,000 of them used for training in our experiments
        We equally distributed 20,000 images to 10 clients
        This means each client has 2000 images

        %20 of client images will be labeled and %80 will be unlabeled

        @param client_id: integer value starting from zero
        @return:
        """
        # 2000 training (1600 unlabeled, 400 labeled), 330 validation
        start_idx = 2000 * client_id
        unlabeled = [i for i in range(start_idx, start_idx + 1600)]
        start_idx = start_idx + 1600
        labeled = [i for i in range(start_idx, start_idx + 400)]
        start_idx = 25000
        validation = [i for i in range(start_idx, start_idx + 330)]
        return labeled, unlabeled, validation

    def __get_client_image_ids_80L_20U(self, client_id):
        # 2000 training (1600 labeled, 400 unlabeled), 330 validation
        start_idx = 2000 * client_id
        labeled = [i for i in range(start_idx, start_idx + 1600)]
        start_idx = start_idx + 1600
        unlabeled = [i for i in range(start_idx, start_idx + 400)]
        start_idx = 25000
        validation = [i for i in range(start_idx, start_idx + 330)]
        return labeled, unlabeled, validation

    def __get_client_image_ids_2labeled(self, client_id):
        # 2000 training (full labeled or unlabeled), 330 validation
        start_idx = 2000 * client_id
        if client_id < 2:  # Clients 0 and 1 have only labeled data
            labeled = [i for i in range(start_idx, start_idx + 2000)]
            unlabeled = [start_idx]
        else:
            labeled = [start_idx]
            unlabeled = [i for i in range(start_idx, start_idx + 2000)]
        start_idx = 25000
        validation = [i for i in range(start_idx, start_idx + 330)]
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
