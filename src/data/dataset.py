from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def get_classes(self):
        pass

    @abstractmethod
    def get_server_data(self):
        pass

    @abstractmethod
    def get_client_data(self, client_id):
        pass

    @abstractmethod
    def get_client_data_counts(self, client_id):
        pass

    @abstractmethod
    def get_client_test_val_data(self, client_id):
        pass

    @abstractmethod
    def get_global_test_data(self):
        pass
