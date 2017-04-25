import torchvision
from torch.utils.data import DataLoader
from .. import local_config

def make_data_mnist(batch_size, train_threads=1, test_threads=1):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.126], std=[0.302])])
    def make_dataset(train):
        return torchvision.datasets.MNIST(local_config.torchvision_path_mnist, 
                                          train=train, transform=transforms, download=True)
    dataset_train= make_dataset(True)
    dataset_test = make_dataset(False)
    loader_train = DataLoader(dataset_train,batch_size=batch_size, shuffle=True, num_workers=train_threads, pin_memory=True)
    loader_test  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,num_workers=test_threads,  pin_memory=True)
    return (dataset_train, loader_train), (dataset_test, loader_test)

def make_data_cifar10(batch_size, train_threads=1, test_threads=1):
    def make_dataset(train):
        transforms = [torchvision.transforms.RandomHorizontalFlip()] if train else [] 
        transforms += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.469, 0.481, 0.451], std=[0.239,0.245,0.272])
        ]
        return torchvision.datasets.CIFAR10( local_config.torchvision_path_cifar10, 
                train=train, transform=torchvision.transforms.Compose(transforms), download=True)
    dataset_train = make_dataset(True)
    dataset_test  = make_dataset(False)
    loader_train = DataLoader(dataset_train,batch_size=batch_size, shuffle=True, num_workers=train_threads, pin_memory=True)
    loader_test  = DataLoader(dataset_test, batch_size=batch_size, shuffle=False,num_workers=test_threads,  pin_memory=True)
    return (dataset_train, loader_train), (dataset_test, loader_test)
