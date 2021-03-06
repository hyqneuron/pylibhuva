import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset
from .. import local_config
import os
import math


class TensorDataset(Dataset):

    def __init__(self, tensor, labels=None, transform=None, pre_transform=None, resize=None):
        """
        pre_transform: per-sample non-stochastic transform
        transform: per-sample stochastic transform
        """
        assert torch.is_tensor(tensor)
        if labels is not None: assert torch.is_tensor(labels)
        self.tensor = tensor
        self.labels = labels
        self.transform = transform
        self.pre_transform = pre_transform
        if pre_transform:
            for i in xrange(self.tensor.size(0)):
                self.tensor[i] = pre_transform(self.tensor[i])
        if resize is not None:
            tensor = tensor.resize_(tensor.size(0), *list(resize))

    def __getitem__(self, index):
        inp   = self.tensor[index]
        if self.transform is not None:
            inp = self.transform(inp)
        if self.labels is not None:
            label = self.labels[index]
            return inp, label
        else:
            return inp

    def __len__(self):
        return self.tensor.size(0)


class TensorDatasetIterator(object):

    def __init__(self, dataset, batch_size, indices):
        assert type(dataset) in [TensorDataset], 'dataset must be a TensorDataset'
        assert dataset.transform is None, 'per-sample stochastic transform not supported'
        self.dataset = dataset
        self.batch_size = batch_size
        self.position = 0
        self.indices = indices

    def __next__(self):
        # FIXME we are bypassing dataset.transform!!!
        if self.position >= len(self.indices):
            raise StopIteration
        index = self.indices[self.position]
        self.position += 1
        result = self.dataset.tensor[index]
        if self.dataset.labels is not None:
            result = result, self.dataset.labels[index]
        return result

    next = __next__


class TensorLoader(object):
    """
    Uses TensorDataset, which keep entire dataset in RAM.
    Speeds up MLP on MNIST by 2x.
    """

    def __init__(self, dataset, batch_size, shuffle=False, indices=None, gpu=False):
        assert isinstance(dataset, TensorDataset), 'dataset must be a TensorDataset'
        assert dataset.transform is None, 'per-sample stochastic transform not supported'
        self.dataset = dataset
        self.batch_size = batch_size
        self.gpu = gpu
        """
        We allow user to specify fixed traversal order using indices. When indices is given, we cannot shuffle
        """
        assert not(indices is not None and shuffle)
        self.shuffle = shuffle
        self.indices = indices

    def __iter__(self):
        if self.indices is not None:
            indices = self.indices
        else:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset)).split(self.batch_size)
            else:
                indices = torch.range(0, len(self.dataset)-1).long().split(self.batch_size)
        if self.gpu: indices = [index.cuda() for index in indices]
        return TensorDatasetIterator(self.dataset, self.batch_size, indices)

    def __len__(self):
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))


"""
class ViewTransform(object):

    def __init__(self, size):
        assert isinstance(size, torch.Size)
        self.size = size

    def __call__(self, tensor):
        return tensor.view(self.size)
"""


def make_data_mnist(batch_size, normalize=True, spatial=False, gpu=True, shuffle_train=True, shuffle_test=False):
    """
    in_ram: store the whole tensor in RAM. Without this we can't feed the GPU fast enough
    """
    trs = []
    if normalize:
        trs += [torchvision.transforms.Normalize(mean=[0.13066], std=[0.308108])]
    resize = torch.Size([784]) if not spatial else torch.Size([1,28,28])
    transforms = torchvision.transforms.Compose(trs)
    def make_dataset(train):
        raw = torchvision.datasets.MNIST(local_config.torchvision_path_mnist, 
                train=train, transform=torchvision.transforms.ToTensor(), download=True)
        inp_tensor = torch.Tensor(len(raw), *list(raw[0][0].size())).zero_()
        lab_tensor = torch.Tensor(len(raw)).zero_()
        if gpu:
            inp_tensor = inp_tensor.cuda()
            lab_tensor = lab_tensor.cuda()
        for i in xrange(len(raw)):
            inp_tensor[i], lab_tensor[i] = raw[i]
        return TensorDataset(inp_tensor, lab_tensor, pre_transform=transforms, resize=resize)
    dataset_train= make_dataset(True)
    dataset_test = make_dataset(False)
    loader_train = TensorLoader(dataset_train,batch_size=batch_size, shuffle=shuffle_train, gpu=gpu)
    loader_test  = TensorLoader(dataset_test, batch_size=batch_size, shuffle=shuffle_test,  gpu=gpu)
    return (dataset_train, loader_train), (dataset_test, loader_test)


binarized_mnist_sources = dict(
    train='http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',
    val  ='http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',
    test ='http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',
)


def make_tensor_binary_mnist():
    """
    Download binarized mnist files from larocheh's website if they are not present on dist
    Parse the amat files and return them as torch.Tensor
    """
    name_fullname = {}
    # build filenames
    for name, source in binarized_mnist_sources.iteritems():
        shortname = source.split('/')[-1]
        fullname  = os.path.join(local_config.path_binary_mnist, shortname)
        name_fullname[name] = fullname
    # check folder exists
    if not os.path.exists(local_config.path_binary_mnist):
        raise Exception, 'please make sure huva.local_config.path_binary_mnist exists.'
    # download files if they don't exist
    for name, fullname in name_fullname.iteritems():
        if not os.path.exists(fullname):
            import urllib
            source = binarized_mnist_sources[name]
            print('Downloading {} from {} to {}'.format(name, source, fullname))
            urllib.urlretrieve(source, fullname)
    # load and parse
    def read_amat_file(filename):
        f = open(filename, 'r')
        lines = f.readlines()
        N = len(lines)                      # number of rows
        M = len(lines[0].strip().split())   # number of cols
        tensor = torch.Tensor(N,M).fill_(0)
        for n in xrange(N):
            numbers = map(float, lines[n].strip().split())
            assert len(numbers) == M
            tensor[n] = torch.Tensor(numbers)
        f.close()
        return tensor
    print("Reading binarized MNIST amat files...")
    tensor_train = read_amat_file(name_fullname['train'])
    tensor_test  = read_amat_file(name_fullname['test'])
    print('Reading done')
    return tensor_train, tensor_test


def make_data_binary_mnist(batch_size, spatial=False, shuffle_train=True, shuffle_test=False, gpu=True):
    resize = torch.Size([784]) if not spatial else torch.Size([1,28,28])
    tensor_train, tensor_test = make_tensor_binary_mnist()
    if spatial:
        tensor_train = tensor_train.view(tensor_train.size(0), 1, 28, 28)
        tensor_test  = tensor_test .view(tensor_test .size(0), 1, 28, 28)
    if gpu:
        tensor_train = tensor_train.cuda()
        tensor_test  = tensor_test.cuda()
    dataset_train = TensorDataset(tensor_train, resize=resize)
    dataset_test  = TensorDataset(tensor_test , resize=resize)
    loader_train  = TensorLoader(dataset_train, batch_size, shuffle=shuffle_train, gpu=gpu)
    loader_test   = TensorLoader(dataset_test,  batch_size, shuffle=shuffle_test,  gpu=gpu)
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

