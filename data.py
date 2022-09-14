import numpy as np
import time 

class IdxData:
    def __init__(self, fname_training, fname_training_labels, fname_test, fname_test_labels):
        self.magic_train, self.train = self.read_idx(fname_training)
        self.magic_train_labels, self.train_labels = self.read_idx_labels(fname_training_labels)

        self.magic_test, self.test = self.read_idx(fname_test)
        self.magic_test_labels, self.test_labels = self.read_idx_labels(fname_test_labels)
        
    def read_idx(self, fname):
        fd = open(fname, 'br')
        dim = []

        # magic_num
        magic = fd.read(4)
        ndim = magic[3]

        # Get dimensions
        for i in range(ndim):
            dim.append(int.from_bytes(fd.read(4), byteorder='big'))

        raw = np.zeros((dim[0], dim[1]*dim[2]), dtype='uint8')

        # Specific for MNIST
        for i_img in range(dim[0]):
            for pixel in range(dim[1]*dim[2]):
                raw[i_img, pixel] = int.from_bytes(fd.read(1), byteorder='big')

        return magic, raw

    def read_idx_labels(self, fname):
        fd = open(fname, 'br')

        # magic_num
        magic = fd.read(4)
        items = int.from_bytes(fd.read(4), byteorder='big')
        
        raw_labels = np.zeros(items, dtype='int')
        
        for i in range(items):
            raw_labels[i] = int.from_bytes(fd.read(1), byteorder='big')

        return magic, raw_labels

    def show(self, dtype = 'None'):
        if dtype.lower() == 'training':
            ptype = dtype.upper()
            magic = self.magic_train
            ndim = self.train.ndim
            shape = self.train.shape
            
        elif dtype.lower() == 'test':
            ptype = dtype.upper()
            magic = self.magic_test
            ndim = self.test.ndim
            shape = self.test.shape

        else:
            print('Unknown type \'{0}\'. Please use one of the following types:\n'
                  '1 - {1}\n'
                  '2 - {2}\n'.format(dtype, 'Training', 'Test'))
            return
        
        print('--Parameters--\n'
              'Type: {0}\n'
              'magic number: {1}\n'
              'ndim: {2}\n'
              'dimensions: {3}\n'.format(ptype, magic, ndim, shape))

def preprocess():
    # -- DATA -- 
    tds = time.process_time()
    dx = IdxData('raw_data/train-images-idx3-ubyte',
                 'raw_data/train-labels-idx1-ubyte',
                 'raw_data/t10k-images-idx3-ubyte',
                 'raw_data/t10k-labels-idx1-ubyte')
    tdf = time.process_time()

    dx.show('training')
    dx.show('test')
    print('Pre-processing_data time: {0}'.format(tdf-tds))
    print('-'*50)
    return dx
