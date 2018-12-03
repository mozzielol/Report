import numpy as np
import keras.backend as K



def get_weights(model):
    w = model.trainable_weights
    weights = []
    for i in w:
        weights.append(K.eval(i))
    return weights


def get_gradients(model):
    '''
    Return the gradient of every trainable weight in model
    '''
    weights = [tensor for tensor in model.trainable_weights]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from skimage.util import random_noise

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from copy import deepcopy
def create_permuted_mnist_task(num_datasets):
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    task_list = [mnist]
    for seed in range(1, num_datasets):
        task_list.append(permute(mnist, seed))
    return task_list

def permute(task, seed):
    np.random.seed(seed)
    perm = np.random.permutation(task.train._images.shape[1])
    permuted = deepcopy(task)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]
    return permuted


def create_disjoint_mnist_task():
    mnist = read_data_sets("MNIST_data/", one_hot=False)
    train_index = split_index(mnist.train.labels)
    test_index = split_index(mnist.test.labels)
    vali_index = split_index(mnist.validation.labels)
    mnist = read_data_sets("MNIST_data/", one_hot=True)

    train_set = {}
    test_set = {}
    vali_set = {}

    for i in range(3):
        train_set[i] = []
        test_set[i] = []
        vali_set[i] = []

        train_set[i].append(mnist.train.images[train_index[i]])
        train_set[i].append(mnist.train.labels[train_index[i]])

        test_set[i].append(mnist.test.images[test_index[i]])
        test_set[i].append(mnist.test.labels[test_index[i]])

        vali_set[i].append(mnist.validation.images[vali_index[i]])
        vali_set[i].append(mnist.validation.labels[vali_index[i]])

    return train_set,test_set,vali_set




def split_index(label):
    indi_1 = np.where(label<=3) 
    indi_2 = np.where(np.logical_and(3<label,label<=6)) 
    indi_3 = np.where(np.logical_and(6<label,label<=9))
    return [indi_1,indi_2,indi_3]




def load_cifar10():
    (X_train,y_train),(X_test,y_test) = _load_cifar10()
    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    indi_train_1 = np.where(y_train < 5)
    indi_train_2 = np.where(y_train >= 5)
    indi_test_1 = np.where(y_test < 5)
    indi_test_2 = np.where(y_test >= 5)
    
    X_train_1 = X_train[indi_train_1]
    X_train_2 = X_train[indi_train_2]
    y_train_1 = to_categorical(y_train[indi_train_1],5)
    y_train_2 = to_categorical(y_train[indi_train_2]-5,5)

    

    #y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)
    X_test_1 = X_test[indi_test_1]
    X_test_2 = X_test[indi_test_2]
    y_test_1 = to_categorical(y_test[indi_test_1],5)
    y_test_2 = to_categorical(y_test[indi_test_2]-5,5)


    train_task = []
    train_task.append([X_train_1, y_train_1])
    train_task.append([X_train_2, y_train_2])

    test_task = []
    test_task.append([X_test_1, y_test_1])
    test_task.append([X_test_2, y_test_2])

    return train_task, test_task




def _load_cifar10():
    from keras.datasets import cifar10
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train  /= 255
    X_test /= 255

    return (X_train,y_train),(X_test,y_test)


def load_cifa10_ori():
    from keras.datasets import cifar10
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()

    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train  /= 255
    X_test /= 255

    return X_train,y_train,X_test,y_test


def load_SVHN_ori():
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    
    train_data = sio.loadmat('/Users/lihonglin/Desktop/dataset/train_32x32.mat')
    test_data = sio.loadmat('/Users/lihonglin/Desktop/dataset/test_32x32.mat')

    # access to the dict
    x_train = np.rollaxis(train_data['X'],3,0).astype('float32')
    y_train = to_categorical((train_data['y'] - 1).reshape(-1,),10)

    x_test = np.rollaxis(test_data['X'],3,0).astype('float32')
    y_test = to_categorical((test_data['y'] - 1).reshape(-1,),10)

    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test


def load_SVHN():
    import scipy.io as sio
    
    train_data = sio.loadmat('/home/dp00143/Desktop/dataset/train_32x32.mat')
    test_data = sio.loadmat('/home/dp00143/Desktop/dataset/test_32x32.mat')

    # access to the dict
    X_train = np.rollaxis(train_data['X'],3,0).astype('float32')
    y_train = train_data['y'] - 1
    X_train /= 255


    X_test = np.rollaxis(test_data['X'],3,0).astype('float32')
    y_test = test_data['y'] - 1
    X_test /= 255

    y_train = y_train.reshape(-1,)
    y_test = y_test.reshape(-1,)

    indi_train_1 = np.where(y_train < 5)
    indi_train_2 = np.where(y_train >= 5)
    indi_test_1 = np.where(y_test < 5)
    indi_test_2 = np.where(y_test >= 5)
    
    X_train_1 = X_train[indi_train_1]
    X_train_2 = X_train[indi_train_2]
    y_train_1 = to_categorical(y_train[indi_train_1],10)
    y_train_2 = to_categorical(y_train[indi_train_2],10)

    

    #y_train = to_categorical(y_train, 10)
    #y_test = to_categorical(y_test, 10)
    X_test_1 = X_test[indi_test_1]
    X_test_2 = X_test[indi_test_2]
    y_test_1 = to_categorical(y_test[indi_test_1],10)
    y_test_2 = to_categorical(y_test[indi_test_2],10)


    train_task = []
    train_task.append([X_train_1, y_train_1])
    train_task.append([X_train_2, y_train_2])

    test_task = []
    test_task.append([X_test_1, y_test_1])
    test_task.append([X_test_2, y_test_2])

    return train_task, test_task
























