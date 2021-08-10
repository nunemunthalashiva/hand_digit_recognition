"""
loading dataset
"""
import numpy as np
import gzip,pickle

def load_dataset():
    """
    return tuple containing training_data( which has 50k train vals,
    each having 784 pixels ) validation_data,test_data which is of 10k
    images of size 784 pixels
    """
    temp = gzip.open('mnist.pkl.gz')
    train, val , test = pickle.load(temp,encoding='latin1')
    temp.close()
    train_inp = [np.reshape(x, (784,1)) for x in train[0]]
    train_outp = [one_hot(y) for y in train[1]]
    training_data = zip(train_inp, train_outp)
    validation_inp = [np.reshape(x, (784, 1)) for x in val[0]]
    validation_data = zip(validation_inp, val[1])
    test_inp = [np.reshape(x, (784, 1)) for x in test[0]]
    test_data = zip(test_inp, test[1])
    return (training_data,validation_data,test_data)

def one_hot(y):
    """
    just returning encoding of our input . For eg: if the given number is
    '8' then we return a 10d vector where its 8th index contains 1 and all others 0
    """
    temp = np.zeros((10,1))
    temp[y]=1
    return temp
