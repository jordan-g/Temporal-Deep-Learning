import numpy as np
import mnist

def load_mnist_data(n_examples, n_test_examples, validation=True):
    try:
        x_set      = np.load("mnist/x_set.npy").astype(np.float32)
        t_set      = np.load("mnist/t_set.npy").astype(np.float32)
        if validation:
            x_test_set = x_set[:, n_examples:n_examples+n_test_examples]
            t_test_set = t_set[:, n_examples:n_examples+n_test_examples]
        else:
            x_test_set = np.load("mnist/x_test_set.npy").astype(np.float32)[:, :n_test_examples]
            t_test_set = np.load("mnist/t_test_set.npy").astype(np.float32)[:, :n_test_examples]

        x_set = x_set[:, :n_examples]
        t_set = t_set[:, :n_examples]
    except:
        print("Converting MNIST data...")
        
        import mnist
        
        try:
            trainfeatures, trainlabels = mnist.traindata()
            testfeatures, testlabels   = mnist.testdata()
        except:
            print("Error: Could not find original MNIST files in the current directory.")
            return
        
        # normalize inputs
        x_set      = trainfeatures/255.0
        x_test_set = testfeatures/255.0
        
        t_set = np.zeros((10, x_set.shape[1])).astype(int)
        for i in range(x_set.shape[1]):
            t_set[int(trainlabels[i]), i] = 1
        
        t_test_set = np.zeros((10, x_test_set.shape[1])).astype(int)
        for i in range(x_test_set.shape[1]):
            t_test_set[int(testlabels[i]), i] = 1
        try:
            import mnist
            
            try:
                trainfeatures, trainlabels = mnist.traindata()
                testfeatures, testlabels   = mnist.testdata()
            except:
                print("Error: Could not find original MNIST files in the current directory.")
                return
            
            # normalize inputs
            x_set      = trainfeatures/255.0
            x_test_set = testfeatures/255.0
            
            t_set = np.zeros((10, x_set.shape[1])).astype(int)
            for i in range(x_set.shape[1]):
                t_set[int(trainlabels[i]), i] = 1
            
            t_test_set = np.zeros((10, x_test_set.shape[1])).astype(int)
            for i in range(x_test_set.shape[1]):
                t_test_set[int(testlabels[i]), i] = 1
        except:
            return

        np.save("mnist/x_set", x_set)
        np.save("mnist/x_test_set", x_test_set)
        np.save("mnist/t_set", t_set)
        np.save("mnist/t_test_set", t_test_set)

    return x_set, t_set, x_test_set, t_test_set

def load_cifar10_data(n_examples, n_test_examples, validation=True):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    data_batches = []
    labels = []

    for i in range(5):
        dictionary = unpickle("cifar-10/data_batch_{}".format(i+1))
        data_batches.append(dictionary[b'data'])
        labels += dictionary[b'labels']

    x_set = np.concatenate(data_batches, axis=0).T.astype(np.float32)/255.0
    t_set = np.zeros((10, len(labels))).astype(np.float32)
    for i in range(len(labels)):
        t_set[labels[i], i] = 1

    if validation:
        x_test_set = x_set[:, n_examples:n_examples+n_test_examples]
        t_test_set = t_set[:, n_examples:n_examples+n_test_examples]
    else:
        dictionary = unpickle("cifar-10/test_batch")
        labels = dictionary[b'labels']
        x_test_set = dictionary[b'data'].T.astype(np.float32)[:, :n_test_examples]/255.0
        t_test_set = np.zeros((10, n_test_examples)).astype(np.float32)
        for i in range(n_test_examples):
            t_test_set[labels[i], i] = 1

    x_set = x_set[:, :n_examples]
    t_set = t_set[:, :n_examples]

    return x_set, t_set, x_test_set, t_test_set