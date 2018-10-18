import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def create_dataset(train_file, test_file):
    train_path = []
    test_path = []
    train_X = []
    y_test = []
    le = LabelEncoder()

    with open(train_file, 'r') as f:
        for line in f:
            train_path.append(line.rstrip().replace("Images", "features/vgg16_fc2").replace(".jpg", ".npy"))
    with open(test_file, 'r') as f:
        for line in f:
            test_path.append(line.rstrip().replace("Images", "features/vgg16_fc2").replace(".jpg", ".npy"))

    print("Loading train dataset")
    for i in tqdm(train_path):
        data = np.load(i)
        data = data.reshape(data.shape[1],)
        train_X.append(data)
    train_X = np.array(train_X)

    print("Loading test dataset")
    for i in tqdm(test_path):
        data = np.load(i)
        data = data.reshape(data.shape[1],)
        y_test.append(data)
    y_test = np.array(y_test)

    train_Y = [i.split('/')[2] for i in train_path]
    y_true = [i.split('/')[2] for i in test_path]
    train_Y = le.fit_transform(train_Y)
    y_true = le.transform(y_true)

    return train_X, y_test, train_Y, y_true