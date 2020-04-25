import struct as st
from array import array

import numpy as np


def readTrainingData(filePath):
    train_imagesfile = open(filePath, 'rb')
    train_imagesfile.seek(0)
    magic = st.unpack('>4B', train_imagesfile.read(4))  # .read(4) czytamy 4 bajty
    nImg = st.unpack('>I', train_imagesfile.read(4))[0]  # num of images
    nR = st.unpack('>I', train_imagesfile.read(4))[0]  # num of rows
    nC = st.unpack('>I', train_imagesfile.read(4))[0]  # num of column
    images_array = np.zeros((nImg, nR * nC))
    nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte
    images_array = float(255) - np.asarray(
        st.unpack('>' + 'B' * nBytesTotal, train_imagesfile.read(nBytesTotal))).reshape(
        (nImg, nR * nC))
    return np.array(images_array), nR, nC


def readTrainingLabels(filePath):
    with open(filePath, 'rb') as file:
        magic, size = st.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())
        return labels


def convertLabels(labelsTraining, param):
    convertedLabels = []
    for i in range(len(labelsTraining)):
        if labelsTraining[i] == param:
            convertedLabels.append(1)
        else:
            convertedLabels.append(-1)
    return convertedLabels


def divideIntoModels(labelsTraining):
    models = []  # lista list gdzie element takiej listy to labele dla danej cyfry
    for i in range(10):
        models.append(convertLabels(labelsTraining, i))
    return models


def sigmoid(param):
    return 1 / (1 + np.exp(-param))


def trainModel(weights, alpha, imagesTraining, param):
    for i in range(len(imagesTraining)):
        realValue = param[i]
        y = np.dot(imagesTraining[i], weights)
        if y * realValue >= 1:  # kiedy dobrze sklasyfikowało
            result = [weights[j] - 2 * alpha * (1 / (i + 1)) * weights[j] for j in range(len(weights))]
            weights = result
        else:
            result = [weights[j] + alpha * (realValue * (imagesTraining[i][j]) - 2 * (1 / (i + 1)) * weights[j]) for j
                      in range(len(imagesTraining[i]))]
            weights = result
    return weights


def predict(feature_vector, weights):
    y = np.dot(feature_vector, weights)
    if (y > 1):
        return 1
    else:
        return -1


def validate_model(feature, weights):
    return np.dot(feature, weights) / (np.linalg.norm(weights))





### CZYTANIE DANYCH ###

imagesTraining, rows, columns = readTrainingData( 'samples/train-images.idx3-ubyte')  # images[0] to wektor pikseli pierwszego obrazka
labelsTraining = readTrainingLabels('samples/train-labels.idx1-ubyte')

for i in range(len(imagesTraining)):  # normalizacja DO WARTOŚCI (0,1)
    imagesTraining[i] = imagesTraining[i] / float(255)

# taktyka one vs all - tworzymy N modeli gdzie N to jest długość zbioru klas czyli w naszym przypadku =10
# 1 model będzie traktował 0 z labelem 1 a resztę z labelem 0; 2 model będzie traktował 1 z labelem 1 a resztę z 0 itd.
nModels = divideIntoModels(labelsTraining)  # nmodels[0] to tam gdzie 0 są oznaczone jako 1 a reszta jako 0

weights = np.zeros((rows * columns, 1)) # inicjalizacja wag
weights_proper = []
for i in range(len(weights)):  # zapisywalo jako lista list jednoelementowych więc zrobiłem taką konwersję na normalne wartości XD
    weights_proper.append(weights[i][0])

alpha = 0.0001  # pilnuje kroku z jakim zmieniamy wagi w trenowaniu
trained_weights = []

for i in range(len(nModels)):
    trained_weights.append(trainModel(weights_proper, alpha, imagesTraining, nModels[i]))
    print('Trained model nr: ', i + 1)


# testowanie modeli i dokonywanie predykcji
imagesPrediction, r, c = readTrainingData('samples/t10k-images.idx3-ubyte')
labelsPrediction = readTrainingLabels('samples/t10k-labels.idx1-ubyte')

for i in range(len(imagesPrediction)):  # normalizacja
    imagesPrediction[i] = imagesPrediction[i] / float(255)



accuracy=0
probes=len(imagesPrediction)
correct=0
validations = []  # ocena prawdopodobieństwa słuszności każdego modelu
for j in range(probes):
    for i in range(len(trained_weights)):
        predict(imagesPrediction[j], trained_weights[i])
        validations.append(validate_model(imagesPrediction[j], trained_weights[i]))

    index=validations.index(max(validations))
    print('Predicted by model: ',index)
    print('Real value: ',labelsPrediction[j])
    if labelsPrediction[j]==index:
        correct=correct+1
    validations.clear()



accuracy=correct/probes
print('Accuracy: ', accuracy)

