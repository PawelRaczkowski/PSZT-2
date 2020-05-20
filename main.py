import struct as st
from array import array

import numpy as np
# uwagi: nie wiem czy do tego co już jest nie trzeba dorobić tzw bias. czyli czegoś co nam przesunie nasz hyperplane
# o pewien skalar

def readTrainingData(filePath): # wczytuje obrazki- zwraca tablice obrazków
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


def readTrainingLabels(filePath): # wczytuje labele
    with open(filePath, 'rb') as file:
        magic, size = st.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())
        return labels


def convertLabels(labelsTraining, param): # konwertuje labele z {0,1,2,3...9} => {-1,1}
    convertedLabels = []
    for i in range(len(labelsTraining)):
        if labelsTraining[i] == param:
            convertedLabels.append(1)
        else:
            convertedLabels.append(-1)
    return convertedLabels


def divideIntoModels(labelsTraining):  # do taktyki one vs all => np jest 0 vs reszta to dla 0 jest label 1 a dla reszty 0
    models = []  # lista list gdzie element takiej listy to labele dla danej cyfry
    for i in range(10):
        models.append(convertLabels(labelsTraining, i))
    return models


def divideonevsonemodel(labelsTraining):  # do taktyki one vs one - tworzy listę list-> 1 lista przetrzymuje indeksy przykładów
    #które mają label 0
    groupedLabels = [[] for i in range(10)]
    for i in range(len(labelsTraining)):
        groupedLabels[labelsTraining[i]].append(i)
    return groupedLabels


def trainModel(function_weights, alpha, imagesTraining, param): # zwraca wytrenowane wagi
    counter=0
    while counter<=2:
        for i in range(len(imagesTraining)):
            realValue = param[i]

            y = np.dot(imagesTraining[i], function_weights)
            # print('Real value: ', realValue)
            # print('Calculated y: ', y)
            if y * realValue >= 1:  # kiedy dobrze sklasyfikowało
                # print('Well classified')
                result = [function_weights[j] - 2 * alpha * (1 / (100000)) * function_weights[j] for j in
                          range(len(function_weights))]
                function_weights = result
            else:
                # print('Bad classification')
                result = [function_weights[j] + alpha * (
                        realValue * (imagesTraining[i][j]) - 2 * (1 / (100000)) * function_weights[j]) for j
                          in range(len(imagesTraining[i]))]
                function_weights = result
        counter+=1
    print('Calculated weights ', function_weights)
    return function_weights


def predict(feature_vector, weights):
    y = np.dot(feature_vector, weights)
    if y > 1:
        return 1
    else:
        return -1


def validate_model(feature, weights): # walidacja - im punkt dalej od granicy to tym lepszy
    return np.dot(feature, weights) / (np.linalg.norm(weights))


### CZYTANIE DANYCH ###

imagesTraining, rows, columns = readTrainingData(
    'samples/train-images.idx3-ubyte')  # images[0] to wektor pikseli pierwszego obrazka
labelsTraining = readTrainingLabels('samples/train-labels.idx1-ubyte')

for i in range(len(imagesTraining)):  # normalizacja DO WARTOŚCI (0,1)
    imagesTraining[i] = imagesTraining[i] / float(255)

##############################################################################################################################################
# TAKTYKA ONE VS ALL
#tworzymy N modeli gdzie N to jest długość zbioru klas czyli w naszym przypadku =10
# 1 model będzie traktował 0 z labelem 1 a resztę z labelem 0; 2 model będzie traktował 1 z labelem 1 a resztę z 0 itd.
nModels = divideIntoModels(labelsTraining)  # nmodels[0] to tam gdzie 0 są oznaczone jako 1 a reszta jako 0

weights = np.zeros((rows * columns, 1))  # inicjalizacja wag
weights_proper = []
for i in range(len(
        weights)):  # zapisywalo jako lista list jednoelementowych więc zrobiłem taką konwersję na normalne wartości XD
    weights_proper.append(weights[i][0])

alpha = 0.001  # pilnuje kroku z jakim zmieniamy wagi w trenowaniu
trained_weights = []

# for i in range(len(nModels)): # trenowanie modeli
#     trained_weights.append(trainModel(weights_proper, alpha, imagesTraining, nModels[i]))
#     print('Trained model nr: ', i + 1)

# testowanie modeli i dokonywanie predykcji
imagesPrediction, r, c = readTrainingData('samples/t10k-images.idx3-ubyte')
labelsPrediction = readTrainingLabels('samples/t10k-labels.idx1-ubyte')

for i in range(len(imagesPrediction)):  # normalizacja
    imagesPrediction[i] = imagesPrediction[i] / float(255)

accuracy = 0
probes = len(imagesPrediction)
correct = 0
validations = []  # ocena prawdopodobieństwa słuszności każdego modelu
# for j in range(probes): # predykcja dla modelu one vs all
#     for i in range(len(trained_weights)):
#         predict(imagesPrediction[j], trained_weights[i])
#         validations.append(validate_model(imagesPrediction[j], trained_weights[i]))
#
#     index = validations.index(max(validations))
#     print('Predicted by model: ', index)
#     print('Real value: ', labelsPrediction[j])
#     if labelsPrediction[j] == index:
#         correct = correct + 1
#     validations.clear()
#
# accuracy = correct / probes
# print('Accuracy: ', accuracy)

################################################################################################################################################

# TAKTYKA ONE VS ONE
groupedLabels = divideonevsonemodel(labelsTraining)


def getImages(label1, label2, groupedLabels,imagesTraining):  # dzieli dane na obrazki z label1 i label2 i nadajemy odpowiednie labele do modelu
    images = []
    convertedLabels = []
    i = 0
    j = 0
    while i < len(groupedLabels[label1]) or j < len(groupedLabels[label2]):
        if i != len(groupedLabels[label1]):
            convertedLabels.append(1)
            images.append(imagesTraining[groupedLabels[label1][i]])
            i += 1
        if j != len(groupedLabels[label2]):
            convertedLabels.append(-1)
            images.append(imagesTraining[groupedLabels[label2][j]])
            j += 1
    # for i in range(len(groupedLabels[label1])):
    #     convertedLabels.append(1)
    #     images.append(imagesTraining[groupedLabels[label1][i]])
    # for i in range(len(groupedLabels[label2])):
    #     convertedLabels.append(-1)
    #     images.append(imagesTraining[groupedLabels[label2][i]])
    return images, convertedLabels

def sigmoid(x):
    return 1/(1+np.exp(-x))


onevsoneweights = [[[] for j in range(10)] for i in range(10)] # lista list list XD ->
counter_models = 1
alpha = 0.001
for i in range(10):
    for j in range(i + 1, 10):
        dividedImages, convertedLabels = getImages(i, j, groupedLabels, imagesTraining)
        trained_weight = trainModel(weights_proper, alpha, dividedImages, convertedLabels)
        onevsoneweights[i][j].append(trained_weight)  # tutaj przykład: i=0 j=1-> w [i][j] trzymamy wagi wytrenowane do
        #modelu wyróżniającego 0 od 1
        onevsoneweights[j][i].append(trained_weight)
        print('Trained model nr ', counter_models)
        counter_models += 1
    # print(onevsoneweights)

# walidacja i dokonywanie predykcji
new_validations = []
for i in range(10):
    new_validations.append(0)
new_counter = 0
for k in range(probes):
    for i in range(10):
        for j in range(i + 1, 10):
            prediction = predict(imagesPrediction[k], onevsoneweights[i][j][0])
            change = sigmoid(validate_model(imagesPrediction[k], onevsoneweights[i][j][0]))
            if prediction == 1:
                new_validations[i] += 10+change

            else:
                new_validations[j] += 10+change

    print('Validations: ', new_validations)
    index = new_validations.index(max(new_validations))
    print('Predicted by model: ', index)
    print('Real value: ', labelsPrediction[k])
    if index == labelsPrediction[k]:
        new_counter += 1
    new_validations = [0 for i in range(10)]

print('Accuracy: ', (new_counter) / probes)
