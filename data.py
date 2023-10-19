import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def to_categorical(y, num_classes):
    # Convierte un vector de etiquetas en una matriz one-hot.
    # y: vector de etiquetas
    # num_classes: número total de clases
    y = np.array(y)
    if len(y.shape) > 1:
        raise ValueError("Input 'y' debe ser un vector de etiquetas, no una matriz.")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for i in range(n):
        categorical[i, y[i]] = 1
    return categorical



dirname = os.path.join(os.getcwd(), 'sportimages')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0
num_clases = 10
limit = 100 #Limita el número de imágenes a usar por clase


print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    count = 0
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
            count = count+1
        if count >= limit:
            break
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))



labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
    if indice >= limit:
        break
print("Cantidad etiquetas creadas: ",len(labels))

deportes=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    deportes.append(name[len(name)-1])
    indice=indice+1

y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy

# Encuentra los números de clases
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


#Mezclar todo y crear los grupos de entrenamiento y testing
train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Cambiar las etiquetas de enteros a categóricas (one-hot encoding)
train_Y_one_hot = to_categorical(train_Y, nClasses)
test_Y_one_hot = to_categorical(test_Y, nClasses)

# Mostrar el cambio de la etiqueta usando one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)

# gurdar los datos en un archivo numpy
np.save('train_X.npy', train_X)
np.save('train_label.npy', train_label)
np.save('valid_X.npy', valid_X)
np.save('valid_label.npy', valid_label)

np.save('test_X.npy', test_X)
np.save('test_Y.npy', test_Y)
