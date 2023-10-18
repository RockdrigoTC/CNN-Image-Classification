
import numpy as np


class CNN:
    def __init__(self, input_shape, num_classes, weight_init='xavier'):
        # Inicializa los pesos y sesgos de las capas de la CNN
        self.input_shape = input_shape  # Forma de entrada de las imágenes
        self.num_classes = num_classes  # Número de clases en tu problema
        self.weight_init = weight_init  # Tipo de inicialización de pesos

        # Define otros atributos, como los pesos y sesgos de las capas
        self.weights = {}  # Diccionario de pesos
        self.biases = {}  # Diccionario de sesgos

        # Define el tamaño del kernel y el stride de cada capa de convolución
        self.kernel_sizes = [3, 3, 3]
        self.strides = [1, 1, 1]

        # Define el tamaño del filtro y el stride de cada capa de max pooling
        self.pool_sizes = [2, 2, 2]
        self.pool_strides = [2, 2, 2]

        self.learning_rate = 0.01  # Tasa de aprendizaje

        # Define las capas de la CNN
        self.conv_layers = []  # Lista de capas de convolución
        self.fc_layers = []  # Lista de capas completamente conectadas

        # Inicializa los pesos y sesgos de las capas de la CNN
        self.initialize_weights()

        # Inicializa las capas de la CNN
        self.initialize_layers()

    def initialize_weights(self):
        # Inicializa los pesos de las capas de la CNN
        if self.weight_init == 'xavier':
            # Inicialización de Xavier
            self.weights['W1'] = np.random.randn(self.kernel_sizes[0], self.kernel_sizes[0], self.input_shape[2], 32) / np.sqrt(
                self.kernel_sizes[0] * self.kernel_sizes[0] * self.input_shape[2])
            self.weights['W2'] = np.random.randn(self.kernel_sizes[1], self.kernel_sizes[1], 32, 64) / np.sqrt(
                self.kernel_sizes[1] * self.kernel_sizes[1] * 32)
            self.weights['W3'] = np.random.randn(self.kernel_sizes[2], self.kernel_sizes[2], 64, 128) / np.sqrt(
                self.kernel_sizes[2] * self.kernel_sizes[2] * 64)
            self.weights['W4'] = np.random.randn(self.input_shape[1] * self.input_shape[2] * 128 // 64, 128) / np.sqrt(
                self.input_shape[1] * self.input_shape[2] * 128 // 64)
            self.weights['W5'] = np.random.randn(128, self.num_classes) / np.sqrt(128)
        elif self.weight_init == 'random':
            # Inicialización aleatoria
            self.weights['W1'] = np.random.randn(self.kernel_sizes[0], self.kernel_sizes[0], self.input_shape[2], 32)
            self.weights['W2'] = np.random.randn(self.kernel_sizes[1], self.kernel_sizes[1], 32, 64)
            self.weights['W3'] = np.random.randn(self.kernel_sizes[2], self.kernel_sizes[2], 64, 128)
            self.weights['W4'] = np.random.randn(self.input_shape[1] * self.input_shape[2] * 128 // 64, 128)
            self.weights['W5'] = np.random.randn(128, self.num_classes)

        # Inicializa los sesgos de las capas de la CNN
        self.biases['b1'] = np.zeros(32)
        self.biases['b2'] = np.zeros(64)
        self.biases['b3'] = np.zeros(128)
        self.biases['b4'] = np.zeros(128)
        self.biases['b5'] = np.zeros(self.num_classes)


    def initialize_layers(self):
        # Inicializa las capas de la CNN
        self.conv_layers.append(Convolution(self.weights['W1'], self.biases['b1'], self.strides[0], self.kernel_sizes[0]))
        self.conv_layers.append(MaxPooling(self.pool_strides[0], self.pool_sizes[0]))
        self.conv_layers.append(Convolution(self.weights['W2'], self.biases['b2'], self.strides[1], self.kernel_sizes[1]))
        self.conv_layers.append(MaxPooling(self.pool_strides[1], self.pool_sizes[1]))
        self.conv_layers.append(Convolution(self.weights['W3'], self.biases['b3'], self.strides[2], self.kernel_sizes[2]))
        self.conv_layers.append(MaxPooling(self.pool_strides[2], self.pool_sizes[2]))
        self.fc_layers.append(FullyConnected(self.weights['W4'], self.biases['b4'], activation='relu'))
        self.fc_layers.append(FullyConnected(self.weights['W5'], self.biases['b5'], activation='sigmoid'))

        
    def forward(self, X):
        # Propagación hacia adelante de la CNN
        for i in range(len(self.conv_layers)):
            X = self.conv_layers[i].forward(X)

        # Aplana la salida de la última capa de convolución
        X = X.reshape(X.shape[0], -1)

        # Propagación hacia adelante de las capas completamente conectadas
        for i in range(len(self.fc_layers)):
            X = self.fc_layers[i].forward(X)

        return X
    
    def backward(self, X, y, y_pred):
        # Propagación hacia atrás de la CNN
        for i in reversed(range(len(self.fc_layers))):
            y_pred = self.fc_layers[i].backward(y_pred, self.learning_rate, i)

        # Aplana la salida de la última capa de convolución
        y_pred = y_pred.reshape(y_pred.shape[0], self.conv_layers[-1].output_shape[1],
                                self.conv_layers[-1].output_shape[2], self.conv_layers[-1].output_shape[3])

        # Propagación hacia atrás de las capas de convolución
        for i in reversed(range(0, len(self.conv_layers), 2)):
            y_pred = self.conv_layers[i].backward(y_pred, self.learning_rate, i)

    def train(self, X, y, learning_rate=0.01, epochs=10):
        # Entrena la CNN
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            # Propagación hacia adelante
            y_pred = self.forward(X)

            # Propagación hacia atrás
            self.backward(X, y, y_pred)

            # Calcula la pérdida
            loss = self.loss(y, y_pred)
            print('Epoch {}: loss = {}'.format(epoch + 1, loss))

    def predict(self, X):
        # Realiza predicciones
        return self.forward(X)
    
    def loss(self, y, y_pred):
        # Calcula la pérdida
        return -np.sum(y * np.log(y_pred + 1e-12)) / y.shape[0]
    
    def accuracy(self, y, y_pred):
        # Calcula la precisión
        return np.sum(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)) / y.shape[0]
    
    def summary(self):
        # Muestra un resumen de la CNN
        print('Input shape:', self.input_shape)
        print('Number of classes:', self.num_classes)
        print('Weight initialization:', self.weight_init)
        print('Learning rate:', self.learning_rate)
        print('Kernel sizes:', self.kernel_sizes)
        print('Strides:', self.strides)
        print('Pool sizes:', self.pool_sizes)
        print('Pool strides:', self.pool_strides)
        print('Convolution layers:')
        for i in range(len(self.kernel_sizes)):
            print('Layer {}:'.format(i + 1))
            print('Kernel size:', self.kernel_sizes[i])
            print('Stride:', self.strides[i])
            print('Pool size:', self.pool_sizes[i])
            print('Pool stride:', self.pool_strides[i])
        print('Fully connected layers:')
        for i in range(len(self.fc_layers)):
            print('Layer {}:'.format(i + 1))
            print('Number of neurons:', self.fc_layers[i].weights.shape[1])
            print('Activation function:', self.fc_layers[i].activation)
        print('Number of trainable parameters:', self.num_parameters())

    def num_parameters(self):
        # Calcula el número de parámetros entrenables de la CNN
        num_parameters = 0
        for i in range(len(self.kernel_sizes)):
            num_parameters += self.kernel_sizes[i] * (self.kernel_sizes[i] * self.kernel_sizes[i] + 1)
        num_parameters += self.fc_layers[0].weights.shape[0] * self.fc_layers[0].weights.shape[1] + self.fc_layers[0].weights.shape[1]
        return num_parameters
    
class Convolution:
    def __init__(self, weights, biases, stride, kernel_size):
        
        # Inicializa los pesos y sesgos de la capa de convolución
        self.weights = weights  # Pesos de la capa de convolución
        self.biases = biases  # Sesgos de la capa de convolución
        self.stride = stride  # Stride de la capa de convolución
        self.kernel_size = kernel_size  # Tamaño del kernel de la capa de convolución

        self.input_shape = None  # Forma de entrada de la capa
        self.output_shape = None  # Forma de salida de la capa

    def forward(self, X):
        # Propagación hacia adelante de la capa de convolución
        # X: entrada de la capa
        self.input_shape = X.shape  # Guarda la forma de entrada de la capa
        batch_size, input_height, input_width, input_channels = self.input_shape  # Obtiene la forma de entrada de la capa
        self.output_shape = (batch_size, int((input_height - self.kernel_size) / self.stride + 1),
                             int((input_width - self.kernel_size) / self.stride + 1), self.weights.shape[-1])  # Calcula la forma de salida de la capa

        # Inicializa la salida de la capa de convolución
        output = np.zeros(self.output_shape)

        # Aplica la operación de convolución
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                output[:, i, j, :] = np.sum(X[:, i * self.stride:i * self.stride + self.kernel_size,
                                            j * self.stride:j * self.stride + self.kernel_size, :, np.newaxis] *
                                            self.weights[np.newaxis, :, :, :], axis=(1, 2, 3)) + self.biases

        return output

    def backward(self, dLdY, learning_rate, layer):
        # Propagación hacia atrás de la capa de convolución
        # dLdY: derivada de la función de pérdida con respecto a la salida de la capa
        # learning_rate: tasa de aprendizaje
        # layer: número de la capa

        # Inicializa la derivada de la función de pérdida con respecto a la entrada de la capa
        dLdX = np.zeros(self.input_shape)

        # Calcula la derivada de la función de pérdida con respecto a los pesos de la capa
        dLdW = np.zeros(self.weights.shape)
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                dLdW += np.sum(self.input_shape[0] * dLdY[:, i, j, :, np.newaxis] *
                               self.weights[np.newaxis, :, :, :], axis=0)
                
        # Calcula la derivada de la función de pérdida con respecto a los sesgos de la capa
        dLdb = np.sum(dLdY, axis=(0, 1, 2))

        # Actualiza los pesos y sesgos de la capa
        self.weights -= learning_rate * dLdW
        self.biases -= learning_rate * dLdb

        # Calcula la derivada de la función de pérdida con respecto a la entrada de la capa
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                dLdX[:, i * self.stride:i * self.stride + self.kernel_size,
                     j * self.stride:j * self.stride + self.kernel_size, :] += np.sum(
                    dLdY[:, i, j, :, np.newaxis] * self.weights[np.newaxis, :, :, :], axis=4)
                
        return dLdX
    
class MaxPooling:
    def __init__(self, stride, pool_size):
        # Inicializa el stride y el tamaño del filtro de la capa de max pooling
        self.stride = stride  # Stride de la capa de max pooling
        self.pool_size = pool_size  # Tamaño del filtro de la capa de max pooling

        # Define otros atributos, como la forma de la salida de la capa
        self.input_shape = None  # Forma de entrada de la capa
        self.output_shape = None  # Forma de salida de la capa

    def forward(self, X):
        # Propagación hacia adelante de la capa de max pooling
        # X: entrada de la capa
        self.input_shape = X.shape  # Guarda la forma de entrada de la capa
        batch_size, input_height, input_width, input_channels = self.input_shape  # Obtiene la forma de entrada de la capa
        self.output_shape = (batch_size, int((input_height - self.pool_size) / self.stride + 1),
                             int((input_width - self.pool_size) / self.stride + 1), input_channels)  # Calcula la forma de salida de la capa

        # Inicializa la salida de la capa de max pooling
        output = np.zeros(self.output_shape)

        # Aplica la operación de max pooling
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                output[:, i, j, :] = np.max(X[:, i * self.stride:i * self.stride + self.pool_size,
                                            j * self.stride:j * self.stride + self.pool_size, :], axis=(1, 2))

        return output
    
    def backward(self, dLdY, learning_rate, layer):
        # Propagación hacia atrás de la capa de max pooling
        # dLdY: derivada de la función de pérdida con respecto a la salida de la capa
        # learning_rate: tasa de aprendizaje
        # layer: número de la capa

        # Inicializa la derivada de la función de pérdida con respecto a la entrada de la capa
        dLdX = np.zeros(self.input_shape)

        # Calcula la derivada de la función de pérdida con respecto a la entrada de la capa
        for i in range(self.output_shape[1]):
            for j in range(self.output_shape[2]):
                dLdX[:, i * self.stride:i * self.stride + self.pool_size,
                     j * self.stride:j * self.stride + self.pool_size, :] += dLdY[:, i, j, :, np.newaxis] * (
                        self.input_shape[1] * self.input_shape[2] * self.input_shape[3] == np.max(
                        self.input_shape[1] * self.input_shape[2] * self.input_shape[3], axis=(1, 2))[:, np.newaxis, np.newaxis, np.newaxis])
                
        return dLdX
    
class FullyConnected:
    def __init__(self, weights, biases, activation='sigmoid'):
        # Inicializa los pesos y sesgos de la capa completamente conectada
        self.weights = weights  # Pesos de la capa completamente conectada
        self.biases = biases  # Sesgos de la capa completamente conectada
        self.activation = activation  # Función de activación de la capa completamente conectada

        # Define otros atributos, como la forma de la salida de la capa
        self.input_shape = None  # Forma de entrada de la capa
        self.output_shape = None  # Forma de salida de la capa

    def forward(self, X):
        # Propagación hacia adelante de la capa completamente conectada
        # X: entrada de la capa
        self.input_shape = X.shape  # Guarda la forma de entrada de la capa
        batch_size = self.input_shape[0]  # Obtiene la forma de entrada de la capa
        self.output_shape = (batch_size, self.weights.shape[1])  # Calcula la forma de salida de la capa

        # Inicializa la salida de la capa completamente conectada
        output = np.zeros(self.output_shape)

        # Aplica la operación de multiplicación de matrices
        output = np.dot(X, self.weights) + self.biases

        # Aplica la función de activación
        if self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        elif self.activation == 'relu':
            output = np.maximum(0, output)

        return output

    def backward(self, dLdY, learning_rate, layer):
        # Propagación hacia atrás de la capa completamente conectada
        # dLdY: derivada de la función de pérdida con respecto a la salida de la capa
        # learning_rate: tasa de aprendizaje
        # layer: número de la capa

        # Inicializa la derivada de la función de pérdida con respecto a la entrada de la capa
        dLdX = np.zeros(self.input_shape)

        # Calcula la derivada de la función de pérdida con respecto a los pesos de la capa
        dLdW = np.dot(self.input_shape[0] * dLdY.T, self.weights.T)

        # Calcula la derivada de la función de pérdida con respecto a los sesgos de la capa
        dLdb = np.sum(dLdY, axis=0)

        # Actualiza los pesos y sesgos de la capa
        self.weights -= learning_rate * dLdW.T
        self.biases -= learning_rate * dLdb

        # Calcula la derivada de la función de pérdida con respecto a la entrada de la capa
        dLdX = np.dot(dLdY, self.weights.T)

        return dLdX
    

# Carga los datos
train_X = np.load('train_X.npy')
train_label = np.load('train_label.npy')
valid_X = np.load('valid_X.npy')
valid_label = np.load('valid_label.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')



# Define la CNN
cnn = CNN(input_shape=(21, 28, 3), num_classes=10, weight_init='xavier')

# Muestra un resumen de la CNN
cnn.summary()

# Entrena la CNN
cnn.train(train_X, train_label, learning_rate=0.01, epochs=10)

# Realiza predicciones
preds = cnn.predict(test_X)

# Calcula la precisión
accuracy = cnn.accuracy(valid_label, preds)
print('Accuracy:', accuracy)

# Muestra algunas predicciones
for i in range(10):
    print('Predicted:', np.argmax(preds[i]), 'Actual:', test_Y[i])

