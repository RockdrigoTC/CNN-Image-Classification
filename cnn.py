import numpy as np

class CNN:
    def __init__(self, input_shape, num_classes, stride=1,kernel_size=3, weight_init='random'):
        self.input_shape = input_shape  # Forma de entrada
        self.num_classes = num_classes # Número de clases
        self.weight_init = weight_init # Inicialización de pesos
        self.kernel_size = kernel_size # tamaño del kernel
        self.stride = stride           # tamaño del stride

        self.layers = []  # Lista de capas
        self.pool_layers = [] # Lista de capas de agrupación
        self.conv_layers = [] # Lista de capas convolucionales
        self.fc_layers = []  # Lista de capas totalmente conectadas

    def add(self, layer):
        # Añade una capa a la CNN
        self.layers.append(layer)
        if layer.name == 'Conv':
            self.conv_layers.append(layer)
        elif layer.name == 'Pool':
            self.pool_layers.append(layer)
        elif layer.name == 'FC':
            self.fc_layers.append(layer)

    def summary(self):
        # Muestra un resumen de la CNN
        print('Input shape:', self.input_shape)
        print('---')
        for i in range(len(self.layers)):
            print('Layer', i + 1, ':', self.layers[i].name)
            print('Output shape:', self.layers[i].output.shape)
            print('Trainable weights:', self.layers[i].trainable_weights)
            print('Trainable biases:', self.layers[i].trainable_biases)
            print('Parameters:', self.layers[i].parameters)
            print('---')

    def forward(self, X):
        self.input = X  # Guarda la entrada original en self.input
        for layer in self.layers:
            X = layer.forward(X)
        return X


    def backward(self, preds, Y, learning_rate):
        # Propagación hacia atrás
        for layer in reversed(self.layers):
            preds = layer.backward(preds, Y, learning_rate)
        return preds

     
    def train(self, X, Y, epochs=10, batch_size=32, learning_rate=0.01):
        loss_function = CrossEntropyLoss()

        for epoch in range(epochs):
            print('Epoch', epoch + 1)
            total_loss = 0
            total_correct = 0

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]

                # Forward pass
                preds = cnn.forward(X_batch)
                #loss = loss_function.forward(preds, Y_batch)

                # Backward pass
                #dLdoutput = loss_function.backward(1)  # Loss gradient is 1 for backpropagation
                preds = cnn.backward(preds, Y_batch, learning_rate)

                #total_loss += loss
                #total_correct += np.sum(np.argmax(preds, axis=1) == np.argmax(Y_batch, axis=1))

            # Calculate and print average loss and accuracy for the epoch
            '''
            average_loss = total_loss / len(X)
            accuracy = total_correct / len(X)
            print('Average Loss:', average_loss)
            print('Accuracy:', accuracy)
            '''

    def predict(self, X):
        # Realiza predicciones
        preds = self.forward(X)
        return preds

    def evaluate(self, X, Y):
        # Evalúa el rendimiento de la CNN
        preds = self.predict(X)
        preds = np.argmax(preds, axis=1)
        Y = np.argmax(Y, axis=1)
        accuracy = np.mean(preds == Y)
        return accuracy
    
class CrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.softmax_output = None
        self.target = None

    def forward(self, inputs, target):
        self.target = target
        self.softmax_output = self.softmax(inputs)
        self.loss = self.cross_entropy(self.softmax_output, target)
        return self.loss

    def backward(self, dLdoutput):
        dLdinput = self.softmax_output - self.target
        return dLdinput

    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def cross_entropy(self, predictions, target):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        batch_size = predictions.shape[0]
        return -np.sum(target * np.log(predictions)) / batch_size


class Conv:
    def __init__(self, num_filters, kernel_size, stride, activation):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.name = 'Conv'
        self.input = None
        self.output = None

        # Inicializa los pesos y sesgos
        self.weights = None
        self.biases = None

    def initialize(self, input_shape):
        input_channels = input_shape[-1]
        
        # Inicialización de Glorot (Xavier)
        limit = np.sqrt(6.0 / (input_channels + self.num_filters))
        self.weights = np.random.uniform(-limit, limit, size=(self.num_filters, self.kernel_size, self.kernel_size, input_channels))
        print(self.weights.shape)
        self.biases = np.zeros((1, 1, 1, self.num_filters))

    def forward(self, input_data):
        print("Conv forward")
        print(input_data.shape)
        self.input = input_data
        self.initialize(input_data.shape)
        batch_size, input_height, input_width, input_channels = input_data.shape
        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        self.output = np.zeros((batch_size, output_height, output_width, self.num_filters))

        for i in range(output_height):
            for j in range(output_width):
                input_slice = input_data[:, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size, :]
                self.output[:, i, j, :] = np.sum(input_slice * self.weights, axis=(1, 2, 3)) + self.biases

        if self.activation == 'relu':
            self.output = np.maximum(0, self.output)

        return self.output


    def backward(self, dLdoutput, Y, learning_rate):
        print("Conv backward")
        batch_size, output_height, output_width, num_filters = dLdoutput.shape
        input_height, input_width, input_channels = self.input.shape[1:]

        dLdinput = np.zeros_like(self.input)
        dLdweights = np.zeros_like(self.weights)
        dLdbiases = np.sum(dLdoutput, axis=(0, 1, 2), keepdims=True)

        if self.activation == 'relu':
            dLdoutput = dLdoutput * (self.output > 0)

        for i in range(output_height):
            for j in range(output_width):
                input_slice = self.input[:, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size, :]
                for f in range(num_filters):
                    dLdweights[f, :, :, :] += np.sum(input_slice * (dLdoutput[:, i, j, f])[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    dLdinput[:, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size, :] += dLdoutput[:, i, j, f][:, np.newaxis, np.newaxis, np.newaxis] * self.weights[f, :, :, :]

        self.weights -= learning_rate * dLdweights
        self.biases -= learning_rate * dLdbiases

        return dLdinput
    

class Pool:
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride
        self.name = 'Pool'
        self.input = None
        self.output = None
        self.mask = None

    def forward(self, input_data):
        self.input = input_data
        batch_size, input_height, input_width, num_channels = input_data.shape
        output_height = (input_height - self.size) // self.stride + 1
        output_width = (input_width - self.size) // self.stride + 1
        self.output = np.zeros((batch_size, output_height, output_width, num_channels))
        mask = np.zeros_like(input_data)  # Nuevo array mask

        for i in range(output_height):
            for j in range(output_width):
                input_slice = input_data[:, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size, :]
                max_values = np.max(input_slice, axis=(1, 2), keepdims=True)
                mask[:, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size, :] = (input_slice == max_values[:, np.newaxis, np.newaxis, :])
                self.output[:, i, j, :] = max_values[:, 0, 0, :]

        self.mask = mask  # Actualiza self.mask con el nuevo array

        return self.output



    def backward(self, dLdoutput, Y, learning_rate):
        dLdinput = np.zeros_like(self.input)
        batch_size, output_height, output_width, num_channels = dLdoutput.shape

        for i in range(output_height):
            for j in range(output_width):
                dLdinput_slice = dLdoutput[:, i, j, :]
                dLdinput_slice = dLdinput_slice[:, np.newaxis, np.newaxis, :]
                dLdinput[:, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size, :] += dLdinput_slice[:, np.newaxis, np.newaxis, :] * self.mask[:, i * self.stride:i * self.stride + self.size, j * self.stride:j * self.stride + self.size, :]

        return dLdinput




# Carga los datos
train_X = np.load('train_X.npy')
train_label = np.load('train_label.npy')
valid_X = np.load('valid_X.npy')
valid_label = np.load('valid_label.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')


# Define la CNN
cnn = CNN(input_shape=(21, 28, 3), num_classes=10, weight_init='random')

# Añade las capas
cnn.add(Conv(num_filters=32, kernel_size=3, stride=1, activation='relu'))

cnn.add(Pool(size=2, stride=2))  # Reducirá las dimensiones a la mitad
cnn.add(Conv(num_filters=32, kernel_size=3, stride=1, activation='relu'))
cnn.add(Pool(size=2, stride=2))  # Reducirá las dimensiones a la mitad nuevamente

'''
cnn.add(FC(num_neurons=100, activation='relu'))
cnn.add(FC(num_neurons=10, activation='softmax'))
'''

# Entrena la CNN
cnn.train(train_X, train_label, epochs=10, batch_size=32, learning_rate=0.01)
