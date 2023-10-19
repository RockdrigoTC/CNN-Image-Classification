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

     
    def train(self, X, Y, epochs=1, batch_size=32, learning_rate=0.01):
        for epoch in range(epochs):
            print('Epoch', epoch + 1)
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]
                print('X_batch shape:', X_batch.shape)
                print('Y_batch shape:', Y_batch.shape)
                preds = self.forward(X_batch)
                print('preds shape:', preds.shape)
                #preds = self.backward(preds, Y_batch, learning_rate)
                print('preds shape:', preds.shape)
                #print('Loss:', self.loss(Y_batch, preds))
                #print('Accuracy:', self.accuracy(Y_batch, preds))

    def predict(self, X):
        # Realiza predicciones
        preds = self.forward(X)
        return preds

    def accuracy(self, Y, preds):
        # Calcula la precisión
        return (np.argmax(preds, axis=1) == np.argmax(Y, axis=1)).mean()
    
    def loss(self, Y, preds):
        # Calcula la pérdida
        return -np.mean(Y * np.log(preds))
    

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

        self.biases = np.zeros((1, 1, 1, self.num_filters))

    def forward(self, input_data):
        print("Conv forward")
        print(input_data.shape)
        self.input = input_data
        batch_size, input_height, input_width, input_channels = input_data.shape
        self.initialize(input_data.shape)

        print("input_height: ",input_height)
        print("input_width: ",input_width)
        print("input_channels: ",input_channels)
        print("batch_size: ",batch_size)

        output_height = (input_height - self.kernel_size) // self.stride + 1
        output_width = (input_width - self.kernel_size) // self.stride + 1
        self.output = np.zeros((batch_size, output_height, output_width, self.num_filters))

        print("output_height: ",output_height)
        print("output_width: ",output_width)
        print("output_shape: ",self.output.shape)

        for i in range(output_height):
            for j in range(output_width):
                input_slice = input_data[:, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size, :]
                self.output[:, i, j, :] = np.sum(input_slice * self.weights, axis=(1, 2, 3)) + self.biases

        if self.activation == 'relu':
            self.output = np.maximum(0, self.output)

        return self.output

    def backward(self, dLdoutput, Y, learning_rate):
        pass





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
'''
cnn.add(Pool(size=2, stride=2))
cnn.add(Conv(num_filters=32, kernel_size=3, stride=1, activation='relu'))
cnn.add(Pool(size=2, stride=2))
cnn.add(FC(num_neurons=100, activation='relu'))
cnn.add(FC(num_neurons=10, activation='softmax'))
'''

# Muestra un resumen de la CNN
#cnn.summary()

# Entrena la CNN
cnn.train(train_X, train_label, epochs=10, batch_size=32, learning_rate=0.01)






