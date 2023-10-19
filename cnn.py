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
            print('Output shape:', self.layers[i].output_shape)
            print('Trainable weights:', self.layers[i].trainable_weights)
            print('Trainable biases:', self.layers[i].trainable_biases)
            print('Parameters:', self.layers[i].parameters)
            print('---')

    def forward(self, X):
        # Propagación hacia adelante
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
                preds = cnn.forward(X_batch)
                preds = cnn.backward(preds, Y_batch, learning_rate)
                print('Loss:', self.loss(Y_batch, preds))
                print('Accuracy:', self.accuracy(Y_batch, preds))

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
    
        
    
# Carga los datos
train_X = np.load('train_X.npy')
train_label = np.load('train_label.npy')
valid_X = np.load('valid_X.npy')
valid_label = np.load('valid_label.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')


# Define la CNN
cnn = CNN(input_shape=(21, 28, 3), num_classes=10, weight_init='ramdom')






