import numpy as np

class CNN:
    def __init__(self, num_classes, input_shape=(21, 28, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape

        # Inicialización de los pesos y sesgos de las capas convolucionales
        self.conv1_filters = 32
        self.conv1_filter_size = 3
        self.conv1_stride = 1
        self.conv1_pad = 1
        self.conv1_weights = np.random.randn(self.conv1_filters, input_shape[2], self.conv1_filter_size, self.conv1_filter_size)
        self.conv1_bias = np.zeros((self.conv1_filters, 1))
        
        self.conv2_filters = 64
        self.conv2_filter_size = 3
        self.conv2_stride = 1
        self.conv2_pad = 1
        self.conv2_weights = np.random.randn(self.conv2_filters, self.conv1_filters, self.conv2_filter_size, self.conv2_filter_size)
        self.conv2_bias = np.zeros((self.conv2_filters, 1))
        
        # Capas completamente conectadas
        self.fc1_units = 128
        self.fc1_weights = np.random.randn(self.fc1_units, self.conv2_filters * (input_shape[0] // 4) * (input_shape[1] // 4))
        self.fc1_bias = np.zeros((self.fc1_units, 1))
        
        self.fc2_weights = np.random.randn(num_classes, self.fc1_units)
        self.fc2_bias = np.zeros((num_classes, 1))

    def conv_layer(self, x, weights, bias, stride, pad, debug=False):
        # Implementar la capa convolucional
        print("conv_layer")
        print("x.shape: ", x.shape)
        (num_filter, num_canal_filter, size_filter, _) = weights.shape  # Dimensiones del filtro
        batch_size, in_dim1, in_dim2, num_canal = x.shape  # Dimensiones de la entrada
        out_dim1 = int((in_dim1 - size_filter + 2 * pad) / stride) + 1  # Dimensiones de la salida
        out_dim2 = int((in_dim2 - size_filter + 2 * pad) / stride) + 1  # Dimensiones de la salida
        output = np.zeros((batch_size, out_dim1, out_dim2, num_filter))
        print("weights.shape: ", weights.shape)
        print("output.shape: ", output.shape)

        # Aplicar el filtro a la entrada
        for batch in range(batch_size):
            for filter_num in range(num_filter):
                for channel in range(num_canal):
                    # Aplicar el filtro a la entrada
                    input_pad = np.pad(x[batch, :, :, channel], pad_width=pad, mode='constant', constant_values=0)
                    for i in range(0, out_dim1, stride):
                        for j in range(0, out_dim2, stride):
                            if debug:
                                print(f"batch: {batch}, filter_num: {filter_num}, channel: {channel}, i: {i}, j: {j}")
                            output[batch, i, j, filter_num] += np.sum( input_pad[i:i+size_filter, j:j+size_filter] * weights[filter_num, :, :, channel])
                output[batch, :, :, filter_num] += bias[filter_num]

        return output
    
    def conv_layer_backward(self, d_output, x, weights, stride, pad):
        # Implementar la retropropagación de la capa convolucional
        print("conv_layer_backward")
        print("x.shape: ", x.shape)
        (num_filter, num_canal_filter, size_filter, _) = weights.shape  # Dimensiones del filtro
        batch_size, in_dim1, in_dim2, num_canal = x.shape
        out_dim1 = int((in_dim1 - size_filter + 2 * pad) / stride) + 1
        out_dim2 = int((in_dim2 - size_filter + 2 * pad) / stride) + 1
        d_weights = np.zeros(weights.shape)
        d_bias = np.zeros((num_filter, 1))
        d_input = np.zeros(x.shape)
        input_pad = np.pad(x, pad_width=pad, mode='constant', constant_values=0)
        d_input_pad = np.zeros(input_pad.shape)

        # Derivada de la función de activación ReLU
        d_output[x <= 0] = 0

        # Derivada de los pesos y sesgos
        for batch in range(batch_size):
            for filter_num in range(num_filter):
                for channel in range(num_canal):
                    for i in range(0, out_dim1, stride):
                        for j in range(0, out_dim2, stride):
                            d_weights[filter_num, :, :, channel] += input_pad[i:i+size_filter, j:j+size_filter] * d_output[batch, i, j, filter_num]
                            d_input_pad[i:i+size_filter, j:j+size_filter] += weights[filter_num, :, :, channel] * d_output[batch, i, j, filter_num]
                d_bias[filter_num] += np.sum(d_output[batch, :, :, filter_num])

        # Derivada de la entrada
        d_input = d_input_pad[pad:-pad, pad:-pad, :]
        return d_input
    
    def maxpool_layer(self, x, pool_size=(2, 2)):
        # Implementar la capa de pooling
        print("maxpool_layer")
        print("x.shape: ", x.shape)
        (pool_height, pool_width) = pool_size
        batch_size, in_dim1, in_dim2, num_canal = x.shape
        out_dim1 = int(in_dim1 / pool_height)
        out_dim2 = int(in_dim2 / pool_width)
        output = np.zeros((batch_size, out_dim1, out_dim2, num_canal))
        print("output.shape: ", output.shape)

        for batch in range(batch_size):
            for channel in range(num_canal):
                for i in range(0, out_dim1):
                    for j in range(0, out_dim2):
                        output[batch, i//pool_height, j//pool_width, channel] = np.max(x[batch, i:i+pool_height, j:j+pool_width, channel])

        return output
    
    def flatten(self, x):
        # Implementar la capa de aplanamiento
        print("flatten")
        return x.reshape(x.shape[0], -1)
    
    def fully_connected(self, x, weights, bias):
        # Implementar la capa completamente conectada
        print("fully_connected")
        return np.dot(weights, x) + bias
    
    def softmax(self, x):
        # Implementar la función softmax
        print("softmax")
        return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
    
    def predict(self, x):
        # Implementar la predicción
        print("predict")
        self.forward(x)
        return np.argmax(self.softmax(self.fc2_output), axis=0)
    
    def forward(self, x):
        # Implementar la propagación hacia adelante
        print("forward")
        # Primera capa convolucional
        conv1_output = self.conv_layer(x, self.conv1_weights, self.conv1_bias, self.conv1_stride, self.conv1_pad)
        conv1_output = np.maximum(conv1_output, 0)
        conv1_output = self.maxpool_layer(conv1_output, pool_size=(2, 2))

        # Segunda capa convolucional
        conv2_output = self.conv_layer(conv1_output, self.conv2_weights, self.conv2_bias, self.conv2_stride, self.conv2_pad, debug=True)
        conv2_output = np.maximum(conv2_output, 0)
        conv2_output = self.maxpool_layer(conv2_output, pool_size=(2, 2))

        # Aplanar la salida de la segunda capa convolucional
        flattened = self.flatten(conv2_output)
        
        # Primera capa completamente conectada
        fc1_output = self.fully_connected(flattened, self.fc1_weights, self.fc1_bias)
        fc1_output = np.maximum(fc1_output, 0)

        # Segunda capa completamente conectada
        fc2_output = self.fully_connected(fc1_output, self.fc2_weights, self.fc2_bias)
        output = self.softmax(fc2_output)
    
    def backward(self, x, y, learning_rate):
        # Implementar la retropropagación
        print("backward")
        batch_size = x.shape[0]

        # Primera capa completamente conectada
        d_fc2_weights = np.dot(self.fc1_output, (self.softmax(self.fc2_output) - y).T)
        d_fc2_bias = np.sum(self.softmax(self.fc2_output) - y, axis=1, keepdims=True)
        d_fc1_output = np.dot(self.fc2_weights.T, self.softmax(self.fc2_output) - y)

        # Segunda capa completamente conectada
        d_fc1_output[self.fc1_output <= 0] = 0  # Derivada de la función ReLU
        d_fc1_weights = np.dot(d_fc1_output, self.flattened.T)
        d_fc1_bias = np.sum(d_fc1_output, axis=1, keepdims=True)
        d_flattened = np.dot(self.fc1_weights.T, d_fc1_output)

        # Aplanar la salida de la segunda capa convolucional
        d_flattened = d_flattened.reshape(self.conv2_output.shape)

        # Segunda capa convolucional y la capa de pooling
        d_conv2_output = d_flattened
        d_conv2_output[self.conv2_output <= 0] = 0  # Derivada de la función ReLU
        d_conv2_weights = np.zeros(self.conv2_weights.shape)
        d_conv2_bias = np.sum(d_conv2_output, axis=(2, 3), keepdims=True)

        d_conv1_output = np.zeros(self.conv1_output.shape)
        d_conv1_weights = np.zeros(self.conv1_weights.shape)
        d_conv1_bias = np.zeros(self.conv1_bias.shape)

        # Retropropagar el gradiente a través de la capa de pooling y la primera capa convolucional
        for i in range(batch_size):
            for j in range(self.conv2_filters):
                d_conv2_weights[j] += self.conv_layer_backward(d_conv2_output[j, i], self.conv1_output[:, i], self.conv2_weights[j], self.conv2_stride, self.conv2_pad)
                d_conv2_bias[j] += d_conv2_output[j, i].sum()

                for k in range(self.conv1_filters):
                    d_conv1_output[k, i] += self.conv_layer_backward(d_conv2_output[j, i], x[i], self.conv1_weights[k], self.conv1_stride, self.conv1_pad)
                    d_conv1_weights[k] += self.conv_layer_backward(d_conv2_output[j, i], x[i], self.conv1_weights[k], self.conv1_stride, self.conv1_pad)
                    d_conv1_bias[k] += d_conv2_output[j, i].sum()

        # Actualizar los pesos y sesgos
        self.fc2_weights -= learning_rate * d_fc2_weights
        self.fc2_bias -= learning_rate * d_fc2_bias
        self.fc1_weights -= learning_rate * d_fc1_weights
        self.fc1_bias -= learning_rate * d_fc1_bias
        self.conv2_weights -= learning_rate * d_conv2_weights
        self.conv2_bias -= learning_rate * d_conv2_bias
        self.conv1_weights -= learning_rate * d_conv1_weights
        self.conv1_bias -= learning_rate * d_conv1_bias

    def train(self, X, y, learning_rate, num_epochs, batch_size=32):
        # Implementar el entrenamiento del modelo
        for epoch in range(num_epochs):
            print("Epoch %d" % (epoch + 1))
            for i in range(0, X.shape[0], batch_size):
                x_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                self.forward(x_batch)
                self.backward(x_batch, y_batch, learning_rate)


# Carga los datos
train_data = np.load('train_X.npy')
train_label = np.load('train_label.npy')
valid_data = np.load('valid_X.npy')
valid_label = np.load('valid_label.npy')
test_data = np.load('test_X.npy')
test_label = np.load('test_Y.npy')

# train_data.shape = (num_examples, 21, 28, 3)
# train_label.shape = (num_examples, 10)
# valid_data.shape = (num_examples, 21, 28, 3)
# valid_label.shape = (num_examples, 10)

num_classes = 10  # Número de clases en tu problema de clasificación
model = CNN(num_classes)
model.train(train_data, train_label, learning_rate=0.01, num_epochs=10, batch_size=32)
prediction = model.predict(test_data)







