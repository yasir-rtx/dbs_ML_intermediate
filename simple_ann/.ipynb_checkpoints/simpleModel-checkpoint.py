from keras import datasets, models, layers
from tensorflow import nn
from tensorflow import _optimizers
from os import system

mnist = datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(512, activation=nn.relu),
    layers.Dense(256, activation=nn.relu),
    layers.Dense(10, activation=nn.softmax)
])

model.compile(  
    optimizer = _optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

system('clear')

model.fit(
    x_train, 
    y_train, 
    epochs = 30,
    validation_data=(x_test, y_test)
)