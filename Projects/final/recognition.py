import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# seed used to verify with the tutorial I was following
# seed = 21

# importing a premade preprocessed testing data set from keras
from keras.datasets import cifar10

# load the data to be trained
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizing the data by setting it to a float type and dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# Specifiying the number of classes inside the data set. 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# selecting a model. Sequential is the most commonly used.
model = Sequential()

# first convolution layer, 32 channels/filters and its a 3 by 3 filter size. 
# Padding same just means we arent changing the size
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))

# most common activation is relu
model.add(Activation('relu'))

# this is a way of stringing those two together.
# model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

# dropout added to keep our network from over training to the data set or overfitting
model.add(Dropout(0.2))

# this makes all the layers use the same distribution for the activations.
model.add(BatchNormalization())

# another convolution with a bigger filter size 64 instead of 32
model.add(Conv2D(64, (4, 4), padding='same'))
model.add(Activation('relu'))

# first pooling layer. helps the network learn more releant patterns. 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# repeat to give more representation.
model.add(Conv2D(64, (4, 4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# size up again for filters. try to keep these as powers of 2    
model.add(Conv2D(128, (4, 4), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# flatten puts on the data into vectors or lists. 
model.add(Flatten())
model.add(Dropout(0.2))

# create the first dense layers. 
# We cut down the 256 a few times until it becomes the number of classes in the set of data  in this case 10
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
    
model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# last dense layer of size 10 using the softmax activation
# it selects the neuron with the highest probablitity as its output. 
# votes that an image belongs to a specific class 
model.add(Dense(class_num))
model.add(Activation('softmax'))

# the number of rounds of training that will occure 
epochs = 5

# tunes weights of the network to approach the lowest loss 
# Adam is the most common because it works well for most networks. 
optimizer = 'adam'

# Compile the model and select accuracy as what we are going for. 
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# print the model to screen, see its stats.
print(model.summary())

# used to seed with the tutorial I followed orginially
# numpy.random.seed(seed)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)

# this evaluates the model and tells you how well it preformed. 
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))