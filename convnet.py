# simple version, No Data Augmentation, No Dropout, No Regularization
import numpy
import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


class AccHistory_train(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('acc'))


class AccHistory_valid(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('val_acc'))

seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
# lrate = 0.01
# decay = lrate/epochs
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

t_accuracy_history = AccHistory_train()
v_accuracy_history = AccHistory_valid()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=100,
          callbacks=[t_accuracy_history, v_accuracy_history])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

ta = t_accuracy_history.accuracy
va = v_accuracy_history.accuracy

fig, ax = plt.subplots()
line_1, = ax.plot(ta, label='Inline label')
line_2, = ax.plot(va, 'r-', label='Inline label')
# Overwrite the label by calling the method.
line_1.set_label('training set')
line_2.set_label('validation set')
ax.legend()
ax.set_ylabel('Accuracy')
ax.set_title('convnet')
ax.set_xlabel('epoch')
plt.show()