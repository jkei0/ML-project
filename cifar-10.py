"""Short script to load CIFAR-10 dataset and classify it by using convolutional neural network"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_data():

    train_x = []
    test_x = []

    # Data Paths
    data_batch_1 = 'recourses/CIFAR-10/cifar-10-batches-py/data_batch_1'
    data_batch_2 = 'recourses/CIFAR-10/cifar-10-batches-py/data_batch_2'
    data_batch_3 = 'recourses/CIFAR-10/cifar-10-batches-py/data_batch_3'
    data_batch_4 = 'recourses/CIFAR-10/cifar-10-batches-py/data_batch_4'
    data_batch_5 = 'recourses/CIFAR-10/cifar-10-batches-py/data_batch_5'
    test_batch = 'recourses/CIFAR-10/cifar-10-batches-py/test_batch'
    meta_batch = 'recourses/CIFAR-10/cifar-10-batches-py/batches.meta'

    # Training data
    dict_data_1 = unpickle(data_batch_1)
    dict_data_2 = unpickle(data_batch_2)
    dict_data_3 = unpickle(data_batch_3)
    dict_data_4 = unpickle(data_batch_4)
    dict_data_5 = unpickle(data_batch_5)

    # Combine images to one matrix
    tr_data = np.vstack((dict_data_1[b'data'],dict_data_2[b'data'],dict_data_3[b'data'],
                         dict_data_4[b'data'],dict_data_5[b'data']))

    # Combine labels to one matrix
    tr_labels = np.concatenate((dict_data_1[b'labels'],dict_data_2[b'labels'],dict_data_3[b'labels'],
                           dict_data_4[b'labels'],dict_data_5[b'labels']),axis=0)

    # Test data
    dict_data_test = unpickle(test_batch)

    te_data = dict_data_test[b'data']
    te_labels = dict_data_test[b'labels']

    # Get meta data
    meta_data = unpickle(meta_batch)
    label_names = meta_data[b'label_names']

    # Make pictures
    for image in tr_data:
        train_x.append(np.transpose(np.reshape(image,(3,32,32)),(1,2,0)))

    for image in te_data:
        test_x.append(np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0)))

    # Make numpy arrays
    train_x = np.array(train_x)
    test_x = np.array(test_x)

    # Transform image type
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')

    # Scale images
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    return train_x, tr_labels, test_x, te_labels, label_names


def change_labels(labels):
    new_labels = np.zeros((len(labels),10))

    for n in range(len(labels)):
        mat = np.zeros(10)
        mat[labels] = 1
        new_labels[n] = mat
    return new_labels


def show_pic(index, tr_data, tr_labels, label_names):

    pic = tr_data[index]

    # Get label
    label = tr_labels[index]
    label_name = label_names[label]
    title = label_name.decode('utf-8')

    # Show Image
    plt.imshow(pic, interpolation='bilinear')
    plt.title(title)
    plt.ion()
    plt.show()
    plt.pause(1)


def get_model(tr_data):


    # Build CNN
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,(3,3), padding='same', input_shape=tr_data.shape[1:]))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(32,(3,3)))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64,(3,3), padding='same'))
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(64,(3,3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512,activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10,activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_model_history(model_history):
    figure, axis = plt.subplots(1,2,figsize=(15,5))

    # Plot model accuracy
    axis[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axis[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axis[0].set_title('Model Accuracy')
    axis[0].set_ylabel('Accuracy')
    axis[0].set_xlabel('Epoch')
    axis[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axis[0].legend(['train', 'val'], loc='best')

    # Plot loss
    axis[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axis[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axis[1].set_title('Model Loss')
    axis[1].set_ylabel('Loss')
    axis[1].set_xlabel('Epoch')
    axis[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axis[1].legend(['train', 'val'], loc='best')
    plt.show(block=True)


if __name__ == '__main__':

    tr_data, tr_labels, te_data, te_labels, label_names = read_data()

    # Show 5 pictures
    for i in range(5):
        index = random.randrange(0, 50000)
        show_pic(index, tr_data, tr_labels, label_names)

    # Convert class labels to binary
    tr_labels = keras.utils.to_categorical(tr_labels, num_classes=10)
    te_labels = keras.utils.to_categorical(te_labels, num_classes=10)

    # Get neural network
    model = get_model(tr_data)

    # Train model
    info = model.fit(tr_data, tr_labels, epochs=10, validation_split=0.2, verbose=2)

    # Plot training
    plot_model_history(info)

    # Evaluate model
    test_loss, test_acc = model.evaluate(te_data, te_labels)

    print("Test accuracy: {}".format(test_acc))
