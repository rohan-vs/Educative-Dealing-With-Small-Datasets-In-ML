from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_preprocess():
    # Loading the cifar-10 dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()

    # Converting to float and normalizing the pixels
    trainX = trainX.astype('float32')
    trainX = trainX / 255.0
    testX = testX.astype('float32')
    testX = testX / 255.0

    # Applying one-hot encoding to the lables
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    return trainX, trainY, testX, testY

def plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()