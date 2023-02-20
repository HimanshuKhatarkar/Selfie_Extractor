import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *


# ## Define Model
def CNN_model(choice):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(124, 124, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    if choice.lower() == 'y':
        # Train model
        model = training_(model)
    else:
        model = load_(model)

    return model


def training_(model):
    training_set = get_train_set()
    test_set = get_test_set()
    # Fit
    history = model.fit_generator(training_set,
                                  steps_per_epoch=50,
                                  nb_epoch=40,
                                  validation_data=test_set,
                                  nb_val_samples=276)
    save_weights(model, training_set, test_set)
    save_and_plot_history(history.history, 'model_1_history.json')
    return model


def load_(model):
    # load trained weights
    model.load_weights('behind_the_scenes/weights.h5')
    print("Loaded model from disk")

    return model


def get_train_set():
    # Training Dataset
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    training_set = train_datagen.flow_from_directory(
        'Dataset/WhatsApp/Media/training_set',
        target_size=(124, 124),
        batch_size=4,
        class_mode='binary')
    return training_set


def get_test_set():
    # Testing Dataset
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(
        'Dataset/WhatsApp/Media/test_set',
        target_size=(124, 124),
        batch_size=4,
        class_mode='binary')
    return test_set


def save_weights(model, X, y):
    model.save_weights("behind_the_scenes/weights.h5")
    print("Saved model to disk")
    return model


def save_and_plot_history(history, filename):
    import json
    with open('behind_the_scenes/{}'.format(filename), 'w') as file:
        file.write(json.dumps(history))
    max_validation_accuracy = max(history['val_acc'])
    epoch = (history['val_acc']).index(max_validation_accuracy) + 1
    print(max_validation_accuracy, epoch)
    plt.scatter([epoch], [max_validation_accuracy], color='red')
    plt.plot(list(range(1, len(history['val_acc']) + 1)), history['val_acc'], color='blue', marker='o',
             linestyle='dashed')
    plt.plot(list(range(1, len(history['val_acc']) + 1)), history['acc'], color='green')
    plt.title(filename.replace('.json', '').replace('_', ' '))
    plt.xlabel('Epochs')
    plt.ylabel('Validation accuracy')
    plt.show()
