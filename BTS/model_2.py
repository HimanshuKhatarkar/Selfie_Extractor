from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *


# ## Define Model
from behind_the_scenes.model import save_and_plot_history


def CNN_model(choice):
    model2 = Sequential()
    model2.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Conv2D(32, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Conv2D(64, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))

    model2.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model2.add(Dense(output_dim=128, activation='relu', ))
    model2.add(Dropout(0.5))
    model2.add(Dense(output_dim=1, activation='sigmoid'))
    model2.summary()
    model2.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

    if choice.lower() == 'y':
        # Train model2
        model2 = training_(model2)
    else:
        model2 = load_(model2)

    return model2


def training_(model2):
    training_set = get_train_set()
    test_set = get_test_set()
    # Fit
    history = model2.fit_generator(training_set,
                                   steps_per_epoch=50,
                                   nb_epoch=40)
    save_weights(model2, training_set, test_set)
    save_and_plot_history(history.history, 'model_2_history.json')
    return model2


def load_(model2):
    # load trained weights
    model2.load_weights('behind_the_scenes/weights_2.h5')
    print("Loaded model2 from disk")

    return model2


def get_train_set():
    # Training Dataset
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=50,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    training_set = train_datagen.flow_from_directory(
        'Dataset/WhatsApp/Media/training_set_2',
        target_size=(128, 128),
        batch_size=6,
        class_mode='binary')
    return training_set


def get_test_set():
    # Testing Dataset
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_set = test_datagen.flow_from_directory(
        'Dataset/WhatsApp/Media/test_set_2',
        target_size=(128, 128),
        batch_size=6,
        class_mode='binary')
    return test_set


def save_weights(model2, X, y):
    model2.save_weights("behind_the_scenes/weights_2.h5")
    print("Saved model2 to disk")
    return model2
