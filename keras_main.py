from datetime import datetime
import os
import numpy as np

from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam


# to split D: 70% for training, 30% for testing
def split_train_test(X_, y_):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_,
                                                            train_size=0.7,
                                                            test_size=0.3)
    return X_train_, X_test_, y_train_, y_test_


def load_image(img_path, size_):
    img = io.imread(img_path)
    # img.shape -> mostly (224, 224, 3)

    img = resize(img, (size_, size_), anti_aliasing=False)

    return img


def create_data(folder_path_, size_, limit_):
    X_, y_ = [], []

    folders = os.listdir(folder_path_)
    for label, folder in enumerate(folders):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(current_time, "|", folder, "| adding files...")

        path_ = "{}/{}".format(folder_path_, folder)
        files_ = os.listdir(path_)

        for no_, file_ in enumerate(files_):
            if no_ == limit_:
                break

            img_path = "{}/{}".format(path_, file_)
            img = load_image(img_path, size_)

            X_.append(img)
            y_.append(label)

    X_ = np.array(X_)
    y_ = np.array(y_)

    return X_, y_


folder_path = "tr_signLanguage_dataset"
resize_ = 32    # img.shape(64,64)
limit = -1      # -1 for no limit

# X, y = create_data(folder_path, resize_, limit)
# X_train, X_test, y_train, y_test = split_train_test(X, y)

# https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/
# https://www.cs.toronto.edu/%7Ekriz/cifar.html

# Model configuration
batch_size = 64
img_width, img_height, img_num_channels = resize_, resize_, 3
loss_function = sparse_categorical_crossentropy
# no_classes = 23
no_classes = 10
no_epochs = 10
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# len(X_test)        10000
# len(X_train)       50000
# X_train.shape     (50000, 32, 32, 3)
# X_train[0].shape  (32, 32, 3)


# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)


# Parse numbers as floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Scale data
X_train = X_train / 255
X_test = X_test / 255


# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbosity,
                    validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

with open("Keras_Score.txt", "w") as txt_file_:
    txt_file_.write(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
