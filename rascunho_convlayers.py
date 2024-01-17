import numpy as np

from classes.data_configuration import Epochs
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


train_epochs = Epochs.dict_from_subject_name("K01", "train")
classes = list(train_epochs.keys())

labels_train = np.array([])
data_train = np.array([])

# Junta todas as amostras de um mesmo sujeito em um unico array e muda a indexação para o padrão do tensorflow
for cls, epoch in train_epochs.items():

    data_smp = epoch.data
    reshaped_data = np.zeros([data_smp.shape[2], data_smp.shape[0], data_smp.shape[1]])
    for nn in range(data_smp.shape[2]):
        reshaped_data[nn] = data_smp[:, :, nn]
        labels_train = np.append(labels_train, cls)

    try:
        data_train = np.append(data_train, reshaped_data, axis=0)
    except ValueError:
        data_train = reshaped_data


##########################################################

test_epochs = Epochs.dict_from_subject_name("K01", "test")
classes = list(test_epochs.keys())

labels_test = np.array([])
data_test = np.array([])

# Junta todas as amostras de um mesmo sujeito em um unico array e muda a indexação para o padrão do tensorflow
for cls, epoch in test_epochs.items():
    labels_test = np.append(labels_test, np.repeat(cls, epoch.n_trials))

    data_smp = epoch.data
    reshaped_data = np.zeros([data_smp.shape[2], data_smp.shape[0], data_smp.shape[1]])
    for nn in range(data_smp.shape[2]):
        reshaped_data[nn] = data_smp[:, :, nn]

    try:
        data_test = np.append(data_test, reshaped_data, axis=0)
    except ValueError:
        data_test = reshaped_data

# Embaralha os dados contidos nas epocas
p = np.random.permutation(len(data_test))
data_test = data_test[p]
labels_test = labels_test[p]

p = np.random.permutation(len(data_train))
data_train = data_train[p]
labels_train = labels_train[p]

# classes = {1: 'l', 2: 'r', 3: 'f', 4: 't'}
classes = {
    1: 'lh',    # left hand
    2: 'rh',    # right hand
    3: 'n',     # neutral
    4: 'll',    # left leg
    5: 't',     # tongue
    6: 'rl',    # right leg
    # 91: 'b',    # beginning
    # 92: 'e',    # experiment end
    # 99: 'r'     # inicial relaxing
}
classes_dict = {i: j for j, i in classes.items()}
labels_train = np.array([classes_dict[i] for i in labels_train]) - 1
labels_test = np.array([classes_dict[i] for i in labels_test]) - 1


model = models.Sequential([
    layers.InputLayer(input_shape=(data_train.shape[1], data_train.shape[2], 1)),
    layers.Conv2D(25, (1, 6), padding='same', activation='relu'),
    layers.Conv2D(25, (1, 6), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Activation(tf.nn.selu),
    layers.AveragePooling2D((1, 3), (1, 2)),
    layers.Dropout(0.4),
    layers.Conv2D(50, (1, 6), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Activation(tf.nn.selu),
    layers.AveragePooling2D((1, 3), (1, 2)),
    layers.Dropout(0.4),
    layers.Conv2D(100, (1, 6), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Activation(tf.nn.selu),
    layers.AveragePooling2D((1, 3), (1, 2)),
    layers.Dropout(0.4),
    layers.Conv2D(200, (1, 6), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Activation(tf.nn.selu),
    layers.AveragePooling2D((1, 3), (1, 2)),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(len(classes)),
    layers.Activation(tf.nn.softmax),
])


# model = models.Sequential([
#     layers.Rescaling(1./100, input_shape=(22, 626, 1)),
#     layers.Conv2D(25, (3, 20), padding='same', activation='relu'),
#     layers.Conv2D(25, (3, 20), padding='same', activation='relu'),
#     layers.MaxPooling2D((1, 10), (1, 5)),
#     layers.Conv2D(25, (3, 20), padding='same', activation='relu'),
#     layers.AveragePooling2D((1, 10), (1, 5)),
#     layers.Conv2D(25, (3, 20), padding='same', activation='relu'),
#     layers.AveragePooling2D((3, 3), (3, 3)),
#     layers.Flatten(),
#     layers.Dense(16, activation="relu"),
#     layers.Dense(4),
# ])

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(22, 626, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(4))

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

data_test = data_test.reshape([data_test.shape[0], data_test.shape[1], data_test.shape[2], 1])
data_train = data_train.reshape([data_train.shape[0], data_train.shape[1], data_train.shape[2], 1])

history = model.fit(
    data_train, labels_train, epochs=50,
    validation_data=(data_test, labels_test)
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(data_test,  labels_test, verbose=2)

print(test_loss)
print(test_acc)
plt.show()

...
...
...
...
