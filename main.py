import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np


def load_mnist():
    # load data
    (train_image, train_label), (test_image, test_label) = datasets.mnist.load_data()

    return train_image, train_label, test_image, test_label


def split_train_data(train_image, train_label, size, n):
    # shuffle target data
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    np.random.shuffle(train_image)
    np.random.seed(seed)
    np.random.shuffle(train_label)

    # split target dataset and shadow pool
    target_image = train_image[:size]
    target_label = train_label[:size]
    shadow_image_pool = train_image[size:]
    shadow_label_pool = train_label[size:]

    # create datasets for shadow models
    shadow_models_image = []
    shadow_models_label = []
    for _ in range(0, n):
        np.random.seed(seed)
        np.random.shuffle(shadow_image_pool)
        shadow_models_image.append(shadow_image_pool[:size])
        np.random.seed(seed)
        np.random.shuffle(shadow_label_pool)
        shadow_models_label.append((shadow_label_pool[:size]))

    return target_image, target_label, shadow_models_image, shadow_models_label


def split_test_data(test_image, test_label, size, n):
    seed = np.random.randint(0, 10000)
    shadow_test_image = []
    shadow_test_label = []
    for _ in range(0, n):
        np.random.seed(seed)
        np.random.shuffle(test_image)
        shadow_test_image.append(test_image[:size])
        np.random.seed(seed)
        np.random.shuffle(test_label)
        shadow_test_label.append((test_label[:size]))
    return shadow_test_image, shadow_test_label


def visualize_training(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def compile_baseline(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def baseline_target(input_shape, epochs, train_image, train_label, test_image, test_label):
    model = compile_baseline(input_shape)
    history = model.fit(train_image, train_label, epochs=epochs, validation_split=0.2)
    visualize_training(history, epochs)
    test_loss, test_acc = model.evaluate(test_image, test_label, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    probability_model = tf.keras.Sequential([model, layers.Softmax()])

    return probability_model


def baseline_shadow(input_shape, epochs, image_pool, label_pool):
    assert len(image_pool) == len(label_pool)
    n = len(image_pool)
    shadow_models = []
    for i in range(0, n):
        model = compile_baseline(input_shape)
        model.fit(image_pool[i], label_pool[i], epochs=epochs)
        probability_model = tf.keras.Sequential([model, layers.Softmax()])
        shadow_models.append(probability_model)
    return shadow_models


def train_attack(shadow_models, train_image_pool, train_label_pool, test_image_pool, test_label_pool):
    train_vectors = []
    train_labels = []
    for i, model in enumerate(shadow_models):
        # predict in vectors for ith shadow model
        prediction = model.predict(train_image_pool[i])
        pred_with_label = [(label, prediction[j]) for j, label in enumerate(train_label_pool[i])]
        train_vectors += pred_with_label
        train_labels += ['in']*len(prediction)
        # predict out vectors for ith shadow model
        prediction = model.predict(test_image_pool[i])
        pred_with_label = [(label, prediction[j]) for j, label in enumerate(test_label_pool[i])]
        train_vectors += pred_with_label
        train_labels += ['out'] * len(prediction)
    print(train_labels)



if __name__ == '__main__':
    train_image, train_label, test_image, test_label = load_mnist()
    train_image = train_image / 255.0
    test_image = test_image / 255.0
    print(train_image.shape[1:])
    input_shape = train_image.shape[1:]
    target_image_train, target_label_train, shadow_image_train, shadow_label_train = split_train_data(train_image, train_label, 10000, 50)
    shadow_image_test, shadow_label_test = split_test_data(test_image, test_label, 3000, 50)
    baseline_target(input_shape, 8, target_image_train, target_label_train, test_image, test_label)
    #shadow_models = baseline_shadow(input_shape, 8, shadow_image_train, shadow_label_train)
    #print('Training the attack model:')
    #train_attack(shadow_models, shadow_image_train, shadow_label_train, shadow_image_test, shadow_label_test)
    # info = tfds.builder('mnist').info
    # print(test)
    # fig = tfds.show_examples(train, info)
    # plt.show()
