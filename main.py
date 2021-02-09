import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


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

    # separate data for target model
    np.random.seed(seed)
    np.random.shuffle(test_image)
    target_test_image = test_image[:size]
    shadow_image_pool = test_image[:size]
    np.random.seed(seed)
    np.random.shuffle(test_label)
    target_test_label = test_label[:size]
    shadow_label_pool = test_label[:size]

    # sample test
    for _ in range(0, n):
        np.random.seed(seed)
        np.random.shuffle(shadow_image_pool)
        shadow_test_image.append(shadow_image_pool[:size])
        np.random.seed(seed)
        np.random.shuffle(shadow_label_pool)
        shadow_test_label.append((shadow_label_pool[:size]))
    return target_test_image, target_test_label, shadow_test_image, shadow_test_label


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
    train_vectors_by_class = defaultdict(lambda: [])
    train_labels_by_class = defaultdict(lambda: [])
    for i, model in enumerate(shadow_models):
        # predict in vectors for ith shadow model
        prediction = model.predict(train_image_pool[i])
        # pred_with_label = [(label, prediction[j], 'in') for j, label in enumerate(train_label_pool[i])]
        for j, label in enumerate(train_label_pool[i]):
            train_vectors_by_class[label].append(prediction[j])
            train_labels_by_class[label].append(1)
        # predict out vectors for ith shadow model
        prediction = model.predict(test_image_pool[i])
        # pred_with_label = [(label, prediction[j], 'out') for j, label in enumerate(test_label_pool[i])]
        for j, label in enumerate(test_label_pool[i]):
            train_vectors_by_class[label].append(prediction[j])
            train_labels_by_class[label].append(0)

    # train models per class
    models_per_class = []
    for i in range(0, 10):
        model = models.Sequential()
        model.add(layers.Dense(30, input_dim=10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(np.array(train_vectors_by_class[i]), np.array(train_labels_by_class[i]), epochs=10)
        models_per_class.append(model)
    return models_per_class


def test_attack_model(target_model, models_per_class, image_train, label_train, image_test, label_test):
    vectors_by_class = defaultdict(lambda: [])
    labels_by_class = defaultdict(lambda: [])

    # make predictions on train and test data
    train_prediction = target_model.predict(image_train)
    test_prediction = target_model.predict(image_test)

    # sort data by classes
    for i, prediction in enumerate(train_prediction):
        vectors_by_class[label_train[i]].append(prediction)
        labels_by_class[label_train[i]].append(1)
    for i, prediction in enumerate(test_prediction):
        vectors_by_class[label_test[i]].append(prediction)
        labels_by_class[label_test[i]].append(0)

    # evaluate models
    for i, model in enumerate(models_per_class):
        test_loss, test_acc = model.evaluate(np.array(vectors_by_class[i]), np.array(labels_by_class[i]), verbose=2)
        print(f'\nTest accuracy for label {i}: {test_acc}')


if __name__ == '__main__':
    train_image, train_label, test_image, test_label = load_mnist()
    train_image = train_image / 255.0
    test_image = test_image / 255.0
    print(train_image.shape[1:])
    input_shape = train_image.shape[1:]
    target_image_train, target_label_train, shadow_image_train, shadow_label_train = split_train_data(train_image, train_label, 10000, 50)
    target_image_test, target_label_test, shadow_image_test, shadow_label_test = split_test_data(test_image, test_label, 10000, 50)
    target_model = baseline_target(input_shape, 8, target_image_train, target_label_train, test_image, test_label)
    shadow_models = baseline_shadow(input_shape, 8, shadow_image_train, shadow_label_train)
    print('Training the attack model:')
    attack_model = train_attack(shadow_models, shadow_image_train, shadow_label_train, shadow_image_test, shadow_label_test)
    print('Evaluating the attack model:\n')
    test_attack_model(target_model, attack_model, target_image_train, target_label_train, target_image_test, target_label_test)
    # info = tfds.builder('mnist').info
    # print(test)
    # fig = tfds.show_examples(train, info)
    # plt.show()
