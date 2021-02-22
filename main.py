import tensorflow as tf
from tensorflow.keras import layers, models, datasets, callbacks
import tensorflow_datasets as tfds
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from absl import flags
from absl import app
import os


flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.25, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.3,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.5, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

DPFLAGS = flags.FLAGS


def load_mnist():
    # load data
    (train_image, train_label), (test_image, test_label) = datasets.fashion_mnist.load_data()

    train_image = np.array(train_image, dtype=np.float32) / 255
    test_image = np.array(test_image, dtype=np.float32) / 255

    train_image = train_image.reshape((train_image.shape[0], 28, 28, 1))
    test_image = test_image.reshape((test_image.shape[0], 28, 28, 1))

    train_label = np.array(train_label, dtype=np.int32)
    test_label = np.array(test_label, dtype=np.int32)

    train_label = tf.keras.utils.to_categorical(train_label, num_classes=10)
    test_label = tf.keras.utils.to_categorical(test_label, num_classes=10)

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
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def compute_epsilon(steps, data_size):
  """Computes epsilon value for given hyperparameters."""
  if DPFLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = DPFLAGS.batch_size / data_size
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=DPFLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def compile_dp(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(10))

    optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
        l2_norm_clip=DPFLAGS.l2_norm_clip,
        noise_multiplier=DPFLAGS.noise_multiplier,
        num_microbatches=DPFLAGS.microbatches,
        learning_rate=DPFLAGS.learning_rate)


    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE),
                  metrics=['accuracy'])
    return model


def evaluate_dp(model, train_image, train_label, test_image, test_label):
    checkpoint_path = "models/training_dp0/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)

    model.fit(train_image,
              train_label,
              epochs=DPFLAGS.epochs,
              batch_size=DPFLAGS.batch_size,
              callbacks=[cp_callback])

    model.save('models/dp1')
    test_loss, test_acc = model.evaluate(test_image, test_label, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    eps = compute_epsilon(DPFLAGS.epochs * len(train_image) // DPFLAGS.batch_size, len(train_image))
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)


def baseline_target(input_shape, epochs, train_image, train_label, test_image, test_label):
    model = compile_baseline(input_shape)
    history = model.fit(train_image, train_label, epochs=epochs, validation_split=0.2)
    visualize_training(history, epochs)
    test_loss, test_acc = model.evaluate(test_image, test_label, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    probability_model = tf.keras.Sequential([model, layers.Softmax()])
    probability_model.save('models/baseline')

    return probability_model


def baseline_shadow(input_shape, epochs, image_pool, label_pool):
    assert len(image_pool) == len(label_pool)
    n = len(image_pool)
    shadow_models = []
    for i in range(0, n):
        print(f'Training model {i+1} out of {n}')
        model = compile_baseline(input_shape)
        model.fit(image_pool[i], label_pool[i], epochs=epochs)
        probability_model = tf.keras.Sequential([model, layers.Softmax()])
        shadow_models.append(probability_model)
        probability_model.save(f'models/shadow_models/{i}')

    return shadow_models


def train_attack(shadow_models, train_image_pool, train_label_pool, test_image_pool, test_label_pool):
    train_vectors_by_class = defaultdict(lambda: [])
    train_labels_by_class = defaultdict(lambda: [])

    for i, model in enumerate(shadow_models):
        # predict in vectors for ith shadow model
        prediction = model.predict(train_image_pool[i])
        # pred_with_label = [(label, prediction[j], 'in') for j, label in enumerate(train_label_pool[i])]
        for j, label in enumerate(train_label_pool[i]):
            lab = list(label).index(1.)
            train_vectors_by_class[lab].append(prediction[j])
            train_labels_by_class[lab].append(1)
        # predict out vectors for ith shadow model
        prediction = model.predict(test_image_pool[i])
        # pred_with_label = [(label, prediction[j], 'out') for j, label in enumerate(test_label_pool[i])]
        for j, label in enumerate(test_label_pool[i]):
            lab = list(label).index(1.)
            train_vectors_by_class[lab].append(prediction[j])
            train_labels_by_class[lab].append(0)

    # train models per class
    models_per_class = []
    for i in range(0, 5):
        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(np.array(train_vectors_by_class[i]), np.array(train_labels_by_class[i]), epochs=30)
        models_per_class.append(model)
        model.save(f'models/attack/class{i}')
    return models_per_class


def test_attack_model(target_model, models_per_class, image_train, label_train, image_test, label_test):
    vectors_by_class = defaultdict(lambda: [])
    labels_by_class = defaultdict(lambda: [])

    # make predictions on train and test data
    train_prediction = target_model.predict(image_train)
    test_prediction = target_model.predict(image_test)

    # sort data by classes
    for i, prediction in enumerate(train_prediction):
        label = list(label_train[i]).index(1.)
        vectors_by_class[label].append(prediction)
        labels_by_class[label].append(1)
    for i, prediction in enumerate(test_prediction):
        label = list(label_test[i]).index(1.)
        vectors_by_class[label].append(prediction)
        labels_by_class[label].append(0)

    # evaluate models
    for i, model in enumerate(models_per_class):
        test_loss, test_acc = model.evaluate(np.array(vectors_by_class[i]), np.array(labels_by_class[i]), verbose=2)
        print(f'\nTest accuracy for label {i}: {test_acc}')


def main(unused_argv):
    # Load and prepare data
    train_image, train_label, test_image, test_label = load_mnist()
    # print(train_image.shape[1:])
    input_shape = train_image.shape[1:]
    target_image_train, target_label_train, shadow_image_train, shadow_label_train = split_train_data(train_image, train_label, 10000, 50)
    target_image_test, target_label_test, shadow_image_test, shadow_label_test = split_test_data(test_image, test_label, 10000, 50)

    target_model = baseline_target(input_shape, 50, target_image_train, target_label_train, test_image, test_label)
    shadow_models = baseline_shadow(input_shape, 50, shadow_image_train, shadow_label_train)
    print('Training the attack model:')
    attack_model = train_attack(shadow_models, shadow_image_train, shadow_label_train, shadow_image_test, shadow_label_test)
    print('Evaluating the attack model:\n')
    test_attack_model(target_model, attack_model, target_image_train, target_label_train, target_image_test, target_label_test)

    # eps = compute_epsilon(DPFLAGS.epochs * len(target_image_train) // DPFLAGS.batch_size, len(target_image_train))
    # print(f'For delta=1e-5, the current epsilon is: {eps}')
    # dp_model = compile_dp(input_shape)
    # evaluate_dp(dp_model, target_image_train, target_label_train, target_image_test, target_label_test)

    # info = tfds.builder('mnist').info
    # print(test)
    # fig = tfds.show_examples(train, info)
    # plt.show()


if __name__ == '__main__':
    app.run(main)