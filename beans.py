import tensorflow as tf
import tensorflow_datasets as tfds
print(f"Tensorflow version : {tf.__version__}")

# Create dataset from TFDS
(train_ds, val_ds), info = tfds.load(
    name="beans", split=['train', 'validation'], with_info=True, as_supervised=True)

# Examine dataset
print(f"No. of training examples : {info.splits['train'].num_examples}")
print(f"No. of validation examples : {info.splits['validation'].num_examples}")
num_classes = info.features['label'].num_classes
print(f"No. of classes : {num_classes}")

# Show training examples
tfds.show_examples(info,train_ds)

# Data preprocessing
IMG_SIZE = 150     # downsize from 500 to 150 to speed up
def format_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))/255.
    return image, label

# Transform data into batches
BATCH_SIZE = 32
training_batches = (
    train_ds.shuffle(30)
    .map(format_image)
    .batch(BATCH_SIZE)
    .prefetch(1)
)
validation_batches = (
    val_ds.map(format_image).batch(BATCH_SIZE).prefetch(1)
)

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
print(model.summary())
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)
history = model.fit(training_batches, epochs=30, validation_data=validation_batches)

# Plot the training curves
import matplotlib.pyplot as plt

def plot_curve(curve_type):
    plt.figure(figsize=(10,6))
    epochs = range(1, len(history.history[f"{curve_type}"]) + 1)
    plt.plot(epochs, history.history[f"{curve_type}"], label=f"Training {curve_type}")
    plt.plot(epochs, history.history[f"val_{curve_type}"], label=f"Validation {curve_type}")
    plt.title(f"Training {curve_type} vs Validation {curve_type}")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

plot_curve('accuracy')
plot_curve('loss')

# Performance evaluation
test_ds = tfds.load(name="beans", split='test', as_supervised=True)
test_batches = (test_ds.map(format_image).batch(BATCH_SIZE).prefetch(1))
loss, acc = model.evaluate(test_batches)
