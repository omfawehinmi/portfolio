import sys
import tarfile
import cv2
import matplotlib.pyplot as plt
import random
import os
import zipfile
import urllib.request
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import DenseNet121
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dropout
import pickle
from PIL import UnidentifiedImageError
import datetime
from io import StringIO
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
print("Current Run-time:", run_time)
'''_________________________________________________________________________________________________________________________________________________________________________'''
# Check if GPU is available
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU Available:", gpu_available)

# Check the name of the GPU (if available)
if gpu_available:
    print("GPU Name:", tf.config.list_physical_devices('GPU')[0].name)
sys.exit()
'''_________________________________________________________________________________________________________________________________________________________________________'''
Set memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

'''_________________________________________________________________________________________________________________________________________________________________________'''
## Set memory growth for the CPU
tf.config.set_logical_device_configuration(
    tf.config.list_logical_devices('CPU')[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=5000)],
)

'''_________________________________________________________________________________________________________________________________________________________________________'''
# Use two GPUs
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print("Logical GPUs:", logical_gpus)

# Create a MirroredStrategy using the available GPUs
strategy = tf.distribute.MirroredStrategy()

'''_________________________________________________________________________________________________________________________________________________________________________'''
def display_images_from_folder(folder):
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    num_images = len(image_files)

    # You can adjust the number of rows and columns in the plot as per your preference
    num_rows = 3
    num_cols = 5
    total_images = num_rows * num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 6))
    for i in range(total_images):
        if i < num_images:
            img_path = os.path.join(folder, image_files[i])
            img = plt.imread(img_path)
            ax = axes[i // num_cols, i % num_cols]
            ax.imshow(img, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(image_files[i])
        else:
            axes[i // num_cols, i % num_cols].axis('off')

    plt.tight_layout()
    plt.show()


# Display images from the specified folders
train_fol = os.path.join('/mnt/d/machine_learning/data/inpainting/CIFAKE', 'train')
test_fol = os.path.join('/mnt/d/machine_learning/data/inpainting/CIFAKE', 'test')

# Display images from the specified folders
folders_to_display = [os.path.join(test_fol, "FAKE"), os.path.join(test_fol, "REAL"),
                      os.path.join(train_fol, "FAKE"), os.path.join(train_fol, "REAL")]

for folder in folders_to_display:
    print(f"Displaying images from '{folder}' folder:")
    display_images_from_folder(folder)

'''_________________________________________________________________________________________________________________________________________________________________________'''
# Data Preparation - Organize your dataset into "Train" and "Test" folders
# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = train_fol
test_dir = test_fol

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['FAKE', 'REAL'],
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['FAKE', 'REAL'],
    subset='validation',
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['FAKE', 'REAL'],
)

'''_________________________________________________________________________________________________________________________________________________________________________'''
# Transfer Learning, using pre-trained CNN
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True


# def train_and_evaluate(params):
    # Extract hyperparameters
    # learning_rate = params['learning_rate']
    # dropout_rate = params['dropout_rate']

# Use Dropout for regularization
dropout_rate = 0.5
model = Sequential([
    base_model,
    Conv2D(16, kernel_size=(3, 3),activation="relu", input_shape=(224, 224, 3), padding="same"),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid'),
])

# Model Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
              loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Save the model summary to a file using StringIO
string_buffer = StringIO()
model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
model_summary_text = string_buffer.getvalue()

with open('/mnt/d/machine_learning/models/model_summary - ' + run_time + '.txt', 'w') as f:
    f.write(model_summary_text)

'''_________________________________________________________________________________________________________________________________________________________________________'''
# Define the log directory where TensorBoard will store the logs
log_dir = "/mnt/d/machine_learning/tensorboard_logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

'''_________________________________________________________________________________________________________________________________________________________________________'''# Early stopping
early = EarlyStopping(monitor="loss", mode="min", patience=3)

# Learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.1, min_lr=0.000001)

callbacks_list = [early, tensorboard_callback, learning_rate_reduction]

# Define a custom generator to handle image loading errors
def custom_generator(generator):
    while True:
        try:
            data, labels = next(generator)
            yield data, labels
        except UnidentifiedImageError as e:
            print(f"Error loading image: {e}")


history = model.fit(
                    custom_generator(train_generator),
                    epochs=10,
                    steps_per_epoch=len(train_generator),
                    validation_data=custom_generator(validation_generator),
                    validation_steps=len(validation_generator),
                    callbacks=callbacks_list)

# Save training history
with open("/mnt/d/machine_learning/training_history/training_history - " + run_time + ".pkl", "wb") as f:
    pickle.dump(history.history, f)

# Plot loss and accuracy
def plot_training_results(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the plot as an image file
    plt.savefig('/mnt/d/machine_learning/training_history/training_results - " + run_time + ".png')

    plot_training_results(history)

space = {
    'learning_rate': hp.loguniform('learning_rate', -2, -1)
    'dropout_rate': hp.uniform('dropout_rate', 0.5, 0.6),
}
# Initialize the Trials
trials = Trials()
# Run Hyperopt optimization:
best = fmin(fn=train_and_evaluate, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
'''_________________________________________________________________________________________________________________________________________________________________________'''# Early stopping
# Model Evaluation
test_loss, test_accuracy = model.evaluate(custom_generator(test_generator), steps=len(test_generator))
print("Test Accuracy:", test_accuracy)

# Save the evaluation results to a file using pickle
evaluation_results = {'test_loss': test_loss, 'test_accuracy': test_accuracy}

with open('/mnt/d/machine_learning/evaluation_results/evaluation_results - ' + run_time + '.pkl', 'wb') as f:
    pickle.dump(evaluation_results, f)

'''_________________________________________________________________________________________________________________________________________________________________________'''# Early stopping
model.save('/mnt/d/machine_learning/models/CIFAKE - ' +run_time + '.keras')

completion_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
print("Completion-time:", completion_time)