# Inpainting Image Detection

## Introduction

In a world inundated with digital media and misinformation, ensuring image authenticity is paramount. This paper introduces an innovative inpainting image detection tool, using machine learning to verify the presence of inpainting in images. Inpainting, the process of filling in missing parts of an image, is often used for image manipulation. The tool analyzes images, detecting irregularities that may indicate inpainting and providing a confidence score. Its purpose is to combat the spread of fake media and misinformation, safeguarding users from deceptive visuals. Journalists, fact-checkers, art experts, e-commerce platforms, and social media networks can benefit from the tool to validate image authenticity and uphold credibility. By providing a reliable defense against manipulated visuals, the tool ensures trust in the integrity of digital media.

### CIFAKE: Real and AI-Generated Synthetic Images

The CIFAKE dataset consists of 60,000 AI-generated images and an equivalent number of 60,000 real images, sourced from CIFAR-10. While the dataset offers a wealth of training data, it's important to note this AI-generated image dataset varies in quality due to the inherent nature of synthetic content. This variability could influence the model's performance, particularly when confronted with lower-quality AI-generated images in the test set.

![CIFAKE Dataset Image](assets/img/CIFAKE_pic.png)

### DeepfakeArt Challenge Image Dataset

The DeepfakeArt Challenge Dataset is comprised of a collection of over 32,000 images, the dataset encompasses a diverse array of generative forgery and data poisoning techniques. Each entry in the dataset consists of a pair of images, categorized as forgeries/adversarially contaminated or authentic. Notably, every generated image has undergone rigorous quality checks to ensure accuracy and consistency.The DeepfakeArt Challenge Dataset encompasses a wide range of generative forgery and data poisoning methods, including inpainting, style transfer, adversarial data poisoning, and Cutmix. I specifically chose to utilize the Inpainting dataset from the DeepfakeArt Challenge repository.

![DeepfakeArt Dataset Image](assets/img/DeepfakeArt_pic.jpg)

These datasets were thoughtfully organized into "Train" and "Test" folders, each containing distinct subfolders that play a crucial role in facilitating effective model training and evaluation. In the "Train" section of each data repository, images are categorized into subfolders named "Real" and "Fake." This organization allows my machine learning models to learn from a diverse range of image types during training phases. By being exposed to both authentic and AI-generated images, the models can develop a robust understanding of the characteristics that distinguish the two categories. Correspondingly, the "Test" section of the dataset mirrors this categorization, with subfolders named "Real" and "Fake".

## System Configuration

In this project, I'll leverage my personal PC build featuring 64 GB of RAM and an AMD Ryzen 5 5600 6-Core Processor. The system boasts a dual GPU setup, including the NVIDIA GeForce RTX 4080 and RTX 3060. The RTX 4080 contains 9728 CUDA cores for parallel processing, 16 GB VRAM, and specialized over 300 Tensor and Ray Tracing cores that optimize deep learning and graphics tasks. Similarly, the RTX 3060 showcases 3584 CUDA cores, 12 GB VRAM, and functional over 100 Tensor and Ray Tracing cores.

To maximize training efficiency, I adopted a distributed strategy using TensorFlow's 'tf.distribute' module. I opted for the MirroredStrategy, facilitating synchronous training across GPUs by replicating the model on each unit and harmonizing updates during training. It processes different data subsets in parallel, synchronizes gradients, averages losses, and updates model parameters consistently. This strategy significantly accelerates the training process. Moreover, I fine-tuned GPU memory allocation with the 'limit_gpu_memory' function. This customization lets me specify the GPU memory fraction, preventing memory overflow and bolstering training stability. This comprehensive approach ensures optimal performance in model training and evaluation.

```python
def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_info = result.stdout.strip().split('\n')
        gpu_info = [info.split(', ') for info in gpu_info]
        return gpu_info
    except FileNotFoundError:
        return None

gpu_order = [0, 1]
strategy = tf.distribute.MirroredStrategy()

def limit_gpu_memory(gpu_index, memory_fraction=0.95):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if len(gpus) > gpu_index:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_index],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(memory_fraction * 1024))]
                )
            except RuntimeError as e:
                print(e) ```

## Approach

This project focuses on detecting image inpainting using convolutional neural networks (CNNs) and an ensemble model. The project takes advantage of a dual dataset approach, combining the CIFAKE: Real and AI-Generated Synthetic Images dataset alongside the esteemed DeepFake Detection Challenge Dataset. Training the CNNs entails an intricate amalgamation of techniques, including the strategic application of transfer learning, which capitalizes on pre-trained models' intrinsic feature extraction capabilities. The arsenal of strategies further expands with the integration of data augmentation to enhance generalization, early stopping for combating overfitting, and regularization methods to ensure optimum model robustness. The models' convergence is expedited through the application of the binary cross-entropy loss function, while the judicious freezing of layers refines efficiency. The project's apex lies in the formulation of an ensemble model, combining the diverse predictions from individual CNNs. Hyperparameter optimization through the sophisticated hyperopt library underscores the meticulous nature of parameter tuning.

## Preprocessing Image Data: ImageDataGenerator

Images & Feature Extraction
In this project, feature extraction through convolutional layers will play a pivotal role in detecting image inpainting. By applying convolutional layers, my models will automatically identify intricate patterns, textures, and irregularities within images. These features will be learned and consolidated into higher-level representations as the models delve deeper into their architecture. This process enables the models to differentiate between authentic and manipulated regions, leveraging the inherent capabilities of CNNs to perform complex image analysis. By utilizing feature extraction, my models will effectively distinguish between inpainted and non-inpainted sections, contributing to accurate image forensics and manipulation detection.

## Feature Extraction Image

The provided code segment demonstrates crucial steps in processing image data. Initially, the function display_images_from_folder is defined to visualize a selection of images from specified folders. Subsequently, the script sets the directories for the training and testing data. Images from these directories are displayed using the defined function. Moving on, the data preparation phase begins. Through the ImageDataGenerator, data augmentation techniques are applied, including rotation, flipping, shifting, zooming, and shearing, all while rescaling pixel values. This prepared data is organized into train and test subsets using the flow_from_directory method, ensuring a consistent image size of 224x224 pixels. Each subset is divided into batches of 32 images with binary classification labels (FAKE or REAL).

```python
Copy code
train_dir = 'train_folder_path'
test_dir = 'test_folder_path'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode="nearest",
    validation_split=0.2  # 20% of the data will be used for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # assuming binary classification
    classes=['FAKE', 'REAL'],  # specify class names
    subset='training'  # indicate training subset
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['FAKE', 'REAL'],
    subset='validation'  # indicate validation subset
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    classes=['FAKE', 'REAL']
)

## Custom Image Generator

The generator operates within an infinite loop using a While True construct. Inside the loop, the next(generator) function is called to retrieve the next batch of data and corresponding labels from the original data generator. If the data loading process encounters an UnidentifiedImageError, which could occur due to corrupted or invalid image files, the generator catches the error and displays an error message indicating the issue. The purpose of this custom generator is to ensure robustness and resilience in handling potential errors when loading images for training or evaluation. By using this generator, the research paper's machine learning pipeline becomes more stable and less prone to interruptions caused by problematic images, ultimately contributing to the reliability and effectiveness of the model training process.

```python
Copy code
def custom_generator(generator):
    while True:
        try:
            data, labels = next(generator)
            yield data, labels
        except UnidentifiedImageError as e:
            print(f"Error loading image: {e}")

### Convolutional Neural Network: Model Build
CIFAKE
As exhibited below the model is formulated within the strategy's scope which allows for the utilization of our mirror strategy. Using transfer learning through a pre-trained DenseNet121 base. The final ten layers of the base are fine-tuned for task-specific adaptation. The architecture's sequential part involves successive Convolutional layers with 16 and 32 filters respectively, employing kernel sizes of 3x3, ReLU activation, and same-padding to extract salient features from input images. Following this, a Flatten layer transforms the feature maps into a one-dimensional vector, facilitating integration into densely connected layers. Within the dense portion, two additional layers with 128 and 64 neurons respectively utilize ReLU activation and L2 regularization (with a regularization strength of 0.01) to enhance feature interpretation and control overfitting. To further curb overfitting, a dropout layer with a rate of 0.5 is introduced. The ultimate layer, equipped with a sigmoid activation function, culminates the architecture, facilitating binary classification output.

python
Copy code
def train_model(model, train_generator, validation_generator, callbacks_list, epochs=10):
    with strategy.scope():
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers[56:]:
            layer.trainable = True

        dropout_rate = 0.5

        model = Sequential([
            base_model,
            Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
Deepfake
The construction commences by importing a pre-trained Xception model, pretrained on ImageNet. The subsequent step involves fine-tuning, where a subset of layers (from index 56 onwards) is rendered trainable, thereby adapting the model to the specific task. To prevent overfitting, a Dropout layer with a rate of 0.5 is introduced, facilitating regularization. Further augmentation occurs through the sequential layer composition. The Xception base model is followed by a Global Average Pooling 2D layer, which distills complex feature maps into a more manageable form. The Dropout layer follows, curtailing overfitting risks. Finally, a single Dense layer with sigmoid activation furnishes the model's classification output. This configuration interlaces the power of transfer learning, enabling the network to leverage learned features, with careful regularization and pooling mechanisms. The tailored architecture strikes a balance between complexity and efficiency, ensuring effective image classification while minimizing overfitting.

python
Copy code
def train_model(model, train_generator, validation_generator, callbacks_list, epochs=15):
    with strategy.scope():
        base_model = Xception(weights="imagenet", include_top=False)

        for layer in base_model.layers[56:]:
            layer.trainable = True

        dropout_rate = 0.5

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')
        ])
Ensemble
Firstly, two pretrained models, loaded from specified paths, are assigned names for differentiation. The models are labeled as 'model1' and 'model2'. Following this, a new input layer is introduced, designed to accommodate the specific input dimensions (224x224x3). This input layer forms the entry point for the ensemble model. Subsequently, the outputs of both 'model1' and 'model2' are retrieved by passing the ensemble input through these models. These outputs are then concatenated together, amalgamating the predictions generated by both individual models. To finalize the ensemble, a Dense layer with sigmoid activation is appended, shaping the output for binary classification (the classification task at hand). Ultimately, the ensemble model is established using the Functional API of TensorFlow. The ensemble's inputs consist of the ensemble input layer, while the outputs are steered through the concatenated and transformed layers, culminating in the ensemble output layer. This ensemble model demonstrates the potency of combining the predictive prowess of two independently trained models to achieve superior classification outcomes. Through this configuration, the ensemble leverages the diversity of learned features from different models, thus leading to a more robust and effective classifier.

python
Copy code
def train_model(ensemble_model, train_generator, validation_generator, callbacks_list, epochs=10):
    with strategy.scope():
        model_path1 = '/home/drill-team/Machine Learning/models/CIFAKE 2023-07-27 15:00.keras'
        model_path2 = '/home/drill-team/M
               
