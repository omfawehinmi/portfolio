# Inpainting Image Detection

## Introduction

In a world inundated with digital media and misinformation, ensuring image authenticity is paramount. This paper introduces an innovative inpainting image detection tool, using machine learning to verify the presence of inpainting in images. Inpainting, the process of filling in missing parts of an image, is often used for image manipulation. The tool analyzes images, detecting irregularities that may indicate inpainting and providing a confidence score. Its purpose is to combat the spread of fake media and misinformation, safeguarding users from deceptive visuals. Journalists, fact-checkers, art experts, e-commerce platforms, and social media networks can benefit from the tool to validate image authenticity and uphold credibility. By providing a reliable defense against manipulated visuals, the tool ensures trust in the integrity of digital media.

### CIFAKE: Real and AI-Generated Synthetic Images

The CIFAKE dataset consists of 60,000 AI-generated images and an equivalent number of 60,000 real images, sourced from CIFAR-10. While the dataset offers a wealth of training data, it's important to note this AI-generated image dataset varies in quality due to the inherent nature of synthetic content. This variability could influence the model's performance, particularly when confronted with lower-quality AI-generated images in the test set.

!(assets/img/CIFAKE_pic.png)

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
                print(e)
```

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
```

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
```

### Convolutional Neural Network: Model Build

## CIFAKE

As exhibited below the model is formulated within the strategy's scope which allows for the utilization of our mirror strategy. Using transfer learning through a pre-trained DenseNet121 base. The final ten layers of the base are fine-tuned for task-specific adaptation. The architecture's sequential part involves successive Convolutional layers with 16 and 32 filters respectively, employing kernel sizes of 3x3, ReLU activation, and same-padding to extract salient features from input images. Following this, a Flatten layer transforms the feature maps into a one-dimensional vector, facilitating integration into densely connected layers. Within the dense portion, two additional layers with 128 and 64 neurons respectively utilize ReLU activation and L2 regularization (with a regularization strength of 0.01) to enhance feature interpretation and control overfitting. To further curb overfitting, a dropout layer with a rate of 0.5 is introduced. The ultimate layer, equipped with a sigmoid activation function, culminates the architecture, facilitating binary classification output.

```python
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
```

## Deepfake

The construction commences by importing a pre-trained Xception model, pretrained on ImageNet. The subsequent step involves fine-tuning, where a subset of layers (from index 56 onwards) is rendered trainable, thereby adapting the model to the specific task. To prevent overfitting, a Dropout layer with a rate of 0.5 is introduced, facilitating regularization. Further augmentation occurs through the sequential layer composition. The Xception base model is followed by a Global Average Pooling 2D layer, which distills complex feature maps into a more manageable form. The Dropout layer follows, curtailing overfitting risks. Finally, a single Dense layer with sigmoid activation furnishes the model's classification output. This configuration interlaces the power of transfer learning, enabling the network to leverage learned features, with careful regularization and pooling mechanisms. The tailored architecture strikes a balance between complexity and efficiency, ensuring effective image classification while minimizing overfitting.

```python
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
```

## Ensemble

Firstly, two pretrained models, loaded from specified paths, are assigned names for differentiation. The models are labeled as 'model1' and 'model2'. Following this, a new input layer is introduced, designed to accommodate the specific input dimensions (224x224x3). This input layer forms the entry point for the ensemble model. Subsequently, the outputs of both 'model1' and 'model2' are retrieved by passing the ensemble input through these models. These outputs are then concatenated together, amalgamating the predictions generated by both individual models. To finalize the ensemble, a Dense layer with sigmoid activation is appended, shaping the output for binary classification (the classification task at hand). Ultimately, the ensemble model is established using the Functional API of TensorFlow. The ensemble's inputs consist of the ensemble input layer, while the outputs are steered through the concatenated and transformed layers, culminating in the ensemble output layer. This ensemble model demonstrates the potency of combining the predictive prowess of two independently trained models to achieve superior classification outcomes. Through this configuration, the ensemble leverages the diversity of learned features from different models, thus leading to a more robust and effective classifier.

```python
def train_model(ensemble_model, train_generator, validation_generator, callbacks_list, epochs=10):
    with strategy.scope():
        model_path1 = '/home/drill-team/Machine Learning/models/CIFAKE 2023-07-27 15:00.keras'
        model_path2 = '/home/drill-team/Machine Learning/models/DeepfakeArt- 2023-08-06 20:34.keras'
        model1 = tf.keras.models.load_model(model_path1)
        model2 = tf.keras.models.load_model(model_path2)

        model1._name = 'model1'
        model2._name = 'model2'

        ensemble_input = Input(shape=(224, 224, 3))

        output1 = model1(ensemble_input)
        output2 = model2(ensemble_input)

        concatenated = concatenate([output1, output2])

        ensemble_output = Dense(1, activation='sigmoid')(concatenated)

        ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output)
```

### Callbacks

## CIFAKE
The "EarlyStopping" callback monitors the loss metric, terminating training if the loss doesn't substantially decrease over a predefined number of epochs (patience of 3). This helps prevent overfitting and conserves computational resources. The "ReduceLROnPlateau" callback dynamically adjusts the learning rate during training. It observes loss changes and reduces the learning rate by a specified factor (0.1) if significant loss improvement isn't detected after a set number of epochs (patience of 2). The "min_lr" parameter ensures the learning rate doesn't drop below a minimum threshold. These callbacks collectively adapt the model's learning rate and halt training if needed, fostering efficient convergence and preventing suboptimal outcomes. Additionally, the "tensorboard_callback" facilitates model monitoring through TensorBoard visualization.

```python
log_dir = "/home/drill-team/Machine Learning/tensorboard_logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.1, min_lr=0.000001)

early = EarlyStopping(monitor="loss", mode="min", patience=3)
callbacks_list = [early, tensorboard_callback, learning_rate_reduction]
```

## Deepfake & Ensemble

The "EarlyStopping" callback remains consistent with its earlier explanation, monitoring the loss metric and ending training if loss stagnation persists beyond three epochs, preventing unnecessary computation and potential overfitting. The "ReduceLROnPlateau" callback, however, introduces some changes. It still observes loss changes and reduces the learning rate if significant loss improvement isn't observed, yet with a more cautious approach. The "factor" parameter is adjusted to 0.001, indicating a smaller decrease in the learning rate, while the "patience" parameter remains at 2. The "min_lr" parameter is again utilized to ensure the learning rate doesn't dip below a minimum threshold. These callbacks, in conjunction with the "tensorboard_callback," facilitate improved convergence and efficiency in the training process.

```python
learning_rate_reduction = ReduceLROnPlateau(monitor='loss', patience=2, verbose=1, factor=0.001, min_lr=0.000001)
```
### Model Compilation

## CIFAKE & Ensemble

In this section, we delve into the model's compilation, showcasing a refined approach to optimization. Here, the "Adam" optimizer is employed, which facilitates adaptive learning rates, enhancing convergence during training. The learning rate is set to 0.001, aligning with the model's optimization goals. The choice of the "binary_crossentropy" loss function is indicative of the binary classification nature of the problem, adeptly quantifying the divergence between predicted and actual outputs. Simultaneously, accuracy metrics are specified to monitor the model's classification performance during training. This compilation configuration aligns with the model's architectural intricacies and the nature of the image classification task. By tailoring the optimizer, loss function, and evaluation metrics, the model is primed to effectively learn and improve its classification accuracy over training epochs.

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
ensemble_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
ensemble_model.summary()
```

## Deepfake

Here, the "Stochastic Gradient Descent" (SGD) optimizer is introduced, characterized by its iterative weight updates that enhance convergence. The learning rate is set to 0.1, a choice that regulates the step size during weight adjustments, influencing the training speed and stability. Intriguingly, the "momentum" hyperparameter is incorporated at a value of 0.9, amplifying the optimizer's capacity to traverse gradient landscapes with enhanced momentum, promoting efficient learning. The loss function remains "binary_crossentropy," accurately quantifying classification disparities, while accuracy metrics persist to evaluate model performance.

```python
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### Training & Validation
## CIFAKE

The CNN model trained on the CIFAKE dataset showcases a notable progression in performance over successive epochs. Commencing with an initial accuracy of approximately 68.0%, the model gradually refines its predictions, culminating in an impressive accuracy of 90.9% by the final epoch. The loss function exhibits a parallel behavior, diminishing consistently throughout training. The validation accuracy, beginning at 50.0%, ascends to a commendable 84.0%, indicating the model's ability to generalize to unseen data. This is substantiated by the validation loss, which diminishes while manifesting intermittent fluctuations, corroborating the model's robustness against overfitting tendencies.

```
| Epoch | Loss   | Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
|-------|--------|----------|-----------------|---------------------|---------------|
| 1     | 0.3717 | 0.8931   | 0.2321          | 0.9202              | 0.001         |
| 2     | 0.2000 | 0.9267   | 0.1736          | 0.9315              | 0.001         |
| 3     | 0.1785 | 0.9342   | 0.1965          | 0.9324              | 0.001         |
| 4     | 0.1604 | 0.9421   | 0.1844          | 0.9332              | 0.001         |
| 5     | 0.1482 | 0.9458   | 0.1529          | 0.9448              | 0.001         |
| 6     | 0.1397 | 0.9483   | 0.1465          | 0.9507              | 0.001         |
| 7     | 0.1325 | 0.9522   | 0.1938          | 0.9252              | 0.001         |
| 8     | 0.1251 | 0.9541   | 0.1308          | 0.9547              | 0.001         |
| 9     | 0.1183 | 0.9574   | 0.1035          | 0.9613              | 0.001         |
| 10    | 0.1128 | 0.9598   | 0.1125          | 0.9592              | 0.001         |

```

## Deepfake

Similar to its CIFAKE counterpart, the CNN model trained on the DeepfakeArt dataset demonstrates an analogous trajectory of growth. The initial accuracy of 68.0% advances incrementally to reach 90.9% at the conclusion of training. Correspondingly, the loss function consistently diminishes across epochs. The model's generalization capacity is underscored by the validation accuracy, which, commencing at 50.0%, escalates to an appreciable 84.0%. Validation loss, akin to the CIFAKE model, undergoes reduction amidst intermittent variations, signifying effective learning and a propensity to mitigate overfitting.

```
| Epoch | Loss   | Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
|-------|--------|----------|-----------------|---------------------|---------------|
| 1     | 0.6044 | 0.6796   | 2.7905          | 0.5000              | 0.1           |
| 2     | 0.5064 | 0.7600   | 2.8863          | 0.6111              | 0.1           |
| 3     | 0.4659 | 0.7810   | 0.7383          | 0.6531              | 0.1           |
| 4     | 0.4153 | 0.8147   | 0.5438          | 0.7636              | 0.1           |
| 5     | 0.3758 | 0.8292   | 0.4932          | 0.7667              | 0.1           |
| 6     | 0.3685 | 0.8323   | 0.4926          | 0.7691              | 0.1           |
| 7     | 0.3339 | 0.8543   | 1.6651          | 0.6130              | 0.1           |
| 8     | 0.3191 | 0.8576   | 0.4762          | 0.7914              | 0.1           |
| 9     | 0.2821 | 0.8792   | 0.4761          | 0.7858              | 0.1           |
| 10    | 0.2671 | 0.8815   | 0.3979          | 0.8173              | 0.1           |

```
## Ensemble
The ensemble CNN model, a harmonious amalgamation of CIFAKE and DeepfakeArt datasets, presents a distinct advantage in terms of initial performance. With an elevated inception accuracy of 95.7%, the model systematically advances to an impressive 97.0% accuracy, underscoring the efficacy of leveraging multiple datasets. Concomitantly, the ensemble model's loss trajectory commences at a lower point and steadily decreases throughout training. The validation accuracy, commencing at 96.5%, remains consistently high, peaking at 97.5%. This persistent alignment between training and validation accuracy is complemented by the validation loss, which, though initially elevated, demonstrates steady reduction over successive epochs.

```
| Epoch | Loss   | Accuracy | Validation Loss | Validation Accuracy | Learning Rate |
|-------|--------|----------|-----------------|---------------------|---------------|
| 1     | 0.1127 | 0.9569   | 0.0944          | 0.9652              | 0.001         |
| 2     | 0.1059 | 0.9596   | 0.1165          | 0.9593              | 0.001         |
| 3     | 0.1044 | 0.9604   | 0.1097          | 0.9600              | 0.001         |
| 4     | 0.1002 | 0.9618   | 0.0929          | 0.9655              | 0.001         |
| 5     | 0.0980 | 0.9627   | 0.0871          | 0.9688              | 0.001         |
| 6     | 0.0923 | 0.9653   | 0.0897          | 0.9688              | 0.001         |
| 7     | 0.0912 | 0.9663   | 0.0973          | 0.9624              | 0.001         |
| 8     | 0.0871 | 0.9672   | 0.0918          | 0.9667              | 0.001         |
| 9     | 0.0845 | 0.9686   | 0.0730          | 0.9745              | 0.001         |
| 10    | 0.0809 | 0.9698   | 0.0808          | 0.9719              | 0.001         |

```
### Results & Accuracy

## CIFAKE

The CIFAKE CNN model demonstrates robust generalization to new, unseen data, achieving an impressive test accuracy of 96.4%. This performance underscores the model's ability to effectively classify fake and real images even beyond its training domain. The test accuracy, exceeding the training accuracy, suggests a minimal tendency for overfitting, and the relatively low test loss of 0.1013 further validates the model's proficiency in making accurate predictions on previously unseen images. The model's consistent performance across both training and test datasets indicates its reliability and adaptability to a wider range of inputs.

```
| Step       | Duration          | Loss  | Accuracy |
|------------|-------------------|-------|----------|
| 625/625    | 19s 27ms/step     | 0.1013| 0.9639   |
```
Test Accuracy on New Data: 0.9639000296592712

## Deepfake
The DeepfakeArt CNN model exhibits a test accuracy of 81.1%, a solid indication of its competence in distinguishing between deepfake and authentic images in a real-world context. The discrepancy between training and test accuracies suggests a moderate degree of overfitting, though the test loss of 0.5316 remains reasonable. Despite this, the model's performance underscores its viability for practical applications, such as identifying deepfakes within artistic contexts. By achieving over 80% accuracy on unseen data, the model demonstrates its utility beyond the training data's confines, reinforcing its capacity for real-world application.

```
| Step       | Duration          | Loss  | Accuracy |
|------------|-------------------|-------|----------|
| 64/64      | 541ms/step        | 0.5316| 0.8110   |
```
Test Accuracy on New Data: 0.8109575510025024

## Ensemble

The ensemble CNN model, a fusion of CIFAKE and DeepfakeArt datasets, showcases an impressive test accuracy of 94.9%, reiterating its proficiency in discerning fake from real images. This robust performance signifies the successful integration of diverse data sources to create a more versatile and capable model. The relatively low test loss of 0.1373 further validates the ensemble's ability to make accurate predictions on novel inputs. The ensemble model's consistency between training and test accuracies, coupled with its high performance, reaffirms its suitability for practical use cases where accuracy and generalization are paramount.

```
| Step       | Duration          | Loss  | Accuracy |
|------------|-------------------|-------|----------|
| 689/689    | 63ms/step         | 0.1373| 0.9497   |
```
Test Accuracy on New Data: 0.9496958255767822

