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
               
