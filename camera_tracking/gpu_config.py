import tensorflow as tf
import sys

# Check if GPU is available
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU Available:", gpu_available)

# Check the name of the GPU (if available)
if gpu_available:
    print("GPU Name:", tf.config.list_physical_devices('GPU')[0].name)
sys.exit()
'''_________________________________________________________________________________________________________________________________________________________________________'''
# Set memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

'''_________________________________________________________________________________________________________________________________________________________________________'''
# Set memory growth for the CPU
tf.config.set_logical_device_configuration(
    tf.config.list_logical_devices('CPU')[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=5000)],
)

'''_________________________________________________________________________________________________________________________________________________________________________'''

# Check if GPU is available
def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'], capture_output=True,
                                text=True)
        gpu_info = result.stdout.strip().split('\n')
        gpu_info = [info.split(', ') for info in gpu_info]
        return gpu_info
    except FileNotFoundError:
        return None


# Set the order of GPUs you want to use. GPU 0 will be used first, then GPU 1.
gpu_order = [0, 1]

# Create a MirroredStrategy using the available GPUs
strategy = tf.distribute.MirroredStrategy()

# Define a function to limit GPU memory growth

def limit_gpu_memory(gpu_index, memory_fraction=0.95):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if len(gpus) > gpu_index:
            try:
                # Limit GPU memory growth for the specified GPU index
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_index],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(memory_fraction * 1024))]
                )
            except RuntimeError as e:
                print(e)
