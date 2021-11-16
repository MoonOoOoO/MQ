import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


BATCH_SIZE = 8
N_WARMUP_RUN = 50
N_RUN = 1000 


def generate_batch():
    """
    Generate batch with BATCH_SIZE for local model benchmarking
    """
    batched_input = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
    for n in range(BATCH_SIZE):
        img_path = './testdata/img%d.JPG' % (n % 4)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        batched_input[n, :] = x
    batched_input = tf.constant(batched_input)
    print('batched_input shape: ', batched_input.shape)
    return batched_input


def benchmark_keras_model(input_model):
    """
    Param:
        input_model: the DNN model to be benchmarked
    Benchmark a DNN model locally use prepared batch input
    """
    elapsed_time = []
    batched_input = generate_batch()
    for _ in range(N_WARMUP_RUN):
        labeling = input_model.predict(batched_input)
        # results = decode_predictions(labeling)
        # print(results)

    for i in range(N_RUN):
        start_time = time.time()
        labeling = input_model.predict(batched_input)
        # results = decode_predictions(labeling)
        # print(results)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 50 == 0:
            print('Step {}: {:4.1f}ms'.format(i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_RUN * BATCH_SIZE / elapsed_time.sum()))


if __name__ == '__main__':
    model = ResNet50(weights='imagenet')
    benchmark_keras_model(model)
