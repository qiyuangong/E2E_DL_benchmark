from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import time


def load_model():
    model = ResNet50(weights='imagenet')
    return model

def preprocessing(image):
    # Pre-processing
    input_tensor = preprocess_input(np.expand_dims(image, axis=0))
    return input_tensor

def predict(model, image):
    result = model.predict(image)
    return result

def postprocessing(result):
    # Top-1
    preds = decode_predictions(result, top=1)
    # print(preds)
    # Top-5

def benchmark(model, image_path="", batch_size=4, iterations=1):
    # create dummy data or read data from file path
    # input_batch = torch.rand(batch_size, 3, 224, 224) * 256
    input_image = image.load_img("/Users/qiyuangong/Develop/Datasets/val_bmp_inception/ILSVRC2012_val_00000001.bmp", target_size=(224, 224))
    start = time.time()
    preprocessing_time = 0
    predict_time = 0
    postprocessing_time = 0
    for _ in range(iterations):
        # preprocessing
        batch_start = time.time()
        model_input = preprocessing(input_image)
        pre_end = time.time()
        preprocessing_time += pre_end - batch_start
        # predict
        result = predict(model, model_input)
        predict_end = time.time()
        predict_time += predict_end - pre_end
        # postprocessing
        postprocessing(result)
        postprocessing_time += time.time() - predict_end
    total_time = time.time() - start
    print("Total Running time %.2f s" % total_time)
    print("Average batch time %.2f ms" % (total_time * 1000 / iterations))
    print("Average preprocessing time %.2f ms" % (preprocessing_time * 1000 / iterations))
    print("Average predict time %.2f ms" % (predict_time * 1000 / iterations))
    print("Average postprocessing time %.2f ms" % (postprocessing_time * 1000 / iterations))


if __name__ == '__main__':
    model = load_model()
    benchmark(model)