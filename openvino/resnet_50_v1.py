import time
import os
import numpy as np
import cv2
from openvino.inference_engine import IECore, IENetwork
from utils import aspect_preserving_resize, central_crop, NHWC2HCHW

ie_network = None

def load_model(model_path):
    global ie_network
    ie = IECore()
    xml_filename = os.path.abspath(model_path)
    head, _ = os.path.splitext(xml_filename)
    bin_filename = os.path.abspath(head + ".bin")
    ie_network = ie.read_network(xml_filename, bin_filename)
    ie_network.batch_size = 1
    exe_network = ie.load_network(ie_network, "CPU")
    return exe_network


def preprocessing(image):
    start_time = time.time()
    image = aspect_preserving_resize(image, 256)
    image = central_crop(image, 224, 224)
    # image = normalization(image, [103.939, 116.779, 123.68])
    image = NHWC2HCHW(image)
    image = np.expand_dims(image, axis=0)
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    return image


def predict(model, image):
    input_blob = next(iter(ie_network.inputs))
    out_blob = next(iter(ie_network.outputs))
    result = model.infer(inputs = {input_blob: image})
    return result[out_blob]

def postprocessing(result):
    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(result):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-1:][::-1]
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            label_length = 1000
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, 1000,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
    return result


def benchmark(model, image_path="", batch_size=4, iterations=1):
    # create dummy data or read data from file path
    # input_batch = torch.rand(batch_size, 3, 224, 224) * 256
    input_image = cv2.imread("/Users/qiyuangong/Develop/Datasets/val_bmp_inception/ILSVRC2012_val_00000001.bmp")
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
    model = load_model("/Users/qiyuangong/Develop/Models/resnet_v1_50/resnet_v1_50.xml")
    benchmark(model)
