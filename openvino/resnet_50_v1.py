import time
from openvino.inference_engine import IENetwork, IECore


ie = IECore()


def load_model(model_path):
    xml_filename = os.path.abspath(model_path)
    head, _ = os.path.splitext(xml_filename)
    bin_filename = os.path.abspath(head + ".bin")
    ie_network = IENetwork(xml_filename, bin_filename)
    exe_network = ie.load_network(ie_network, "CPU")
    return exe_network


def preprocessing(image):
    start_time = time.time()
    image = aspect_preserving_resize(image, 256)
    image = central_crop(image, output_height, output_width)
    image = normalization(image, [103.939, 116.779, 123.68])
    image = NHWC2HCHW(image)
    # ensure our NumPy array is C-contiguous as well,
    # otherwise we won't be able to serialize it
    print("Pre-processing %d ms" % int(round((time.time() - start_time) * 1000)))
    return image


def predict(model, image_path):
    infer_requests = model.requests
    return result

def postprocessing(result):
    return result


def benchmark(image_path)
    


if __name__ == '__main__':
    benchmark()
