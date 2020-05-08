import time
import cv2
import PIL
import os
from glob import glob

"""
Base class for benchmark
"""
class Benchmark:

    def __init__(self):
        pass

    def read_image(self, input_path):
        input_path = os.path.normpath(input_path)
        files = []
        if os.path.isdir(input_path):
            print("Reading from dir")
            path = os.path.join(input_path, '*')
            files = glob(path, recursive=True)
        elif os.path.isfile(input_path) and input_path.endswith(".txt"):
            print("Reading from val.txt")
            with open(input_path, "r") as test_file:
                for line in test_file:
                    files.append(os.path.join(os.path.dirname(input_path), line.split()[0]))
        else:
            print("Reading from file")
            files = [os.path.normpath(input_path)]
        return files

    def load_model(self, model_path=""):
        # Return model
        pass

    def preprocessing(self, image):
        # Preprocessing data
        pass

    def predict(self, model, image):
        # result = model.predict(image)
        # return result
        pass

    def postprocessing(self, result):
        # Top-1
        # Top-5
        pass

    def benchmark(self, model, image, batch_size=4, iterations=10):
        # create dummy data or read data from file path
        # input_batch = torch.rand(batch_size, 3, 224, 224) * 256
        input_image = image
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
    benchmark = Benchmark()
    print(benchmark.read_image("/Users/qiyuangong/Develop/Datasets/val_bmp_32/val.txt"))