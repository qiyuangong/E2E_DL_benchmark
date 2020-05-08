import time


class Benchmark(Object):

    def __init__(self):
        pass

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
