import torch
from PIL import Image
import time
from torchvision import transforms

# input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path=""):
    if model_path:
        torch.load(model_path)
    else:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
    model.eval()
    return model

def preprocessing(image):
    # Pre-processing
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict(model, image):
    result = model.model(image)
    return result

def postprocessing(result):
    # Top-1
    preds = torch.topk(result, 1)
    # print(preds)
    # Top-5

def benchmark(model, image_path, batch_size=4, iterations=1):
    # create dummy data or read data from file path
    # input_batch = torch.rand(batch_size, 3, 224, 224) * 256
    input_image = Image.open(image_path)
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
        result = model.forward(model_input)
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
