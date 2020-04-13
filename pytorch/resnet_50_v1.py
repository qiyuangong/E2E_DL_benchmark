import torch
from PIL import Image
from torchvision import transforms


input_image = Image.open(filename)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    if model_path:
        torch.load(model_path)
    else:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
    model.eval()
    return model

def preprocessing(image):
    # Pre-processing
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 

def predict(model, image):
    result = model.model(image)
    return result

def postprocessing(result):
    # Top-1
    print(torch.nn.functional.softmax(output[0], dim=0))
    # Top-5

def benchmark():
    # create dummy data or read data from file path
    # preprocessing
    # predict
    # postprocessing


if __name__ == '__main__':
    benchmark()


