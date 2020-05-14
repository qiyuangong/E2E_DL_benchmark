# End to end benchmark fro deep learning frameworks

In paper and existing benchmarks, we focus too much on this stage. However, during production, inference or predict is only one stage of end to end pipeline. This repo tries to build a end to end benchmark for existing deep learning frameworks.

End2End in this benchmark:

Image (in memory) -> Pre-processing -> Inference -> Post-processing -> Result (in meory)

**Note that:** We only collect metrics (latency and throughput) from Pre-processing to Post-processing.

## General Benchmark API

```python
def load_model(model_path):
    return model

def preprocessing(image):
    # Pre-processing
    return image

def predict(model, image):
    result = model.predict(image)
    return result

def postprocessing(result):
    # Top-1
    # Top-5

def benchmark(iteration=200):
    # create dummy data or read data from file path
    # preprocessing
    # predict
    # postprocessing
```

## Models

1. Resnet_50_v1
2. Inception_V3
3. MobileNet
4. FasterRCNN

## Usage

```bash
python ${benchmark name}.py -m ${model_path} -b ${batch_size} -i ${iteration}
```

## OpenVINO

```bash
#install openvino
# source openvino config
```

OpenVINO Python API.

## TensorFlow

```bash
pip install tensorflow==1.15.0
```

Keras (TensorFlow.Keras) and TensorFlow 1.15.X API.

## PyTorch

```bash
pip install pytorch
```

PyTorch and Torch Vision API.

## Reference

1. [OpenVINO](https://software.intel.com/en-us/openvino-toolkit)
2. [TensorFlow](https://www.tensorflow.org/)
3. [PyTorch](https://pytorch.org/)
4. [Analytics-Zoo](https://github.com/intel-analytics/analytics-zoo)
