# OpenVINO E2E Benchmark

Requirements:

1. OpenVINO

Get OpenVINO Resnet_50_v1 from OpenVINO model Zoo, or convert it using OpenVINO model optimizer.

```bash
python resnet_50_v1.py -m ${model.xml} -b ${batch_size} -i ${iteration}
```
