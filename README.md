# tensorflow-yolov4-tflite

### Demo

```bash
# Convert darknet weights to tensorflow
## yolov4
python save_model.py --weights ./data/yolov4.weights --output ./weights/yolov4-416 --input_size 416 --model yolov4 

## yolov4-tiny
python save_model.py --weights ./data/yolov4-tiny.weights --output ./weights/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny 

# Run demo tensorflow
python detect.py --weights ./weights/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg

python detect.py --weights ./weights/yolov4-tiny-416 --size 416 --model yolov4 --image ./data/kite.jpg --tiny

```
If you want to run yolov3 or yolov3-tiny change ``--model yolov3`` in command


### Convert to tflite

```bash
# Save tf model for tflite converting
python save_model.py --weights ./data/yolov4.weights --output ./weights/yolov4-416 --input_size 416 --model yolov4 --framework tflite

# yolov4
python convert_tflite.py --weights ./weights/yolov4 --output ./weights/yolov4.tflite

# Run demo tflite model
python detect.py --weights ./weights/yolov4-416.tflite --size 416 --model yolov4 --image ./data/kite.jpg --framework tflite
```

### References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)


