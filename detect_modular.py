import tensorflow as tf
import cv2
import numpy as np


import core.utils as utils
from core.yolov4 import filter_boxes

from tensorflow.python.saved_model import tag_constants

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class FLAGS:
    framework = "tflite"
    weights = "weights/yolov4-tiny.tflite"
    size = 416
    tiny = True
    model = "yolov4"
    iou = 0.45
    score = 0.52

def preprocess_cam(image_path):
    original_image = cv2.imread(image_path)
    return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

def detect_image(original_image):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size


    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    
    images_data = np.asarray(images_data).astype(np.float32)

    interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return utils.detect_coordinates(original_image, pred_bbox)


### USAGE ### 
img = preprocess_cam(r"4727955343_0bb23ac4ae.jpg")
x = detect_image(img)

### TESTING ###
original_image = cv2.imread("4727955343_0bb23ac4ae.jpg")
for i in x:
    original_image = cv2.circle(original_image, i['center'], 1, (0, 0, 255), 5)
    cv2.imshow("x", cv2.cvtColor(i['image'], cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imshow("x", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
