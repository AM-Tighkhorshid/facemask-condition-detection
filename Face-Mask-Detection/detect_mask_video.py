# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pathlib
import serial
s = serial.Serial('COM1',9600)

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

# load the face mask detector model from disk
filenames = list(pathlib.Path('./files/training/').glob('*.index'))
filenames.sort()
print(filenames)

pipeline_file = "./files/pipeline.config"

#recover our saved model
pipeline_config = pipeline_file
#generally you want to put the last ckpt from training in here
model_dir = str(filenames[-1]).replace('.index','')
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join(str(filenames[-1]).replace('.index','')))

detect_fn = get_model_detection_function(detection_model)


# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    image_np = frame.astype(np.uint8)
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    # (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    new_class = detections['detection_classes'][0][0]
    score = detections['detection_scores'][0][0]
    new_box = detections['detection_boxes'][0][0]
    

    # label = "without_mask"
    if float(score) > 0.5:
        if int(new_class) == 0:  
            s.write(b'0')
            label = "incorrect_mask_wearing"
            color = (0, 255, 255)
        elif int(new_class) == 1:
            s.write(b'1')
            label = "with_mask"
            color = (0, 255, 0)
        else:
            s.write(b'0')
            label = "without_mask"
            color = (0, 0, 255)
            
        
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, float(score) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (int(new_box[1]* frame.shape[1]), int(new_box[0] * frame.shape[0]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (int(new_box[1] * frame.shape[1]), int(new_box[0] * frame.shape[0])), (int(new_box[3] * frame.shape[1]), int(new_box[2] * frame.shape[0])), color, 2)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

