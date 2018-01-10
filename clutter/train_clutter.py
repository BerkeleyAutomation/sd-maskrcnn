import os, numpy as np, random
import maskrcnn.model as modellib, visualize
from maskrcnn.model import log
from clutter import ClutterDataset, ClutterConfig
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt

def train():
  # Load the datasets, configs.
  config = ClutterConfig()
  config.display()
  model_dir = os.path.join('outputs/v1/') 

  # Training dataset
  dataset_train = ClutterDataset()
  dataset_train.load('train', 'gray', 0)
  dataset_train.prepare()

  # Validation dataset
  dataset_val = ClutterDataset()
  dataset_val.load('test', 'gray', 0)
  dataset_val.prepare()

  # Create the model.
  model = modellib.MaskRCNN(mode="training", config=config,
    model_dir=model_dir)
  
  # model.load_weights(model.find_last()[1], by_name=True)

  # Train the model.
  # Train the head branches
  # Passing layers="heads" freezes all layers except the head
  # layers. You can also pass a regular expression to select
  # which layers to train by name pattern.
  model.train(dataset_train, dataset_val, 
              learning_rate=config.LEARNING_RATE, 
              epochs=10, 
              layers='all')

def prepare_for_test():
  class InferenceConfig(ClutterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

  inference_config = InferenceConfig()
  model_dir = 'outputs/v1/clutter20180109T1810/'

  # Recreate the model in inference mode
  model = modellib.MaskRCNN(mode="inference", config=inference_config,
    model_dir=model_dir)
  
  model_path = 'outputs/v1/clutter20180109T1810/mask_rcnn_clutter_0008.h5'

  # Load trained weights (fill in path to trained weights here)
  assert model_path != "", "Provide path to trained weights"
  print("Loading weights from ", model_path)
  model.load_weights(model_path, by_name=True)

  # Load dataset.
  dataset_val = ClutterDataset()
  dataset_val.load('test', 'gray', 0)
  dataset_val.prepare()
  return inference_config, model, dataset_val

def benchmark():
  # Compute VOC-Style mAP @ IoU=0.5
  # Running on 10 images. Increase for better accuracy.
  inference_config, model, dataset_val = prepare_for_test()

  image_ids = np.random.choice(dataset_val.image_ids, 100)
  APs = []
  for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
      modellib.load_image_gt(dataset_val, inference_config, image_id,
        use_mini_mask=False)
    molded_images = modellib.mold_image(image, inference_config)
    molded_images = np.expand_dims(molded_images, 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id,
      r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
  print("mAP: ", np.mean(APs))

def vis():
  # Test on a random image
  image_id = random.choice(dataset_val.image_ids)
  original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
      modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
  
  log("original_image", original_image)
  log("image_meta", image_meta)
  log("gt_class_id", gt_bbox)
  log("gt_bbox", gt_bbox)
  log("gt_mask", gt_mask)

  visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    dataset_val.class_names, figsize=(8, 8))

  results = model.detect([original_image], verbose=1)

  r = results[0]
  visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
    dataset_val.class_names, r['scores'], ax=get_ax())
  file_name = os.path.join(model.model_dir, 'vis.png')
  plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
  plt.close()

def get_ax(rows=1, cols=1, size=8):
  """Return a Matplotlib Axes array to be used in
  all visualizations in the notebook. Provide a
  central point to control graph sizes.
  
  Change the default size attribute to control the size
  of rendered images
  """
  _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
  return ax

def main(_):
  # train()
  benchmark()

if __name__ == '__main__':
  app.run()
