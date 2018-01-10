import os, numpy as np, random
import model as modellib, visualize, utils, det_utils as du
from maskrcnn.model import log
from clutter import ClutterDataset, ClutterConfig
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string('task', '', '')
# flags.DEFINE_string('logdir_prefix', 'output/', '')
flags.DEFINE_string('logdir', 'outputs/v1', '')
# flags.DEFINE_string('config_name', '', '')

def train():
  # Load the datasets, configs.
  config = ClutterConfig()
  config.display()
  model_dir = os.path.join(FLAGS.logdir)

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
  model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
    epochs=20, layers='all')
  model_path = os.path.join(model_dir, "mask_rcnn_clutter.h5")
  model.keras_model.save_weights(model_path)

def prepare_for_test():
  class InferenceConfig(ClutterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

  inference_config = InferenceConfig()
  model_dir = FLAGS.logdir

  # Recreate the model in inference mode
  model = modellib.MaskRCNN(mode="inference", config=inference_config,
    model_dir=model_dir)
  
  model_path = os.path.join(FLAGS.logdir, 'mask_rcnn_clutter.h5')

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
  # rng = np.random.RandomState(0)
  # image_ids = rng.choice(dataset_val.image_ids, 100)
  image_ids = dataset_val.image_ids
  
  tps, fps, scs, num_insts, dup_dets, inst_ids, ovs = [], [], [], [], [], [], []
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

    # Make sure scores are sorted.
    sc = r['scores']
    is_sorted = np.all(np.diff(sc) <= 0)
    assert(is_sorted)
    overlaps = utils.compute_overlaps(r['rois'], gt_bbox)
    dt = {'sc': sc[:,np.newaxis]*1.}
    gt = {'diff': np.zeros((gt_bbox.shape[0],1), dtype=np.bool)}
    tp, fp, sc, num_inst, dup_det, inst_id, ov = \
      du.inst_bench_image(dt, gt, {'minoverlap': 0.5}, overlaps)
    tps.append(tp); fps.append(fp); scs.append(sc); num_insts.append(num_inst);
    dup_dets.append(dup_det); inst_ids.append(inst_id); ovs.append(ov);
    
  # Compute AP
  ap, rec, prec, npos, _ = du.inst_bench(None, None, None, tp=tps, fp=fps, score=scs, numInst=num_insts)
  print("mAP: ", ap[0], "prec: ", np.max(prec), "rec: ", np.max(rec), "prec-1: ", prec[-1], "npos: ", npos)

def subplot(plt, Y_X, sz_y_sz_x=(10,10), space_y_x=(0.1,0.1), T=False):
  Y,X = Y_X
  sz_y, sz_x = sz_y_sz_x
  hspace, wspace = space_y_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X, squeeze=False)
  plt.subplots_adjust(wspace=wspace, hspace=hspace)
  if T:
    axes_list = axes.T.ravel()[::-1].tolist()
  else:
    axes_list = axes.ravel()[::-1].tolist()
  return fig, axes, axes_list

def vis():
  inference_config, model, dataset_val = prepare_for_test()
  
  # Test on a random image
  rng = np.random.RandomState(0)
  for i in range(10):
    image_id = rng.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_bbox)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    class_names = dataset_val.class_names
    class_names = {1: ''}

    fig, _, axes = subplot(plt, (2,6), sz_y_sz_x=(5,5))
    ax = axes.pop(); ax.imshow(original_image); ax.set_axis_off();
    for i in range(5):
      ax = axes.pop()
      if i < gt_bbox.shape[0]:
        visualize.display_instances(original_image, gt_bbox[i:i+1,:], gt_mask[:,:,i:i+1], 
          gt_class_id[i:i+1], class_names, ax=ax, title='')

    results = model.detect([original_image], verbose=1)
    r = results[0]

    ax = axes.pop(); ax.imshow(original_image); ax.set_axis_off()
    for i in range(5):
      ax = axes.pop()
      if i < r['rois'].shape[0]:
        visualize.display_instances(original_image, r['rois'][i:i+1,:], r['masks'][:,:,i:i+1], 
          r['class_ids'][i:i+1], class_names, title='{:0.3f}'.format(r['scores'][i]), ax=ax)
    
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
    #   dataset_val.class_names, r['scores'], ax=get_ax())
    file_name = os.path.join(model.model_dir, 'vis',
      'vis_{:06d}.png'.format(dataset_val.image_id[image_id]))
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
  if FLAGS.task == 'train':
    train()

  elif FLAGS.task == 'vis':
    vis()

  elif FLAGS.task == 'bench':
    benchmark()

  else:
    assert(False), 'Unknown option {:s}.'.format(FLAGS.task)

if __name__ == '__main__':
  app.run()
