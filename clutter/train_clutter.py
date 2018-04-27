import os, numpy as np, logging, skimage, datetime, itertools
from tqdm import tqdm
import model as modellib, visualize, utils, det_utils as du
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from model import log
from clutter import ClutterDataset, ClutterConfig
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt

import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'train', '')
flags.DEFINE_string('im_type', 'gray', '')
# flags.DEFINE_string('logdir_prefix', 'output/', '')
flags.DEFINE_string('logdir', 'outputs/v1', '')
flags.DEFINE_bool('vis_fn', False, '')
# flags.DEFINE_string('config_name', '', '')
flags.DEFINE_string('im_dir', '', '')

# CUDA_VISIBLE_DEVICES='2' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py \
#   --logdir outputs/v3_512_40_flip_depth --im_type depth --task train
# CUDA_VISIBLE_DEVICES='3' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py \
#   --logdir outputs/v3_512_40_flip --im_type gray --task benchmark
# CUDA_VISIBLE_DEVICES='2' PYTHONPATH='.:maskrcnn/' python clutter/train_clutter.py \
#   --logdir outputs/v3_512_40_flip_depth --im_type depth --task train

def mkdir_if_missing(output_dir):
  if not os.path.exists(output_dir):
    try:
      os.makedirs(output_dir)
    except:
      logging.error("Something went wrong in mkdir_if_missing. "
        "Probably some other process created the directory already.")

def tune_params(params, num_epochs=30):
  """Trains many nets with different hyperparameter settings for fewer epochs

  params: dictionary with the following structure:
    config_param1: [val1, val2, ..., valn]
    config_param2: [val1, val2, ..., valm]
  """
  m = 166. if FLAGS.im_type=='depth' else 128
  model_dir = os.path.join(FLAGS.logdir)
  tune_dir = 'tune_params{:%Y%m%dT%H%M}'.format(datetime.datetime.now())

  model_dir = os.path.join(model_dir, tune_dir)
  mkdir_if_missing(model_dir)

  config_list = (dict(zip(params, x)) for x in itertools.product(*params.values()))
  # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
  config_txt_path = os.path.join(model_dir, 'configs.txt')

  for i, params in enumerate(config_list):
    # open, write, close. So we can check on the file halfway through training
    f = open(config_txt_path, 'a')
    f.write(str(i) + " " + str(params) + '\n')
    f.close()

    config = ClutterConfig(mean=m)

    # Set the parameters in the config!
    print(params)
    for param, val in params.items():
      setattr(config, param, val)

    config_name = 'tune_' + str(i) + '_'
    setattr(config, 'NAME', config_name)
    config.display()

    # Training dataset
    dataset_train = ClutterDataset()
    dataset_train.load('train', FLAGS.im_type, 0)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ClutterDataset()
    dataset_val.load('test', FLAGS.im_type, 0)
    dataset_val.prepare()

    # Create the model.
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_dir)
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
                epochs=num_epochs, layers='all')

    model_name = 'mask_rcnn_clutter_tune_{}.h5'.format(i)

    model_path = os.path.join(model_dir, model_name)
    model.keras_model.save_weights(model_path)

def train():
  # Load the datasets, configs.
  m = 166. if FLAGS.im_type=='depth' else 128
  config = ClutterConfig(mean=m)
  config.LEARNING_RATE = 5e-6
  config.display()
  model_dir = os.path.join(FLAGS.logdir)

  # Training dataset
  dataset_train = ClutterDataset()
  dataset_train.load('train', FLAGS.im_type, 0)
  dataset_train.prepare()

  # Validation dataset
  dataset_val = ClutterDataset()
  dataset_val.load('test', FLAGS.im_type, 0)
  dataset_val.prepare()

  # Create the model.
  model = modellib.MaskRCNN(mode="training", config=config,
    model_dir=model_dir)

  model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
    epochs=100, layers='all')
  model_path = os.path.join(model_dir, "mask_rcnn_clutter.h5")
  model.keras_model.save_weights(model_path)

def prepare_for_test():
  class InferenceConfig(ClutterConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
  m = 166. if FLAGS.im_type=='depth' else 128
  inference_config = InferenceConfig(mean=m)
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
  dataset_val.load('test', FLAGS.im_type, 0)
  dataset_val.prepare()
  return inference_config, model, dataset_val

def compute_gt_stats(gt_bbox, gt_mask):
  # Compute statistics for all the ground truth things.
  hw = gt_bbox[:,2:] - gt_bbox[:,:2]
  hw = hw*1.
  min_side = np.min(hw,1)[:,np.newaxis]
  max_side = np.max(hw,1)[:,np.newaxis]
  aspect_ratio = np.max(hw, 1) / np.min(hw, 1)
  aspect_ratio = aspect_ratio[:,np.newaxis]
  log_aspect_ratio = np.log(aspect_ratio)
  box_area = np.prod(hw, 1)[:,np.newaxis]
  log_box_area = np.log(box_area)
  sqrt_box_area = np.sqrt(box_area)
  modal_area = np.sum(np.sum(gt_mask, 0), 0)[:,np.newaxis]*1.
  log_modal_area = np.log(modal_area)
  sqrt_modal_area = np.sqrt(modal_area)

  # Number of distinct components
  ov_connected = sqrt_box_area*1.
  for i in range(gt_mask.shape[2]):
    aa = skimage.measure.label(gt_mask[:,:,i], background=0)
    sz = np.bincount(aa.ravel())[1:]
    biggest = np.argmax(sz)+1
    big_comp = utils.extract_bboxes(aa[:,:,np.newaxis]==biggest)
    ov_connected[i,0] = utils.compute_overlaps(big_comp, gt_bbox[i:i+1,:])

  a = np.concatenate([min_side, max_side, aspect_ratio, log_aspect_ratio,
    box_area, log_box_area, sqrt_box_area, modal_area, log_modal_area,
    sqrt_modal_area, ov_connected], 1)
  n = ['min_side', 'max_side', 'aspect_ratio', 'log_aspect_ratio', 'box_area',
    'log_box_area', 'sqrt_box_area', 'modal_area', 'log_modal_area',
    'sqrt_modal_area', 'ov_connected']
  return a, n

def benchmark():
  inference_config, model, dataset_val = prepare_for_test()
  # rng = np.random.RandomState(0)
  # image_ids = rng.choice(dataset_val.image_ids, 100)
  image_ids = dataset_val.image_ids
  mkdir_if_missing(os.path.join(model.model_dir, 'vis_fn'))

  ms = [[] for _ in range(10)]
  thresh_all = [0.25, 0.5, 0.75]
  for ov in thresh_all:
    for m in ms:
      m.append([])
  ms.append(thresh_all)
  ms = list(zip(*ms))

  for image_id in tqdm(image_ids):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
      modellib.load_image_gt(dataset_val, inference_config, image_id,
        use_mini_mask=False)
    molded_images = modellib.mold_image(image, inference_config)
    molded_images = np.expand_dims(molded_images, 0)
    gt_stat, stat_name = compute_gt_stats(gt_bbox, gt_mask)

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

    for tps, fps, scs, num_insts, dup_dets, inst_ids, ovs, tp_inds, fn_inds, \
      gt_stats, thresh in ms:
      tp, fp, sc, num_inst, dup_det, inst_id, ov = \
        du.inst_bench_image(dt, gt, {'minoverlap': thresh}, overlaps)
      tp_ind = np.sort(inst_id[tp]); fn_ind = np.setdiff1d(np.arange(num_inst), tp_ind);
      tps.append(tp); fps.append(fp); scs.append(sc); num_insts.append(num_inst);
      dup_dets.append(dup_det); inst_ids.append(inst_id); ovs.append(ov);
      tp_inds.append(tp_ind); fn_inds.append(fn_ind);
      gt_stats.append(gt_stat)

    # Visualize missed objects.
    if FLAGS.vis_fn:
      fn_ind = ms[1][8][-1] # missing objects at threshold 0.5
      if fn_ind.size > 0:
        fig, _, axes = subplot(plt, (fn_ind.size+1, 1), sz_y_sz_x=(5,5))
        ax = axes.pop(); ax.imshow(image); ax.set_axis_off();
        class_names = {1: ''}
        for _ in range(fn_ind.size):
          j = fn_ind[_]
          ax = axes.pop()
          visualize.display_instances(image, gt_bbox[j:j+1,:],
              gt_mask[:,:,j:j+1], gt_class_id[j:j+1], class_names, ax=ax, title='')
        file_name = os.path.join(model.model_dir, 'vis_fn',
          'vis_{:06d}.png'.format(dataset_val.image_id[image_id]))
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        plt.close()

  # Compute AP
  for tps, fps, scs, num_insts, dup_dets, inst_ids, ovs, tp_inds, fn_inds, \
    gt_stats, thresh in ms:
    ap, rec, prec, npos, _ = \
      du.inst_bench(None, None, None, tp=tps, fp=fps, score=scs, numInst=num_insts)
    str_ = 'mAP: {:.3f}, prec: {:.3f}, rec: {:.3f}, npos: {:d}'.format(
      ap[0], np.min(prec), np.max(rec), npos)
    logging.error('%s', str_)
    # print("mAP: ", ap[0], "prec: ", np.max(prec), "rec: ", np.max(rec), "prec-1: ",
    #   prec[-1], "npos: ", npos)
    plt.style.use('fivethirtyeight') #bmh')
    fig, _, axes = subplot(plt, (3,4), (8,8), space_y_x=(0.2,0.2))
    ax = axes.pop(); ax.plot(rec, prec, 'r'); ax.set_xlim([0,1]); ax.set_ylim([0,1]);
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(str_) #'{:5.3f}'.format(ap[0]*100))
    plot_stats(stat_name, gt_stats, tp_inds, fn_inds, axes)
    file_name = os.path.join(model.model_dir, 'pr_stats_{:d}.png'.format(int(thresh*100)))
    logging.error('plot file name: %s', file_name)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_stats(stat_name, gt_stats, tp_inds, fn_inds, axes):
  # Accumulate all stats for positives, and negatives.
  tp_stats = [gt_stat[tp_ind, :] for gt_stat, tp_ind in zip(gt_stats, tp_inds)]
  tp_stats = np.concatenate(tp_stats, 0)
  fn_stats = [gt_stat[fn_ind, :] for gt_stat, fn_ind in zip(gt_stats, fn_inds)]
  fn_stats = np.concatenate(fn_stats, 0)
  all_stats = np.concatenate(gt_stats, 0)

  for i in range(all_stats.shape[1]):
    ax = axes.pop()
    ax.set_title(stat_name[i])
    # np.histogram(all_stats
    min_, max_ = np.percentile(all_stats[:,i], q=[1,99])
    all_stats_ = all_stats[:,i]*1.
    all_stats_ = all_stats_[np.logical_and(all_stats_ > min_, all_stats_ < max_)]
    _, bins = np.histogram(all_stats_, 'auto')
    bin_size = bins[1]-bins[0]
    s = np.unique(all_stats_); s = s[1:]-s[:-1]
    if bin_size < np.min(s):
      bin_size = np.min(s)
      bins = np.arange(bins[0], bins[-1]+bin_size, bin_size)
    for j, (m, n) in \
        enumerate(zip([tp_stats[:,i], fn_stats[:,i]], ['tp', 'fn'])):
      ax.hist(m, bins, alpha=0.5, label=n, linewidth=1, linestyle='-', ec='k')
    ax2 = ax.twinx()
    t, _ = np.histogram(all_stats[:,i], bins)
    mis, _ = np.histogram(fn_stats[:,i], bins)
    mis_rate, = ax2.plot((bins[:-1] + bins[1:])*0.5, mis / np.maximum(t, 0.00001),
      'm', label='mis rate')
    fract_data, = ax2.plot((bins[:-1] + bins[1:])*0.5, t / np.sum(t),
      'm--', label='data fraction')
    ax2.set_ylim([0,1])
    ax2.tick_params(axis='y', colors=mis_rate.get_color())
    ax2.yaxis.label.set_color(mis_rate.get_color())
    ax2.grid('off')
    ax.legend(loc=2)
    ax2.legend()

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

def vis_ext():
  """Visualize predictions on images provided from external location"""
  inference_config, model, dataset_val = prepare_for_test()

  mkdir_if_missing(os.path.join(model.model_dir, 'vis_ext'))

  assert FLAGS.im_dir, 'please specify a directory!'
  im_dir = FLAGS.im_dir

  print('Directory:', im_dir)

  for i, img in enumerate(os.scandir(im_dir)):
    im_path = img.path
    filename = os.path.split(im_path)[1]
    print('im_path', im_path)
    if im_path.endswith('.png'):
      # Load image
      image = skimage.io.imread(im_path)
      # If grayscale. Convert to RGB for consistency.
      if image.ndim != 3:
        image = skimage.color.gray2rgb(image)

      image, window, scale, padding = utils.resize_image(
        image,
        min_dim=inference_config.IMAGE_MIN_DIM,
        max_dim=inference_config.IMAGE_MAX_DIM,
        padding=inference_config.IMAGE_PADDING)

      results = model.detect([image], verbose=1)
      r = results[0]

      visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        dataset_val.class_names, r['scores'], ax=get_ax())
      file_name = os.path.join(model.model_dir, 'vis_ext',
                               'vis_{}.png'.format(filename))
      plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
      plt.close()

def vis():
  inference_config, model, dataset_val = prepare_for_test()

  # Test on a random image
  rng = np.random.RandomState(0)
  mkdir_if_missing(os.path.join(model.model_dir, 'vis'))

  for i in tqdm(range(100)):
    image_id = rng.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
    print("image_id", image_id)
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_bbox)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # class_names = dataset_val.class_names
    class_names = {1: ''}

    fig, _, axes = subplot(plt, (2,6), sz_y_sz_x=(5,5))
    ax = axes.pop(); ax.imshow(original_image); ax.set_axis_off();
    for i in range(5):
      ax = axes.pop()
      if i < gt_bbox.shape[0]:
        visualize.display_instances(original_image, gt_bbox[i:i+1,:],
          gt_mask[:,:,i:i+1], gt_class_id[i:i+1], class_names, ax=ax, title='')

    results = model.detect([original_image], verbose=1)
    r = results[0]

    ax = axes.pop(); ax.imshow(original_image); ax.set_axis_off()
    for i in range(5):
      ax = axes.pop()
      if i < r['rois'].shape[0]:
        visualize.display_instances(original_image, r['rois'][i:i+1,:],
          r['masks'][:,:,i:i+1], r['class_ids'][i:i+1], class_names,
          title='{:0.3f}'.format(r['scores'][i]), ax=ax)

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
  print("Task: {}\n".format(FLAGS.task))
  assert(FLAGS.task in ['train', 'vis', 'bench', 'tune', 'vis_ext'])
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    set_session(sess)

    if FLAGS.task == 'train':
      train()

    elif FLAGS.task == 'vis':
      vis()

    elif FLAGS.task == 'bench':
      benchmark()

    elif FLAGS.task == 'tune':
      params = {
        'LEARNING_RATE': [0.0005, 0.001, 0.005],
        'WEIGHT_DECAY': [0.00005, 0.0001, 0.0005],
      }
      tune_params(params, num_epochs=30)

    elif FLAGS.task == 'vis_ext':
      vis_ext()

    else:
      assert(False), 'Unknown option {:s}.'.format(FLAGS.task)

if __name__ == '__main__':
  app.run()
