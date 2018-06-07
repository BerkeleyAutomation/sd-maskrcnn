import numpy as np
import skimage
import utils

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
    # s = np.unique(all_stats_); s = s[1:]-s[:-1]
    # if bin_size < np.min(s):
    #   bin_size = np.min(s)
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
