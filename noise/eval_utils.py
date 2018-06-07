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

def voc_ap_fast(rec, prec):
  rec = rec.reshape((-1,1))
  prec = prec.reshape((-1,1))
  z = np.zeros((1,1)) 
  o = np.ones((1,1))
  mrec = np.vstack((z, rec, o))
  mpre = np.vstack((z, prec, z))
  mpre_ = np.maximum.accumulate(mpre[::-1])[::-1]
  I = np.where(mrec[1:] != mrec[0:-1])[0]+1;
  ap = np.sum((mrec[I] - mrec[I-1]) * mpre[I])
  return np.array(ap).reshape(1,)

def calc_pr(gt, out, wt=None, fast=False):
  """Computes VOC 12 style AP (dense sampling).
  returns ap, rec, prec"""
  if wt is None:
    wt = np.ones((gt.size,1))

  gt = gt.astype(np.float64).reshape((-1,1))
  wt = wt.astype(np.float64).reshape((-1,1))
  out = out.astype(np.float64).reshape((-1,1))

  gt = gt*wt
  tog = np.concatenate([gt, wt, out], axis=1)*1.
  ind = np.argsort(tog[:,2], axis=0)[::-1]
  tog = tog[ind,:]
  cumsumsortgt = np.cumsum(tog[:,0])
  cumsumsortwt = np.cumsum(tog[:,1])
  prec = cumsumsortgt / cumsumsortwt
  rec = cumsumsortgt / np.sum(tog[:,0])

  if fast:
    ap = voc_ap_fast(rec, prec)
  else:
    ap = voc_ap(rec, prec)
  return ap, rec, prec

def inst_bench_image(dt, gt, bOpts, overlap=None):
  nDt = len(dt['sc'])
  nGt = len(gt['diff'])
  numInst = np.sum(gt['diff'] == False)

  if overlap is None:
    overlap = bbox_utils.bbox_overlaps(dt['boxInfo'].astype(np.float), gt['boxInfo'].astype(np.float))
  
	# assert(issorted(-dt.sc), 'Scores are not sorted.\n');
  sc = dt['sc'];

  det    = np.zeros((nGt,1)).astype(np.bool)
  tp     = np.zeros((nDt,1)).astype(np.bool)
  fp     = np.zeros((nDt,1)).astype(np.bool)
  dupDet = np.zeros((nDt,1)).astype(np.bool)
  instId = np.zeros((nDt,1)).astype(np.int32)
  ov     = np.zeros((nDt,1)).astype(np.float32)

  # Walk through the detections in decreasing score
  # and assign tp, fp, fn, tn labels
  for i in range(nDt):
    # assign detection to ground truth object if any
    if nGt > 0:
      maxOverlap = overlap[i,:].max(); maxInd = overlap[i,:].argmax();
      instId[i] = maxInd; ov[i] = maxOverlap;
    else:
      maxOverlap = 0; instId[i] = -1; maxInd = -1;
    # assign detection as true positive/don't care/false positive
    if maxOverlap >= bOpts['minoverlap']:
      if gt['diff'][maxInd] == False:
        if det[maxInd] == False:
          # true positive
          tp[i] = True;
          det[maxInd] = True;
        else:
          # false positive (multiple detection)
          fp[i] = True;
          dupDet[i] = True;
    else:
      # false positive
      fp[i] = True;
  return tp, fp, sc, numInst, dupDet, instId, ov

def inst_bench(dt, gt, bOpts, tp=None, fp=None, score=None, numInst=None):
  """
  ap, rec, prec, npos, details = inst_bench(dt, gt, bOpts, tp = None, fp = None, sc = None, numInst = None)
  dt  - a list with a dict for each image and with following fields
    .boxInfo - info that will be used to cpmpute the overlap with ground truths, a list
    .sc - score
  gt
    .boxInfo - info used to compute the overlap,  a list
    .diff - a logical array of size nGtx1, saying if the instance is hard or not
  bOpt
    .minoverlap - the minimum overlap to call it a true positive
  [tp], [fp], [sc], [numInst]
      Optional arguments, in case the inst_bench_image is being called outside of this function
  """
  details = None
  if tp is None:
    # We do not have the tp, fp, sc, and numInst, so compute them from the structures gt, and out
    tp = []; fp = []; numInst = []; score = []; dupDet = []; instId = []; ov = [];
    for i in range(len(gt)):
      # Sort dt by the score
      sc = dt[i]['sc']
      bb = dt[i]['boxInfo']
      ind = np.argsort(sc, axis = 0);
      ind = ind[::-1]
      if len(ind) > 0:
        sc = np.vstack((sc[i,:] for i in ind))
        bb = np.vstack((bb[i,:] for i in ind))
      else:
        sc = np.zeros((0,1)).astype(np.float)
        bb = np.zeros((0,4)).astype(np.float)

      dtI = dict({'boxInfo': bb, 'sc': sc})
      tp_i, fp_i, sc_i, numInst_i, dupDet_i, instId_i, ov_i = inst_bench_image(dtI, gt[i], bOpts)
      tp.append(tp_i); fp.append(fp_i); score.append(sc_i); numInst.append(numInst_i);
      dupDet.append(dupDet_i); instId.append(instId_i); ov.append(ov_i);
    details = {'tp': list(tp), 'fp': list(fp), 'score': list(score), 'dupDet': list(dupDet),
      'numInst': list(numInst), 'instId': list(instId), 'ov': list(ov)}

  tp = np.vstack(tp[:])
  fp = np.vstack(fp[:])
  sc = np.vstack(score[:])

  cat_all = np.hstack((tp,fp,sc))
  ind = np.argsort(cat_all[:,2])
  cat_all = cat_all[ind[::-1],:]
  tp = np.cumsum(cat_all[:,0], axis = 0);
  fp = np.cumsum(cat_all[:,1], axis = 0);
  thresh = cat_all[:,2];
  npos = np.sum(numInst, axis = 0);

  # Compute precision/recall
  rec = tp / npos;
  prec = np.divide(tp, (fp+tp));
  ap = voc_ap_fast(rec, prec);
  return ap, rec, prec, npos, details