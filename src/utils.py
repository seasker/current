
"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import cv2
import random
import imageio
import scipy.misc
import scipy.io as sio
import numpy as np
import os



def gaussian_normalize(data,data_meta):
  data=data-data_meta['mean']
  data=data/(data_meta['std']+np.finfo(np.float64).eps)
  return data

def maxmin_normalize(data,data_meta):
  data=data-data_meta['min']
  data=data/(data_meta['max']-data_meta['min']+np.finfo(np.float64).eps)
  return data


def inverse_gassian_normalize(data,data_meta):
  data=data*data_meta['std']
  data=data+data_meta['mean']
  return data

def inverse_maxmin_normalize(data,data_meta):
  data=data*(data_meta['max']-data_meta['min'])
  data=data+data_meta['min']
  return data

def nomalize(data,data_meta,method='gaussian'):
  if method=='gaussian':
    return gaussian_normalize(data,data_meta)
  elif method=='maxmin':
    return maxmin_normalize(data,data_meta)


def inverse_normalize(data,data_meta,method):
  if method=='gaussian':
    return inverse_gassian_normalize(data,data_meta)
  elif method=='maxmin':
    return inverse_maxmin_normalize(data,data_meta)

def computeMRE(pred,real):
  mre_mat=np.zeros_like(pred)
  elig_ind=np.where(real>0)
  mre_mat[elig_ind]=np.abs((pred[elig_ind]-real[elig_ind])/real[elig_ind])
  mre=mre_mat.mean()
  return mre
  
def computeMAE(pred,real):
  mae_mat=np.abs(pred-real)
  mae=mae_mat.mean()
  return mae

def computeRMSE(pred,real):
  rmse_mat=np.power(pred-real,2)
  rmse=np.sqrt(rmse_mat.mean())
  return rmse


def compute_matrics(pred,real,index_type='all'):
  if index_type=='mae':
    return computeMAE(pred,real)
  elif index_type=='mre':
    return computeMRE(pred,real)
  elif index_type=='rmse':
    return computeRMSE(pred,real)
  else:
    mae=computeMAE(pred,real)
    mre=computeMRE(pred,real)
    rmse=computeRMSE(pred,real)
    return mae,mre,rmse

def transform(image):
    return image/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def save_images(images, size, image_path):
  return imsave(inverse_transform(images)*255., size, image_path)


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h * size[0], w * size[1], 3))

  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx / size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img


def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0 
  for i in range(n // minibatch_size):
    minibatches.append(idx_list[minibatch_start:
                                minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n): 
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)


def draw_frame(img, is_input):
  if img.shape[2] == 1:
    img = np.repeat(img, [3], axis=2)

  if is_input:
    img[:2,:,0]  = img[:2,:,2] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,2] = 0 
    img[:,-2:,0] = img[:,-2:,2] = 0 
    img[:2,:,1]  = 255 
    img[:,:2,1]  = 255 
    img[-2:,:,1] = 255 
    img[:,-2:,1] = 255 
  else:
    img[:2,:,0]  = img[:2,:,1] = 0 
    img[:,:2,0]  = img[:,:2,2] = 0 
    img[-2:,:,0] = img[-2:,:,1] = 0 
    img[:,-2:,0] = img[:,-2:,1] = 0 
    img[:2,:,2]  = 255 
    img[:,:2,2]  = 255 
    img[-2:,:,2] = 255 
    img[:,-2:,2] = 255 

  return img 


def load_kth_data(f_name, data_path, image_size, K, T): 
  flip = np.random.binomial(1,.5,1)[0]
  tokens = f_name.split()
  vid_path = data_path + tokens[0] + "_uncomp.avi"
  vid = imageio.get_reader(vid_path,"ffmpeg")
  low = int(tokens[1])
  high = np.min([int(tokens[2]),vid.get_length()])-K-T+1
  if low == high:
    stidx = 0 
  else:
    if low >= high: print(vid_path)
    stidx = np.random.randint(low=low, high=high)
  seq = np.zeros((image_size, image_size, K+T, 1), dtype="float32")
  for t in xrange(K+T):
    img = cv2.cvtColor(cv2.resize(vid.get_data(stidx+t),
                       (image_size,image_size)),
                       cv2.COLOR_RGB2GRAY)
    seq[:,:,t] = transform(img[:,:,None])

  if flip == 1:
    seq = seq[:,::-1]

  diff = np.zeros((image_size, image_size, K-1, 1), dtype="float32")
  for t in xrange(1,K):
    prev = inverse_transform(seq[:,:,t-1])
    next = inverse_transform(seq[:,:,t])
    diff[:,:,t-1] = next.astype("float32")-prev.astype("float32")

  return seq, diff


def load_data_from_days_mat(mat_data_path_name,mat_variable_name,get_meta=False):
  """
  mat

  """
  mat_data=sio.loadmat(mat_data_path_name)
  data=mat_data[mat_variable_name].astype('float32')
  if get_meta:
    meta=compute_data_meta(data,axis=(0,1,2))
    return data,meta
  else:
    return data

def compute_data_meta(data,axis=None):
  meta={}
  meta['max']=data.max(axis=axis)
  meta['min']=data.min(axis=axis)
  meta['mean']=data.mean(axis=axis)
  meta['std']=data.std(axis=axis)
  return meta

def get_sample_data(data,day_start_index, K, T):
  sample_seq=data[day_start_index:day_start_index+K+T].transpose([1,2,0,3])
  sample_diff=(data[day_start_index+1:day_start_index+K]-data[day_start_index:day_start_index+K-1]).transpose([1,2,0,3])
  return sample_seq,sample_diff

def load_s1m_data(f_name, data_path, trainlist, K, T):
  flip = np.random.binomial(1,.5,1)[0]
  vid_path = data_path + f_name
  img_size = [240,320]

  while True:
    try:
      vid = imageio.get_reader(vid_path,"ffmpeg")
      low = 1
      high = vid.get_length()-K-T+1
      if low == high:
        stidx = 0
      else:
        stidx = np.random.randint(low=low, high=high)
      seq = np.zeros((img_size[0], img_size[1], K+T, 3),
                     dtype="float32")
      for t in xrange(K+T):
        img = cv2.resize(vid.get_data(stidx+t),
                         (img_size[1],img_size[0]))[:,:,::-1]
        seq[:,:,t] = transform(img)

      if flip == 1:
        seq = seq[:,::-1]

      diff = np.zeros((img_size[0], img_size[1], K-1, 1),
                      dtype="float32")
      for t in xrange(1,K):
        prev = inverse_transform(seq[:,:,t-1])*255
        prev = cv2.cvtColor(prev.astype("uint8"),cv2.COLOR_BGR2GRAY)
        next = inverse_transform(seq[:,:,t])*255
        next = cv2.cvtColor(next.astype("uint8"),cv2.COLOR_BGR2GRAY)
        diff[:,:,t-1,0] = (next.astype("float32")-prev.astype("float32"))/255.
      break
    except Exception:
      # In case the current video is bad load a random one 
      rep_idx = np.random.randint(low=0, high=len(trainlist))
      f_name = trainlist[rep_idx]
      vid_path = data_path + f_name

  return seq, diff

