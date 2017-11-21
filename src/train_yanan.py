import cv2
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio
import math

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed
'''
by seasker
lr:学习率
batch_size:

'''
def main(lr, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu):
  data_path = "../data/traffic_data/"
  data=load_data_from_days_mat('/home/seasker/CNN-master/dataset/Traffic-data-mat/trafficDayFSOdata.mat','trafficDayFSOdata')
  data_meta=compute_data_meta(data,(0,1,2))
  trans_data=nomalize(data,data_meta,'maxmin')
  train_num=(data.shape[0]-K-T+1)//10*9
  margin = 0.3 
  updateD = True
  updateG = True
  iters = 0
  prefix  = ("traffic_MCNET"
          + "_image_size="+str(image_size[0])
          +"x"+str(image_size[1])
          + "_K="+str(K)
          + "_T="+str(T)
          + "_batch_size="+str(batch_size)
          + "_alpha="+str(alpha)
          + "_beta="+str(beta)
          + "_lr="+str(lr))

  print("\n"+prefix+"\n")
  checkpoint_dir = "../models/"+prefix+"/"
  samples_dir = "../samples/"+prefix+"/"
  summary_dir = "../logs/"+prefix+"/"

  if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
  if not exists(samples_dir):
    makedirs(samples_dir)
  if not exists(summary_dir):
    makedirs(summary_dir)

  with tf.device("/gpu:%d"%gpu):
    model = MCNET(image_size=[image_size[0],image_size[1]], c_dim=3,
                  K=K, batch_size=batch_size, T=T,
                  checkpoint_dir=checkpoint_dir)
    d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        model.d_loss, var_list=model.d_vars
    )
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        alpha*model.L_img+beta*model.L_GAN, var_list=model.g_vars
    )

 # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                  log_device_placement=False
                  #gpu_options=gpu_options
                  )) as sess:

    tf.global_variables_initializer().run()

    if model.load(sess, checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    g_sum = tf.summary.merge([model.L_p_sum,
                              model.L_gdl_sum, model.loss_sum,
                              model.L_GAN_sum])
    d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                              model.d_loss_fake_sum])
    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    counter = iters+1
    start_time = time.time()

    with Parallel(n_jobs=4) as parallel:
      while iters < 100:
        mini_batches = get_minibatches_idx(train_num, batch_size, shuffle=False)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            seq_batch  = np.zeros((batch_size, image_size[0], image_size[1],
                                   K+T, 3), dtype="float32")
            diff_batch = np.zeros((batch_size, image_size[0], image_size[1],
                                   K-1, 3), dtype="float32")
            t0 = time.time()
            Ts = np.repeat(np.array([T]),batch_size,axis=0)
            Ks = np.repeat(np.array([K]),batch_size,axis=0)
            
            shapes = np.repeat(np.array([image_size]),batch_size,axis=0)
            #f: txt file, p
            output = parallel(delayed(get_sample_data)(trans_data,i, k, t)
                                                 for i,k,t in zip(batchidx,Ks, Ts))
            print(type(output[0][1]))
            for i in range(batch_size): 
            
              seq_batch[i] = output[i][0]
              diff_batch[i] = output[i][1]

            if updateD:
              _, summary_str = sess.run([d_optim, d_sum],
                                         feed_dict={model.diff_in: diff_batch,
                                                    model.xt: seq_batch[:,:,:,K-1],
                                                    model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            if updateG:
              _, summary_str = sess.run([g_optim, g_sum],
                                         feed_dict={model.diff_in: diff_batch,
                                                    model.xt: seq_batch[:,:,:,K-1],
                                                    model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            errD_fake = model.d_loss_fake.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})
            errD_real = model.d_loss_real.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})
            errG = model.L_GAN.eval({model.diff_in: diff_batch,
                                     model.xt: seq_batch[:,:,:,K-1],
                                     model.target: seq_batch})
            errG_img = model.L_img.eval({model.diff_in: diff_batch,
                                     model.xt: seq_batch[:,:,:,K-1],
                                     model.target: seq_batch}) 

            pred=model.G.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})

            print('real')
            print(seq_batch[0,:,:,K:]) 
            print('pred')

            print(np.squeeze(pred[0]))    


            print('diff')
            print(seq_batch[0,:,:,K]-np.squeeze(pred[0]))                  

            if errD_fake < margin or errD_real < margin:
              updateD = False
            if errD_fake > (1.-margin) or errD_real > (1.-margin):
              updateG = False
            if not updateD and not updateG:
              updateD = True
              updateG = True

            counter += 1
  
            print(
                "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f,L_img:%.8f" 
                % (iters, time.time() - start_time, errD_fake+errD_real,errG,errG_img)
            )

            if np.mod(counter, 50) == 1:
              batch_pred = sess.run([model.G],
                                  feed_dict={model.diff_in:diff_batch,
                                             model.xt: seq_batch[:,:,:,K-1],
                                             model.target: seq_batch})[0]
              batch_pred = batch_pred[0].swapaxes(0,2).swapaxes(1,2)
              batch_real = seq_batch[0,:,:,K:].swapaxes(0,2).swapaxes(1,2)
              pred=inverse_maxmin_normalize(batch_pred,data_meta)
              real=inverse_maxmin_normalize(batch_real,data_meta)
              print(pred)
              print('')
              print(real)
              mae,mre,rmse=compute_matrics(pred,real)
              print('mae:',mae,'  ','mre:',mre,'  ','rmse:',rmse)
              time.sleep(30)


              
              # samples = np.concatenate((samples,sbatch), axis=0)
              # print("Saving sample ...")
              # save_images(samples[:,:,:,:], [2, T], 
              #             samples_dir+"train_%s.png" % (iters))
            if np.mod(counter, 500) == 2:
              model.save(sess, checkpoint_dir, counter)
  
            iters += 1

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=7, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=0.02, help="GAN loss weight")
  parser.add_argument("--image_size", type=list, dest="image_size",
                      default=[35,168], help="Mini-batch size")
  parser.add_argument("--K", type=int, dest="K",
                      default=7, help="Number of steps to observe from the past")
  parser.add_argument("--T", type=int, dest="T",
                      default=1, help="Number of steps into the future")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=100000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", default=0,
                      help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))
