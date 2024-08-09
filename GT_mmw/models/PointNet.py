import os
import sys
import tensorflow as tf
import numpy as np
"""
Implement a PointNet Architecture like the paper

"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))

from pointnet2_color_feat_states import *
import graph_rnn_modules as modules
import tf_util
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, seq_length, num_points):
  
 
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, seq_length, num_points,1 ))
  
  return pointclouds_pl, labels_pl
  
def get_model(point_cloud, is_training, model_params):
  
  """ Classification PointNet, input is BxNx3, output BxNx2 """
  # Get model parameters
  batch_size = point_cloud.get_shape()[0].value
  seq_length =point_cloud.get_shape()[1].value
  num_points = point_cloud.get_shape()[2].value
  sampled_points = num_points
  num_samples = model_params['num_samples']
  context_frames = model_params['context_frames']
  sampled_points_down1 = model_params['sampled_points_down1'] #not used
  sampled_points_down2 = model_params['sampled_points_down2'] #not used
  sampled_points_down3 = model_params['sampled_points_down3'] #not used
  BN_FLAG = model_params['BN_FLAG']
  bn_decay = model_params['bn_decay']
  out_channels = model_params['out_channels'] #not used 
  drop_rate = model_params['drop_rate'] # not used
  graph_module_name = model_params['graph_module'] #not used
  end_points = {}
  
  context_frames = 0
  original_num_points =num_points
  
  
  # Add relative time-stamp to to point cloud
  timestep_tensor = tf.zeros( (batch_size,1,original_num_points,1) )
  for f in range(1, seq_length):
    frame_tensor = tf.ones( (batch_size,1,original_num_points,1) ) * f
    timestep_tensor = tf.concat( (timestep_tensor, frame_tensor) , axis = 1 )
    
    
  num_points = original_num_points *seq_length
  point_cloud = tf.reshape(point_cloud, (batch_size, seq_length * original_num_points, 3) )
  timestep_tensor = tf.reshape(timestep_tensor, (batch_size,seq_length *original_num_points, 1, 1) )
  print("point_cloud.shape", point_cloud.shape)
  
  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
  point_cloud_transformed = tf.matmul(point_cloud, transform)
  input_image = tf.expand_dims(point_cloud_transformed, -1)
 
  #save transform matrix
  end_points['transform_l0']= transform
  
    
  print("input_image", input_image)
  print("timestep_tensor", timestep_tensor)
  input_image = tf.concat( (input_image, timestep_tensor), axis =2 )
  print("input_image", input_image)

  
  net = tf_util.conv2d(input_image, 64, [1,4],
                      padding='VALID', stride=[1,1],
                      bn=BN_FLAG, is_training=is_training,
                      scope='conv1', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 64, [1,1],
                      padding='VALID', stride=[1,1],
                      bn=BN_FLAG, is_training=is_training,
                      scope='conv2', bn_decay=bn_decay)
  
  print("[1] net", net)
  with tf.variable_scope('transform_net2') as sc:
      transform = feature_transform_net(net, is_training, bn_decay, K=64)
  end_points['transform'] = transform
  net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
  point_feat = tf.expand_dims(net_transformed, [2])

  #save transform matrix
  end_points['transform_l1']= transform
  end_points['transform_l2']= transform*0
  end_points['transform_l3']= transform*0
  
  print("[2] point_feat", point_feat)
  
  net = tf_util.conv2d(point_feat, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv3', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 1024, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)  
  
 
  print("num_points", num_points)
  global_feat = tf_util.max_pool2d(net, [num_points,1],
                                    padding='VALID', scope='maxpool')
  print("global_feat", global_feat)

  global_feat_expand = tf.tile(global_feat, [1, num_points*1, 1, 1])
  #concat_feat = tf.concat(3, [point_feat, global_feat_expand])
  concat_feat = tf.concat( (point_feat, global_feat_expand), axis =3 )
  print("concat_feat", concat_feat)

  net = tf_util.conv2d(concat_feat, 512, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv6', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 256, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv7', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv8', bn_decay=bn_decay)
  net_last = tf_util.conv2d(net, 128, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv9', bn_decay=bn_decay)

  net = tf_util.conv2d(net_last, 2, [1,1],
                        padding='VALID', stride=[1,1], activation_fn=None,
                         is_training=is_training, scope='conv10')
  
  print("net", net)

  
  net = tf.squeeze(net, [2]) # BxNxC

  print("net", net)
  
  net_last =  tf.reshape(net_last, (batch_size, seq_length, original_num_points,net_last.shape[-1] ))
  print("net_last", net_last) # (batch, frames, npoints, dims)
  
  end_points['last_d_feat'] = net_last 
  end_points['points'] = point_cloud 
  
  predicted_labels = tf.reshape(net, (batch_size,seq_length,original_num_points, 2) )
  print("predicted_labels", predicted_labels)
  return predicted_labels, end_points
       

def get_loss(predicted_labels, ground_truth_labels, context_frames):

  """ Calculate loss 
   inputs: predicted labels : 
   	   ground_truth_labels: (batch, seq_length, num_points, 1)
   	   predicted_labels : (batch,seq_length, num_points, 2
  """
  batch_size = ground_truth_labels.get_shape()[0].value
  seq_length = ground_truth_labels.get_shape()[1].value
  num_points = ground_truth_labels.get_shape()[2].value
  
  # Convert labels to a list - This can be improved but it works
  ground_truth_labels = tf.split(value = ground_truth_labels , num_or_size_splits=seq_length, axis=1)
  ground_truth_labels = [tf.squeeze(input=label, axis=[1]) for label in ground_truth_labels] 
  
  predicted_labels = tf.split(value = predicted_labels , num_or_size_splits=seq_length, axis=1) 
  predicted_labels = [tf.squeeze(input=label, axis=[1]) for label in predicted_labels]  
  
  sequence_loss = 0
  #Calculate loss frame by frame
  for frame in range(context_frames,seq_length ):
    logits = predicted_labels[frame]
    labels = ground_truth_labels[frame]

    logits = tf.reshape(logits, [batch_size * num_points , 2])
    labels = tf.reshape(labels, [batch_size*num_points ,])
    labels = tf.cast(labels, tf.int32)

    frame_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    
    frame_loss = tf.reduce_mean(frame_loss)
    sequence_loss = sequence_loss + frame_loss  	
  	
  sequence_loss = sequence_loss/(seq_length)
  return sequence_loss 
  

def get_balanced_loss(predicted_labels, ground_truth_labels, context_frames):

  """ Calculate balanced loss 
   inputs: predicted labels : 
   	   ground_truth_labels: (batch, seq_length, num_points, 1)
   	   predicted_labels : (batch,seq_length, num_points, 2
  """
  batch_size = ground_truth_labels.get_shape()[0].value
  seq_length = ground_truth_labels.get_shape()[1].value
  num_points = ground_truth_labels.get_shape()[2].value
  
  # Convert labels to a list - This can be improved but it works
  ground_truth_labels = tf.split(value = ground_truth_labels , num_or_size_splits=seq_length, axis=1)
  ground_truth_labels = [tf.squeeze(input=label, axis=[1]) for label in ground_truth_labels] 
  
  predicted_labels = tf.split(value = predicted_labels , num_or_size_splits=seq_length, axis=1) 
  predicted_labels = [tf.squeeze(input=label, axis=[1]) for label in predicted_labels]  
  
  sequence_loss = 0
  #Calculate loss frame by frame
  for frame in range(context_frames,seq_length ):
    logits = predicted_labels[frame]
    labels = ground_truth_labels[frame]

    logits = tf.reshape(logits, [batch_size * num_points , 2])
    labels = tf.reshape(labels, [batch_size*num_points ,])
    labels = tf.cast(labels, tf.int32)

    #print("--- Normal Classification ---")
    frame_loss =0
    frame_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    frame_loss = tf.cast(frame_loss, tf.float32)
    
    labels = tf.cast(labels, tf.float32)
    print("---Weighted Classification ---")
    mask_0 = tf.where(labels < 0.5, tf.ones_like(labels), tf.zeros_like(labels))  # 0 -Noise points
    mask_1 = tf.where(labels > 0.5, tf.ones_like(labels), tf.zeros_like(labels))  # 1 -Clean points
    mask_0 = tf.cast(mask_0, tf.float32)
    mask_1 = tf.cast(mask_1, tf.float32)
    
    print("mask_0", mask_0)
    print("frame_loss", frame_loss)
    frame_loss_0 = frame_loss * mask_0 * 0.7 # worth less
    frame_loss_1 = frame_loss * mask_1 * 1.3 # worth more
    frame_loss = frame_loss_0 + frame_loss_1

    frame_loss = tf.reduce_mean(frame_loss)
    sequence_loss = sequence_loss + frame_loss  	
  	
  sequence_loss = sequence_loss/(seq_length)
  return sequence_loss  
