import os
import sys
import tensorflow as tf
import numpy as np
"""
Implement a Basic GNN Architecture 

"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))

from pointnet2_color_feat_states import *
import graph_rnn_modules as modules
import tf_util
from transform_nets import input_transform_net, feature_transform_net
from tf_grouping import query_ball_point, group_point, knn_point, knn_feat

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
  out_channels = model_params['out_channels'] #use?
  drop_rate = model_params['drop_rate']
  graph_module_name = model_params['graph_module']
  end_points = {}
  
  #Stacked Frames processing
  context_frames = 0
  hop0 = 1
  hop1 = 2
  hop2 = 8
  hop3 = 12


  print("[Load Module]: ",graph_module_name) # GNN
  print("point_cloud.shape", point_cloud.shape)
  
  original_num_points = num_points
  num_points = original_num_points * seq_length
  
  # Add relative time-stamp to to point cloud
  timestep_tensor = tf.zeros( (batch_size,1,original_num_points,1) )
  for f in range(1, seq_length):
    frame_tensor = tf.ones( (batch_size,1,original_num_points,1) ) * f
    timestep_tensor = tf.concat( (timestep_tensor, frame_tensor) , axis = 1 )
  
  point_cloud = tf.reshape(point_cloud, (batch_size, seq_length * original_num_points, 3) )
  timestep_tensor = tf.reshape(timestep_tensor, (batch_size,seq_length *original_num_points, 1) )
  
  """ -----  Layer 0    -----  """
  l0_xyz = point_cloud
  l0_feats = timestep_tensor
  l0_feats = tf.concat( (l0_xyz,l0_feats) ,axis = 2)
  
  # Create adjacent matrix on cordinate space
  l0_adj_matrix = tf_util.pairwise_distance(l0_xyz)
  l0_nn_idx = tf_util.knn(l0_adj_matrix, k= int(num_samples * hop0) )   
  # Group Points
  l0_xyx_grouped = group_point(l0_xyz, l0_nn_idx) 
  l0_feats_grouped = group_point(l0_feats, l0_nn_idx)  
  # Calculate Edge feature
  l0_xyz_expanded = tf.expand_dims(l0_xyz, 2)
  l0_displacement = l0_xyx_grouped - l0_xyz_expanded  
  l0_feats_expanded = tf.expand_dims(l0_feats, 2)
  l0_feats_expanded = tf.tile(l0_feats_expanded, [1, 1, int(num_samples), 1])   
  l0_xyz_expanded = tf.tile(l0_xyz_expanded, [1, 1, int(num_samples), 1]) 
  edge_feature = tf.concat([l0_feats_expanded,l0_feats_grouped, l0_displacement], axis=3)   


  
  # GNN layer 1
  with tf.variable_scope('GNN_0') as sc:
    l0_feats = tf_util.conv2d(edge_feature, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l0_feats = tf_util.conv2d(l0_feats, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)
    l0_feats = tf.reduce_max(l0_feats, axis=[2], keepdims=False)

  """ -----  Layer 1    -----  """
  l1_xyz = l0_xyz # original point cloud cordinates
  l1_feats = l0_feats 


  # Create adjacent matrix on new cordinate spa ce
  l1_adj_matrix = tf_util.pairwise_distance(l1_xyz)
  l1_nn_idx = tf_util.knn(l1_adj_matrix, k= num_samples*hop1) 
  print("l1_nn_idx", l1_nn_idx)
  l1_nn_idx = l1_nn_idx[:,:,num_samples*(hop1-1):] # 2-hop neighborhood
  print("l1_nn_idx", l1_nn_idx)
  # Group Points
  l1_xyx_grouped = group_point(l1_xyz, l1_nn_idx) 
  l1_feat_grouped = group_point(l1_feats, l1_nn_idx)  
  # Calculate displacements
  l1_xyz_expanded = tf.expand_dims(l1_xyz, 2)
  l1_feat_expanded = tf.expand_dims(l1_feats, 2)
  l1_displacement = l1_xyx_grouped - l1_xyz_expanded  
  l1_xyz_expanded = tf.tile(l1_xyz_expanded, [1, 1, num_samples, 1]) 
  l1_feat_expanded = tf.tile(l1_feat_expanded, [1, 1, num_samples, 1]) 
  
  # Concatenate Message passing
  edge_feature = tf.concat([l1_feat_expanded, l1_feat_grouped, l1_displacement], axis=3)   
  
  # MLP message -passing
  with tf.variable_scope('GNN_1') as sc:
    l1_feats = tf_util.conv2d(edge_feature, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l1_feats = tf_util.conv2d(l1_feats, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)
    l1_feats = tf.reduce_max(l1_feats, axis=[2], keepdims=False)
    
  print("l1_feats", l1_feats)
  
  """ -----  Layer 2    -----  """
  l2_xyz = l1_xyz
  l2_feats = l1_feats


  # Create adjacent matrix on cordinate space
  l2_adj_matrix = tf_util.pairwise_distance(l2_xyz)
  l2_nn_idx = tf_util.knn(l2_adj_matrix, k= num_samples * hop2) 
  l2_nn_idx = l2_nn_idx[:,:, num_samples*(hop2-1):] # 2-hop neighborhood
  # Group Points
  l2_xyx_grouped = group_point(l2_xyz, l2_nn_idx)  
  l2_feats_grouped = group_point(l2_feats, l2_nn_idx)  
  # Calculate displacements
  l2_xyz_expanded = tf.expand_dims(l2_xyz, 2)
  l2_feat_expanded = tf.expand_dims(l2_feats, 2)
  l2_displacement = l2_xyx_grouped - l2_xyz_expanded   
  l2_xyz_expanded = tf.tile(l2_xyz_expanded, [1, 1, num_samples, 1]) 
  l2_feat_expanded = tf.tile(l2_feat_expanded, [1, 1, num_samples, 1]) 
  # Concatenate Message passing
  edge_feature = tf.concat([l2_feat_expanded, l2_feats_grouped, l2_displacement], axis=3)     
  
  # MLP message -passing
  with tf.variable_scope('GNN_2') as sc:
    l2_feats = tf_util.conv2d(edge_feature, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l2_feats = tf_util.conv2d(l2_feats, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)
    
  l2_feats = tf.reduce_max(l2_feats, axis=[2], keepdims=False)
  
  """ -----  Layer 3    -----  """
  l3_xyz = l2_xyz
  l3_feats = l2_feats

  # Create adjacent matrix on cordinate space
  l3_adj_matrix = tf_util.pairwise_distance(l3_xyz)
  l3_nn_idx = tf_util.knn(l3_adj_matrix, k= num_samples *hop3) 
  l3_nn_idx = l3_nn_idx[:,:, num_samples*(hop3-1):] # n-hop neighborhood
  # Group Points
  l3_xyx_grouped = group_point(l3_xyz, l3_nn_idx) 
  l3_feats_grouped =   group_point(l3_feats, l3_nn_idx) 
  # Calculate displacements
  l3_xyz_expanded = tf.expand_dims(l3_xyz, 2)
  l3_feat_expanded = tf.expand_dims(l3_feats, 2)
  l3_displacement = l3_xyx_grouped - l3_xyz_expanded   
  l3_xyz_expanded = tf.tile(l3_xyz_expanded, [1, 1, num_samples, 1]) 
  l3_feat_expanded = tf.tile(l3_feat_expanded, [1, 1, num_samples, 1]) 
  
  # Concatenate Message passing
  edge_feature = tf.concat([l3_feat_expanded, l3_feats_grouped, l3_displacement ], axis=3)   
  print("edge_feature", edge_feature)
  # MLP message -passing
  with tf.variable_scope('GNN_3') as sc:
    l3_feats = tf_util.conv2d(edge_feature, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l3_feats = tf_util.conv2d(l3_feats, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)

    l3_feats = tf.reduce_max(l3_feats, axis=[2], keepdims=False)
    
  print("l3_feats", l3_feats)
  
  # Global feature Representation 
  global_concatenation =  tf.concat( (l0_feats, l1_feats, l2_feats,l3_feats) , axis=2 )
  global_concatenation =  tf.expand_dims(global_concatenation, 2)
  
  """ --- Global Representation  ---"""
  #global_feat = tf_util.conv2d(global_concatenation, 256, [1,1], padding='VALID',stride=[1,1],  bn=BN_FLAG, is_training=is_training, scope='g_fc1', bn_decay=bn_decay)
  global_feat = tf_util.conv2d(global_concatenation, 512, [1,1], padding='VALID',stride=[1,1],  bn=BN_FLAG, is_training=is_training, scope='g_fc2', bn_decay=bn_decay)
  global_feat = tf_util.conv2d(global_feat, 1024, [1,1], padding='VALID',stride=[1,1],  bn=BN_FLAG, is_training=is_training, scope='g_fc3', bn_decay=bn_decay)
  global_feat = tf.reduce_max(global_feat, axis=[1], keepdims=False)
  
  #Concate global representation with full concatenation
  global_feat_repeat = tf.tile(global_feat, [1, global_concatenation.shape[1], 1])
  print("global_concatenation", global_concatenation)
  print("global_feat_repeat", global_feat_repeat)
  global_concatenation = tf.reshape(global_concatenation,(global_feat_repeat.shape[0],global_feat_repeat.shape[1], global_concatenation.shape[3] ) )
  print("global_concatenation", global_concatenation)
  concatenation = tf.concat( (global_concatenation, global_feat_repeat) , axis=2 )
  print("concatenation", concatenation)
  

  concatenation = tf_util.dropout(concatenation, keep_prob=0.75, is_training=is_training, scope='dp1')
  net = tf_util.conv1d(concatenation, 512, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc0', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.65, is_training=is_training, scope='dp2')
  net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc1', bn_decay=bn_decay)
  net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.60, is_training=is_training, scope='dp3')
  net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc3', bn_decay=bn_decay)
  net_last =  net
  predicted_labels = tf_util.conv1d(net, 2, 1, padding='VALID',activation_fn=None, bn=BN_FLAG, is_training=is_training, scope='fc4', bn_decay=bn_decay)
  
  #print("predicted_labels", predicted_labels)
  #print("net_last", net_last)
  net_last =  tf.reshape(net_last, (batch_size, seq_length, original_num_points, 64 ))
  predicted_labels = tf.reshape(predicted_labels, (batch_size,seq_length,original_num_points, 2) )
  end_points['last_d_feat'] = net_last 


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
  	
  sequence_loss = sequence_loss/(seq_length )
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
    frame_loss_0 = frame_loss * mask_0 * 0.65 # worth less
    frame_loss_1 = frame_loss * mask_1 * 1 # worth more
    frame_loss = frame_loss_0 + frame_loss_1

    frame_loss = tf.reduce_mean(frame_loss)
    sequence_loss = sequence_loss + frame_loss  	
  	
  sequence_loss = sequence_loss/(seq_length)
  return sequence_loss  

