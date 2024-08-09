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
  hop1 = 1
  hop2 = 1
  hop3 = 1
  
  temp_hop1 = 1
  temp_hop0 = 1

  original_num_points = num_points
  num_points = original_num_points * seq_length
  
  
  temporal_feat_list =[]
  l0_temp_feat_list = []
  

  print("[Load Module]: ",graph_module_name) # GNN
  print("point_cloud.shape", point_cloud.shape)
  print("original_num_points", original_num_points)

  
  # Add relative time-stamp to to point cloud
  timestep_tensor = tf.zeros( (batch_size,1,original_num_points,1) )
  for f in range(1, seq_length):
    frame_tensor = tf.ones( (batch_size,1,original_num_points,1) ) * f
    timestep_tensor = tf.concat( (timestep_tensor, frame_tensor) , axis = 1 )
  
  stacked_point_cloud = tf.reshape(point_cloud, (batch_size, seq_length * original_num_points, 3) )
  stacked_timestep_tensor = tf.reshape(timestep_tensor, (batch_size,seq_length *original_num_points, 1) )



  print("==== Geomerical  Module ===== ")
  # Lest stack frames now
  stacked_point_cloud = tf.reshape(point_cloud, (batch_size, seq_length * original_num_points, 3) )
  stacked_timestep_tensor = tf.reshape(timestep_tensor, (batch_size,seq_length *original_num_points, 1) )
  
  print("stacked_point_cloud", stacked_point_cloud)
  print("stacked_timestep_tensor", stacked_timestep_tensor)
  
  
    
  """ -----  Layer 0    -----  """
  l0_xyz = stacked_point_cloud
  l0_feats = tf.concat( (l0_xyz,stacked_timestep_tensor) ,axis = 2)
  stacked_l0_feats_input  = l0_feats
  
  # GNN layer 1
  with tf.variable_scope('Transformer_0') as sc:
    Q_l0 = tf_util.conv1d(l0_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'Q_l0', 
                              bn_decay=bn_decay)
    
    K_l0 = tf_util.conv1d(l0_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'K_l0', 
                              bn_decay=bn_decay)

    
    V_l0 = tf_util.conv1d(l0_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'V_l0', 
                              bn_decay=bn_decay)

  print("Q_l0",Q_l0)
  print("K_l0", K_l0)
  print("V_l0", V_l0)  
  
  attention_scores = tf.matmul(Q_l0, tf.transpose(K_l0, perm=[0, 2, 1]) )
  print("attention_scores", attention_scores)
  
  embedding_size  = 64
  attention_scores =attention_scores/ tf.sqrt(tf.cast(embedding_size, dtype=tf.float32))
  print("[1] attention_scores", attention_scores)
  
  
  l0_adj_matrix = tf_util.pairwise_distance(l0_xyz)
  

  attention_weights_l0 = tf.nn.softmax(attention_scores, axis=-1)
  print("attention_weights_l0", attention_weights_l0)
  
  
  
  
  
  print("----") 
  # Do a Message-Passing operation
  l0_nn_idx, l0_top_values = tf_util.knn_and_values(-attention_weights_l0, k= int(num_samples * hop1) )   
  print("l0_nn_idx", l0_nn_idx)
  # Group Points
  V_l0_grouped = group_point(V_l0, l0_nn_idx) 
  V_l0_expanded = tf.expand_dims(V_l0, 2)
  V_l0_expanded = tf.tile(V_l0_expanded, [1, 1, num_samples, 1])
   
  attended_Vl0 = tf.matmul(attention_weights_l0, V_l0)
  attended_Vl0_expanded =  tf.expand_dims(attended_Vl0, 2)
  attended_Vl0_expanded = tf.tile(attended_Vl0_expanded, [1, 1, num_samples, 1]) 
  print("V_l0_grouped", V_l0_grouped)
  print("V_l0_expanded", V_l0_expanded)
  
  print("l0_top_values", l0_top_values)
  attention_expanded = tf.expand_dims(l0_top_values, axis=-1)
  edge_feature = tf.concat([V_l0_expanded, attended_Vl0_expanded,V_l0_grouped, attention_expanded], axis =3)
    
  
  # GNN layer 0
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
  print("l0_feats", l0_feats)
  

  
  """ -----  Layer 1    -----  """
  l1_xyz = stacked_point_cloud
  l1_feats = tf.concat( (l1_xyz,stacked_l0_feats_input) ,axis = 2)
  
  # GNN layer 1
  with tf.variable_scope('Transformer_1') as sc:
    Q_l1 = tf_util.conv1d(l1_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'Q_l1', 
                              bn_decay=bn_decay)
    K_l1 = tf_util.conv1d(l1_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'K_l1', 
                              bn_decay=bn_decay)

    V_l1 = tf_util.conv1d(l1_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'V_l1', 
                              bn_decay=bn_decay)
    

  attention_scores = tf.matmul(Q_l1, tf.transpose(K_l1, perm=[0, 2, 1]) )
  embedding_size  = 64
  attention_scores =attention_scores/ tf.sqrt(tf.cast(embedding_size, dtype=tf.float32))

  attention_weights_l1 = tf.nn.softmax(attention_scores, axis=-1)
  
  # Do a Message-Passing operation
  l1_nn_idx, l1_top_values = tf_util.knn_and_values(-attention_weights_l1, k= int(num_samples * hop1) )   
  # Group Points
  V_l1_grouped = group_point(V_l1, l1_nn_idx) 
  V_l1_expanded = tf.expand_dims(V_l1, 2)
  V_l1_expanded = tf.tile(V_l1_expanded, [1, 1, num_samples, 1]) 
  attention_expanded = tf.expand_dims(l1_top_values, axis=-1)
  attended_Vl1 = tf.matmul(attention_weights_l1, V_l1)
  attended_Vl1_expanded =  tf.expand_dims(attended_Vl1, 2)
  attended_Vl1_expanded = tf.tile(attended_Vl1_expanded, [1, 1, num_samples, 1]) 
  
  edge_feature = tf.concat([V_l1_expanded,attended_Vl1_expanded, V_l1_grouped, attention_expanded], axis =3)
  
  # GNN layer 0
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
  l2_xyz = stacked_point_cloud
  l2_feats = tf.concat( (l2_xyz,stacked_l0_feats_input) ,axis = 2)

  

      
  # GNN layer 2
  with tf.variable_scope('Transformer_2') as sc:
    Q_l2 = tf_util.conv1d(l2_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'Q_l1', 
                              bn_decay=bn_decay)
    K_l2 = tf_util.conv1d(l2_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'K_l1', 
                              bn_decay=bn_decay)
    V_l2 = tf_util.conv1d(l2_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'V_l1', 
                              bn_decay=bn_decay)
    

  attention_scores = tf.matmul(Q_l2, tf.transpose(K_l2, perm=[0, 2, 1]) )
  embedding_size  = 64
  attention_scores =attention_scores/ tf.sqrt(tf.cast(embedding_size, dtype=tf.float32))


  attention_weights_l2 = tf.nn.softmax(attention_scores, axis=-1)
  
  
  # Do a Message-Passing operation
  l2_nn_idx, l2_top_values = tf_util.knn_and_values(-attention_weights_l2, k= int(num_samples * hop1) )   
  # Group Points
  V_l2_grouped = group_point(V_l2, l2_nn_idx) 
  V_l2_expanded = tf.expand_dims(V_l2, 2)
  V_l2_expanded = tf.tile(V_l2_expanded, [1, 1, num_samples, 1]) 
  attention_expanded = tf.expand_dims(l2_top_values, axis=-1)
  attended_Vl2 = tf.matmul(attention_weights_l2, V_l2)
  attended_Vl2_expanded =  tf.expand_dims(attended_Vl2, 2)
  attended_Vl2_expanded = tf.tile(attended_Vl2_expanded, [1, 1, num_samples, 1]) 
  
  
  edge_feature = tf.concat([V_l2_expanded,attended_Vl2_expanded, V_l2_grouped, attention_expanded], axis =3)
  
  # GNN layer 0
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
  print("l2_feats", l2_feats)
    
  
  """ -----  Layer 3    -----  """
  l3_xyz = stacked_point_cloud
  l3_feats = tf.concat( (l3_xyz,stacked_l0_feats_input) ,axis = 2)

    
  # GNN layer 3
  with tf.variable_scope('Transformer_3') as sc:
    Q_l3 = tf_util.conv1d(l3_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'Q_l1', 
                              bn_decay=bn_decay)
    
    K_l3 = tf_util.conv1d(l3_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'K_l1', 
                              bn_decay=bn_decay)
    V_l3 = tf_util.conv1d(l3_feats, 
                              64, 1, 
                              padding='VALID', 
                              bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'V_l1', 
                              bn_decay=bn_decay)

  attention_scores = tf.matmul(Q_l3, tf.transpose(K_l3, perm=[0, 2, 1]) )
  embedding_size  = 64
  attention_scores =attention_scores/ tf.sqrt(tf.cast(embedding_size, dtype=tf.float32))

  attention_scores = attention_scores 
  
  attention_weights_l3 = tf.nn.softmax(attention_scores, axis=-1)
  # Do a Message-Passing operation
  l3_nn_idx, l3_top_values = tf_util.knn_and_values(-attention_weights_l3, k= int(num_samples * hop1) )   
  # Group Points
  V_l3_grouped = group_point(V_l3, l3_nn_idx) 
  V_l3_expanded = tf.expand_dims(V_l3, 2)
  V_l3_expanded = tf.tile(V_l3_expanded, [1, 1, num_samples, 1]) 
  attention_expanded = tf.expand_dims(l3_top_values, axis=-1)
  attended_Vl3 = tf.matmul(attention_weights_l3, V_l3)
  attended_Vl3_expanded =  tf.expand_dims(attended_Vl3, 2)
  attended_Vl3_expanded = tf.tile(attended_Vl3_expanded, [1, 1, num_samples, 1]) 
  
  edge_feature = tf.concat([V_l3_expanded,attended_Vl3_expanded, V_l3_grouped, attention_expanded], axis =3)
  
  # GNN layer 0
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
  
  print("predicted_labels", predicted_labels)
  print("net_last", net_last)
  net_last =  tf.reshape(net_last, (batch_size, seq_length, original_num_points, 64 ))
  predicted_labels = tf.reshape(predicted_labels, (batch_size,seq_length,original_num_points, 2) )
  end_points['last_d_feat'] = net_last 
  
  
  #end_points['temp_feat'] = stacked_temporal_feat
  end_points['attention_weights_l0'] = attention_weights_l0
  end_points['attention_weights_l1'] = attention_weights_l1 
  end_points['attention_weights_l2'] = attention_weights_l2 
  end_points['attention_weights_l3'] = attention_weights_l3 
  
  
  
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

