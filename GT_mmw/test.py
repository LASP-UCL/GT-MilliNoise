import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from datasets.bari_test_data import MMW as Dataset_mmW_eval

import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import roc_auc_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


parser = argparse.ArgumentParser()
""" --  Training  hyperparameters --- """
parser.add_argument('--gpu', default='0', help='Select GPU to run code [default: 0]')
parser.add_argument('--data-dir', default='/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset', help='Dataset directory')
parser.add_argument('--batch-size', type=int, default=409, help='Batch Size during training [default: 16]')
parser.add_argument('--data-split', type=int, default=11, help='Select the train/test/ data split  [default: 0,1,2]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--learning-rate', type=float, default=0.0, help='Learning rate [default: 1e-4]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients[default: 5.0 or 1e10 no clip].')
parser.add_argument('--restore-training', type= int , default = 0 , help='restore-training [default: 0=False 1= True]')
parser.add_argument('--velocity_intensity', type= int , default = 0, help='[default:0 #context frames')



""" --  Save  hyperparameters --- """
parser.add_argument('--save-cpk', type=int, default=2, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--save-summary', type=int, default=2, help='Iterations to update summary [default: 20]')
parser.add_argument('--save-iters', type=int, default=2, help='Iterations to save examples [default: 100000]')

""" --  Model  hyperparameters --- """
parser.add_argument('--model', type=str, default='GraphRNN_cls', help='Simple model or advanced model [default: advanced]')
parser.add_argument('--graph_module', type=str, default='Simple_GraphRNNCell', help='Simple model or advanced model [default: Simple_GraphRNNCell]')
parser.add_argument('--out_channels', type=int, default=32, help='Dimension of feat [default: 64]')
parser.add_argument('--num-samples', type=int, default=4, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=30, help='Length of sequence [default: 30]')
parser.add_argument('--num-points', type=int, default=200, help='Number of points [default: 1000]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--log-dir', default='/scratch/uceepdg/Millinoise_outputs', help='Log dir [default: outputs/mmw]')

parser.add_argument('--version', default='v0', help='Model version')
parser.add_argument('--down-points1', type= float , default = 2 , help='[default:2 #points layer 1')
parser.add_argument('--down-points2', type= float , default = 2*2 , help='[default:2 #points layer 2')
parser.add_argument('--down-points3', type= float , default = 2*2*2, help='[default:2 #points layer 3')
parser.add_argument('--context-frames', type= int , default = 1, help='[default:1 #contex framres')
parser.add_argument('--manual-restore', type= int , default = 0, help='[default:1 #contex framres')
parser.add_argument('--restore-ckpt', type= int , default = 76, help='[default:1 #contex framres')


""" --  Additional  hyperparameters --- """
parser.add_argument('--bn_flag', type=int, default=1, help='Do batch normalization[ 1- Yes, 0-No]')
parser.add_argument('--weight_decay', type=int, default=0, help='Do Weight Decay normalization [ 1- Yes, 0-No]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]') 
parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate in second last layer[default: 0.0]') 

print("\n ==== EVALUATION AND DEBUG OF THE MMW POINT CLODU DENOISING (BARI DATASET) ====== \n")

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

# Define the GPU to run the code
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

# Flags
BATCH_SIZE = args.batch_size
NUM_POINTS = args.num_points
SEQ_LENGTH = args.seq_length

FLAG_VEL_INT  = args.velocity_intensity
BASE_LEARNING_RATE = args.learning_rate
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

BN_FLAG = True if args.bn_flag == 1 else False
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Analyze Flags
id_seq_to_visualize = [40, 53, 200, 214, 235,500,600,700]
TNET_FLAG = False
DO_SAMPLED_ACC_FLAG = False
TC_MODULE_FLAG = False
NUM_SAMPLED_POINTS = 10 * SEQ_LENGTH
if (SEQ_LENGTH == 1 ): DO_SAMPLED_ACC_FLAG = False
ATTETION_VISU=  False


"""  Setup Directorys """
MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', args.model+'.py')
print("MODEL_FILE", MODEL_FILE)
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_DIR = os.path.join(LOG_DIR, args.model + '_'+ args.version)
print("LOG_DIR", LOG_DIR)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
BEST_MODEL_DIR = os.path.join(LOG_DIR, 'best_model')
if not os.path.exists(BEST_MODEL_DIR): os.mkdir(BEST_MODEL_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp evaluate.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_debug.txt'), 'a')
#write input arguments organize
LOG_FOUT.write("\n ========  Training Log ========  \n")
LOG_FOUT.write(":Input Arguments\n")
for var in args.__dict__:
	LOG_FOUT.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )


"""  Load Dataset """
#Load Test Dataset
test_dataset = Dataset_mmW_eval(root=args.data_dir,
                        seq_length= args.seq_length,
                        num_points=args.num_points,
                        split_number = args.data_split,
                        train= False)

def get_classification_metrics(predicted_labels, ground_truth_labels, batch_size, seq_length,num_points,context_frames ): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy, F1 and Recall in numpy
  In : predicted_labels : (batch , seq_length, num_points, 2)
       ground_truth_labels : (batch , seq_length, num_points, 1) 
  Out: accuracy, f1_score, recall (1)
  """
  
  # cut context frames 
  predicted_labels = predicted_labels[:,:,:,:]
  ground_truth_labels = ground_truth_labels[:,:,:,:]
  
  predicted_labels = np.argmax(predicted_labels, axis=3)
  predicted_labels = np.expand_dims(predicted_labels, axis=3)
  correct = np.equal(predicted_labels,ground_truth_labels)
  accuracy = np.sum(correct.astype(int)) /( int(batch_size)  *  int(seq_length) *int(num_points) )

  true_positives = np.sum((predicted_labels == 1) & (ground_truth_labels == 1))
  false_positives = np.sum((predicted_labels == 1) & (ground_truth_labels == 0))
  true_negatives = np.sum((predicted_labels == 0) & (ground_truth_labels == 0))
  false_negatives = np.sum((predicted_labels == 0) & (ground_truth_labels == 1))

  return (accuracy, true_positives, false_positives, true_negatives, false_negatives)

def get_acurracy_tensor(predicted_labels, ground_truth_labels, batch_size, seq_length,num_points, context_frames ): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy
  In : predicted_labels : Tensor (batch , seq_length, num_points, 2)
       ground_truth_labels : Tensor(batch , seq_length, num_points, 1) 
  Out: accuracy (1)
  """
  predicted_labels = predicted_labels[:,context_frames:, :, :]
  ground_truth_labels = ground_truth_labels[:,context_frames:, :, :]
  predicted_labels = tf.argmax(predicted_labels, axis =3)
  predicted_labels = tf.expand_dims(predicted_labels, axis = 3)
  ground_truth_labels = tf.cast(ground_truth_labels, tf.int32)
  predicted_labels = tf.cast(predicted_labels, tf.int32)
  correct = tf.equal( predicted_labels ,ground_truth_labels ) 
  accuracy = tf.reduce_sum( (tf.cast(correct,tf.float32) ) )/( batch_size* (seq_length)  *num_points )    

  return accuracy

def print_weights(sess, params, layer_nr):
    """ Visualize weights in terminal """
    layers = np.array(params)
    layers = layers[layer_nr]
    W = layers#[layer_nr]
    print("W", W)
    print("Layer[",layer_nr, "]", W, "\n")    

def farthest_point_sampling(point_cloud, num_samples):
  
  sampled_point_cloud = np.zeros((num_samples, 3))
   
  # Select the first point randomly
  first_point_index = np.random.randint(point_cloud.shape[0])
  sampled_point_cloud[0] = point_cloud[first_point_index]
  
  indices = [first_point_index]

  # Compute distance to all other points
  distances = np.linalg.norm(point_cloud - sampled_point_cloud[0], axis=1)

  # Choose the remaining points using farthest point sampling
  for i in range(1, num_samples):
      farthest_point_index = np.argmax(distances)
      indices.append(farthest_point_index)    
      sampled_point_cloud[i] = point_cloud[farthest_point_index]
      new_distances = np.linalg.norm(point_cloud - sampled_point_cloud[i], axis=1)
      distances = np.minimum(distances, new_distances)

  return indices,  
      
      
def random_sampling(point_cloud, num_samples):
    indices = np.random.choice(point_cloud.shape[0], num_samples, replace=False)
    return indices
  
def get_batch(dataset, batch_size):
    """ load from the dataset at random """
    batch_data = []
    for i in range(batch_size):
        sample = dataset[0]
        batch_data.append(sample)
    return np.stack(batch_data, axis=0)

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay
    
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate,BASE_LEARNING_RATE) # CLIP THE LEARNING RATE!
    return learning_rate   
    
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def normalize_pca_to_color(l0_pca):
  """
  Normalize pca to color
  Input: PCA of feat (1000,3)
  Output: Normaize RGB (1000,3)
  """
  r= l0_pca[:,0]
  g= l0_pca[:,1]
  b= l0_pca[:,2]
  rgb = np.reshape(l0_pca, ( l0_pca.shape[0]*3 , 1) )
  r = np.reshape(r, (r.shape[0],1) )
  g = np.reshape(g, (g.shape[0],1) )
  b = np.reshape(b, (b.shape[0],1) )
  

  # Normalize all the colors togeder
  r = (r-min(rgb))/(max(rgb)-min(rgb))
  g = (g-min(rgb))/(max(rgb)-min(rgb))
  b = (b-min(rgb))/(max(rgb)-min(rgb))

  color = np.concatenate( (r, g,b ), axis =1)

  return (color)
    

def evaluate():
  with tf.Graph().as_default():
    
    is_training_pl = tf.placeholder(tf.bool, shape=())
    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, SEQ_LENGTH, NUM_POINTS)

  
    print("pointclouds_pl", pointclouds_pl)
    print("labels_pl", labels_pl)
    print("is_training_pl:", is_training_pl)

    batch = tf.Variable(0)
    if args.weight_decay == 0: bn_decay = 0.0
    else: bn_decay = get_bn_decay(batch) 
    
    tf.summary.scalar('bn_decay', bn_decay)
        
    model_params = {'context_frames': int(int(args.context_frames)),
  	   	    'num_samples': int(args.num_samples),
  	   	    'graph_module': args.graph_module,
  	   	    'BN_FLAG':BN_FLAG, # decides if there is batch normalization
  	   	    'bn_decay':bn_decay,
  	   	    'out_channels':args.out_channels,
  	   	    'drop_rate': args.drop_rate,
  	   	    'sampled_points_down1':int(args.down_points1),
  	   	    'sampled_points_down2':int(args.down_points2),
  	   	    'sampled_points_down3':int(args.down_points3)}

    pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, model_params)
    

    loss = MODEL.get_loss(pred, labels_pl, context_frames = model_params['context_frames'])
    tf.summary.scalar('loss', loss)

    # Calculate accuracy tensor
    accuracy = get_acurracy_tensor(pred, labels_pl,BATCH_SIZE,SEQ_LENGTH, NUM_POINTS, context_frames = model_params['context_frames'])
    tf.summary.scalar('accuracy', accuracy)

    # Get training operator
    learning_rate = get_learning_rate(batch) 
    #learning_rate = BASE_LEARNING_RATE
    tf.summary.scalar('learning_rate', learning_rate)
  
    params = tf.trainable_variables()
    #gradients = tf.gradients(loss, params)
    #clipped_gradients, norm = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
    #train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=batch)    

    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours = 5)
    
    print(" Trainable paramenters: ")
    for layer in params:
    	print(layer)
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session( config =config)
    
    # Add summary writers
    merged = tf.summary.merge_all()
    #train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train_debug'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test_debug'))

    # Restore Session
    if (args.manual_restore == 0):
      print("LOG_DIR", LOG_DIR)
      checkpoint_path_automatic = tf.train.latest_checkpoint(LOG_DIR)
      ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
      restore_checkpoint_path = checkpoint_path_automatic
      ckpt_number=ckpt_number[11:]
      ckpt_number=int(ckpt_number)
      print ("\n** Restore from checkpoint ***: ", restore_checkpoint_path)
      
    
    if (args.manual_restore == 1): # Select automatically
      checkpoint_path_automatic = tf.train.latest_checkpoint(LOG_DIR)
      ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
      ckpt_number=ckpt_number[11:]
      restore_checkpoint_path =checkpoint_path_automatic.replace(ckpt_number, str(args.restore_ckpt) )
      ckpt_number= args.restore_ckpt
      print ("\n** Restore from checkpoint ***: ", restore_checkpoint_path)
    
    if (args.manual_restore == 2): # Best Validation model
      restore_checkpoint_path = os.path.join(LOG_DIR, "best_model.ckpt")
      print ("\n** Restore from checkpoint ***: ", restore_checkpoint_path)
      ckpt_number= 1
      

    saver.restore(sess, restore_checkpoint_path)
    # change random seed
    np.random.seed(ckpt_number)
    tf.set_random_seed(ckpt_number)

    ops = {'pointclouds_pl': pointclouds_pl,
  	   'labels_pl': labels_pl,
  	   'is_training_pl': is_training_pl,
  	   'pred': pred,
  	   'loss': loss,
  	   'acc':accuracy,
  	   'params': params,
  	   'merged': merged,
       'end_points': end_points,
  	   'step': batch}

    
    for epoch in range(ckpt_number, ckpt_number + 1):
      
      sys.stdout.flush() 
      #Evaluate
      print(" **  Evalutate Test Data ** ")
      mean_loss, mean_accuracy   = eval_one_epoch(sess, ops, test_writer, epoch)   
        	
        
""" ------------------   """
def eval_one_epoch(sess,ops,test_writer, epoch):
    
    """ Eval all sequences of test dataset """
    is_training = False
    
    nr_tests = len(test_dataset) 
    num_batches = nr_tests // BATCH_SIZE
    point_loss_list = []

      
    x = [i for i in range(1, nr_tests+1) if nr_tests % i == 0]
    if (BATCH_SIZE not in x): 
      print("[NOT LOADING ALL TEST DATA] - To test the full test data: select a batch size:", x)

    total_accuracy =0
    total_loss = 0
    Tp =0 #true positives total
    Tn =0
    Fp =0
    Fn =0
    total_auc_roc = 0
    total_sampled_accuracy = 0
    skip_count = 0
        
    for batch_idx in tqdm ( range(num_batches) ):
      start_idx = batch_idx * BATCH_SIZE
      end_idx = (batch_idx+1) * BATCH_SIZE
      cur_batch_size = end_idx - start_idx
      input_point_clouds =[]
      input_labels =[]
    
      for idx  in range(start_idx,end_idx): #sequences to be tested
        test_seq = test_dataset[idx]   
        test_seq =np.array(test_seq)
        #print("test_seq", test_seq.shape)
        if (FLAG_VEL_INT == 0):
          point_clouds = test_seq[:,:,0:3]
          labels = test_seq[:,:,3:4]
        if (FLAG_VEL_INT == 1):
          point_clouds = test_seq[:,:,:]
          labels = test_seq[:,:,5:6]        
        input_point_clouds.append(point_clouds)
        input_labels.append(labels)
 
    
      
      input_point_clouds = np.array(input_point_clouds)
      input_labels = np.array(input_labels)
      
        
      
      # Send to model to be evaluated
      feed_dict = {ops['pointclouds_pl']: input_point_clouds, ops['labels_pl']: input_labels, ops['is_training_pl']: is_training}
      pred, summary, step, loss, accuracy, params, end_points =  sess.run([ops['pred'], ops['merged'], ops['step'], ops['loss'], ops['acc'], ops['params'] , ops['end_points']], feed_dict=feed_dict) 
      #test_writer.add_summary(summary, step)  # Do not write to tensroboard      
      total_accuracy = total_accuracy + accuracy
      total_loss = total_loss + loss
      accuracy, true_positives, false_positives, true_negatives,false_negatives = get_classification_metrics(pred, input_labels, args.batch_size, args.seq_length,args.num_points, args.context_frames )
      Tp = Tp + (true_positives) 
      Fp = Fp + (false_positives)
      Tn = Tn + (true_negatives)
      Fn = Fn + (false_negatives)

      # Calculate AUC-ROC
      aux_pred = pred[0]
      y_pred_proba = np.reshape(aux_pred, (aux_pred.shape[0]*aux_pred.shape[1], 2))
      y_pred_proba = y_pred_proba[:,1]# probaibility of classe 1
      y_true = input_labels
      y_true = y_true[0]
      y_true = np.reshape(y_true, (y_true.shape[0]*y_true.shape[1], 1))
      if len(np.unique(y_true)) == 1:
        #print("Only one class present in y_true [skip AUC-ROC]")
        skip_count =  skip_count +1 
      else: 
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        total_auc_roc = total_auc_roc +auc_roc
        
    
      
      
      """ Visualize weights in terminal """
      #print_weights(sess, params, 82)
      
      """ Data Analyze """
      DATA_DIR = '/scratch/uceepdg/Bari_Denoising_Analyze/'+ args.model + '_' + args.version +'/'
      if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
      FPS_IDX_PATH = '/scratch/uceepdg/Fps_idxs_'+str(SEQ_LENGTH)+'_frames/'
      if not os.path.exists(FPS_IDX_PATH): os.mkdir(FPS_IDX_PATH)


      # Sampled evaluation of the point cloud
      if DO_SAMPLED_ACC_FLAG == True :
        # This is very heavy since the farthest_point_sampling is not optimzed
        pc_to_sample = input_point_clouds[0]
        pc_to_sample= np.reshape(pc_to_sample, (SEQ_LENGTH*NUM_POINTS,3  ) )
        
        #Save indices to file
        indices = farthest_point_sampling(pc_to_sample,NUM_SAMPLED_POINTS  )
        print("idx", idx)
        indices = np.array(indices)
        indices = indices[0]
        np.save( FPS_IDX_PATH +'indices_' + str(idx) +'.npy', indices, )
        
        #load from file
        loaded_indices = np.load(FPS_IDX_PATH +'indices_' + str(idx) +'.npy')

        aux_pred = np.reshape(pred, (SEQ_LENGTH*NUM_POINTS,2  ) )
        sampled_pred = aux_pred[loaded_indices, :]
        aux_input_labels = input_labels[0]
        aux_input_labels = np.reshape(aux_input_labels, (SEQ_LENGTH*NUM_POINTS,1 ) )
        sampled_true_labels = aux_input_labels[loaded_indices, :]
        sampled_true_labels = np.reshape(sampled_true_labels, (1,SEQ_LENGTH,int(NUM_SAMPLED_POINTS/SEQ_LENGTH),1 ) )
        sampled_pred = np.reshape(sampled_pred, (1,SEQ_LENGTH,int(NUM_SAMPLED_POINTS/SEQ_LENGTH),2 ) )

        sampled_accuracy, sampled_true_positives, sampled_false_positives, sampled_true_negatives, sampled_false_negatives = get_classification_metrics(sampled_pred, sampled_true_labels, BATCH_SIZE, SEQ_LENGTH,NUM_SAMPLED_POINTS/SEQ_LENGTH,context_frames=0) 
        total_sampled_accuracy = total_sampled_accuracy+ sampled_accuracy
      
      if( idx%1000 ==0 and idx >0):
        print("accuracy:", total_accuracy/(idx+1))
        print("sampled_accuracy:", total_sampled_accuracy/(idx+1) )

      """  Save the data in folder """
      if( idx in id_seq_to_visualize):
        tosave_input_pc = input_point_clouds[0]
        tosave_input_labels = input_labels[0]     
        tosave_pred = pred[0]
        tosave_last_d_feat = end_points['last_d_feat']
        tosave_last_d_feat = tosave_last_d_feat[0]
           
      
      
        np.save(DATA_DIR + 'point_cloud_seq_' + str(idx) +'.npy', tosave_input_pc)
        np.save(DATA_DIR + 'prediction_seq_' + str(idx) +'.npy', tosave_pred)
        np.save(DATA_DIR + 'last_d_feat_seq_' + str(idx) +'.npy', tosave_last_d_feat)
        np.save(DATA_DIR + 'labels_' + str(idx) +'.npy', tosave_input_labels)

    mean_loss = total_loss/ num_batches
    mean_accuracy = total_accuracy/ num_batches
    mean_sampled_accuracy = total_sampled_accuracy/ num_batches
    precision = Tp / ( Tp+Fp)
    recall = Tp/(Tp+Fn)
    f1_score =2 * ( (precision * recall)/(precision+recall) )
    mean_auc_roc = total_auc_roc/(num_batches-skip_count)
    
    print('**** EVAL: %03d **** %s_%s' % (epoch,  str(args.model), str(args.version) ) )
    print("[Test] Loss   %f\t  Accuracy: %f\t"%( mean_loss, mean_accuracy) )
    print("Precision: ", precision, "\nRecall: ", recall, "\nF1 Score:", f1_score)
    print("AUC-ROC: ", mean_auc_roc)
    print("Sampled Accuracy: ", mean_sampled_accuracy)
    print(' -- ')  

    # Define the confusion matrix data
    confusion_data = [[Tp, Fp], [Fn, Tn]]
    # Plot the confusion matrix
    labels = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_data, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, fmt='g')
    fig.savefig(DATA_DIR+"/confusion_matrix" + ".png")

    
    
    # Write to File
    #log_string('****  %03d ****' % (epoch))
    log_string('%03d  eval mean loss, accuracy: %f \t  %f \t' % (epoch, mean_loss , mean_accuracy))
    log_string('%03d sampled accuracy: %f \t AUC-ROC  %f \t' % (epoch, mean_sampled_accuracy, mean_auc_roc ))
    if not np.isnan(precision) and not np.isnan(recall) and not np.isnan(f1_score):
      log_string('Precision %f Recall, F1 Score: %f \t  %f \t ]' % (precision, recall , f1_score))
    
    
    return mean_loss, mean_accuracy        
      
                
                  
if __name__ == "__main__":
  
    evaluate()
    LOG_FOUT.close()        




