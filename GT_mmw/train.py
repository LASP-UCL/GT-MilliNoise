import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
#from datasets.bari_train_data_mem_efficient import MMW as Dataset_mmW
from datasets.bari_train_data import MMW as Dataset_mmW
from datasets.bari_val_data import MMW as Dataset_mmW_val
import importlib
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))


parser = argparse.ArgumentParser()
""" --  Training  hyperparameters --- """
parser.add_argument('--gpu', default='0', help='Select GPU to run code [default: 0]')
parser.add_argument('--data-dir', default='/scratch/uceepdg/Labelled_mmW/', help='Dataset directory')
parser.add_argument('--batch-size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--data-split', type=int, default=13, help='Select the train/test/ data split  [default: 0,1,2]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate [default: 1e-4]')
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
parser.add_argument('--out_channels', type=int, default=64, help='Dimension of feat [default: 64]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=12, help='Length of sequence [default: 30]')
parser.add_argument('--num-points', type=int, default=200, help='Number of points [default: 1000]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--log-dir', default='/scratch/uceepdg/Millinoise_outputs', help='Log dir [default: outputs/mmw]')
#parser.add_argument('--log-dir', default='outputs', help='Log dir [default: outputs/mmw]')
parser.add_argument('--version', default='v0', help='Model version')
parser.add_argument('--down-points1', type= float , default = 1 , help='[default:2 #points layer 1')
parser.add_argument('--down-points2', type= float , default = 2 , help='[default:2 #points layer 2')
parser.add_argument('--down-points3', type= float , default = 4, help='[default:2 #points layer 3')
parser.add_argument('--context-frames', type= int , default = 0, help='[default:0 #context frames')

""" --  Additional  hyperparameters --- """
parser.add_argument('--bn_flag', type=int, default=1, help='Do batch normalization[ 1- Yes, 0-No]')
parser.add_argument('--balanced_loss', type=int, default=0, help='Do balanced loss [ 1- Yes, 0-No]')
parser.add_argument('--weight_decay', type=int, default=0, help='Do Weight Decay normalization [ 1- Yes, 0-No]')
parser.add_argument('--lr_scheduler', type=int, default=2, help='lr_scheduler [default: 0, 2-> keeps getting reduced]')
parser.add_argument('--decay_step', type=int, default=10000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.25, help='Decay rate for lr decay [default: 0.8]') 
parser.add_argument('--drop_rate', type=float, default=0.5, help='Dropout rate in second last layer[default: 0.0]') 
parser.add_argument('--regularizer_scale', type=float, default=0.00, help='Regulaizer value[default: 0.0- or 0.001]') 
parser.add_argument('--regularizer_alpha', type=float, default=0.0, help='Regulaizer value[default: 0.0- or 0.001]') 
print("\n ==== MMW POINT CLODU DENOISING (BARI DATASET) ====== \n")

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

# Define the GPU to run the code
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

# Flags
BATCH_SIZE = args.batch_size
NUM_POINTS = args.num_points
SEQ_LENGTH = args.seq_length
DATA_SPLIT = args.data_split
FLAG_VEL_INT  = args.velocity_intensity

BASE_LEARNING_RATE = args.learning_rate
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate

BN_FLAG = True if args.bn_flag == 1 else False
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

PATIENCE_LIMIT = 20 #-> also a good choice
LR_MIN = 1e-4
#Train forever
#PATIENCE_LIMIT = np.inf #-> also a good choice
#LR_MIN = 0.5e-3

"""  Setup Directorys """
MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', args.model+'.py')
print("MODEL_FILE", MODEL_FILE)
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_DIR = os.path.join(LOG_DIR, args.model + '_'+ args.version)
print("LOG_DIR", LOG_DIR)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
#write input arguments organize
LOG_FOUT.write("\n ========  Training Log ========  \n")
LOG_FOUT.write(":Input Arguments\n")
for var in args.__dict__:
	LOG_FOUT.write('[%10s]\t[%10s]\n'%(str(var), str(args.__dict__[var]) ) )


"""  Load Dataset """
#Load training dataset
train_dataset = Dataset_mmW(root=args.data_dir,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        split_number = args.data_split,
                        train=True)
#Load Validation Dataset
test_dataset = Dataset_mmW_val(root=args.data_dir,
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
  predicted_labels = predicted_labels[:,context_frames:,:,:]
  ground_truth_labels = ground_truth_labels[:,context_frames:,:,:]
  
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
    print("Layer[",layer_nr, "]", W, "\n")    

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
    learning_rate = tf.maximum(learning_rate,LR_MIN) # CLIP THE LEARNING RATE!
    return learning_rate   
    
def get_ReduceOnPlateu_learning_rate(lr_patience):
    learning_rate = BASE_LEARNING_RATE
    learning_rate = tf.train.natural_exp_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        lr_patience,  # Current index into the dataset.
                        1, #DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    
    learning_rate = tf.maximum(learning_rate,1e-4) # CLIP THE LEARNING RATE!
    
    return learning_rate  
  
  
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
    

def train():
  with tf.Graph().as_default():
    
    is_training_pl = tf.placeholder(tf.bool, shape=())
    pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, SEQ_LENGTH, NUM_POINTS)

    batch = tf.Variable(0)
    lr_patience = tf.Variable(0.0)
  

    
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
  	   	    'sampled_points_down1':args.down_points1,
  	   	    'sampled_points_down2':args.down_points2,
  	   	    'sampled_points_down3':args.down_points3}

    pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, model_params)
    
    # Normal Loss
    if args.balanced_loss == 0:
      loss = MODEL.get_loss(pred, labels_pl, context_frames = model_params['context_frames'])
      tf.summary.scalar('BCE_loss', loss)
    if args.balanced_loss == 1:
      # Balanced Loss
      loss = MODEL.get_balanced_loss(pred, labels_pl, context_frames = model_params['context_frames'])
      tf.summary.scalar('balanced_loss', loss)
    if args.balanced_loss == 2:
      # Balanced Loss
      loss = MODEL.get_MSE_loss(pred, labels_pl, context_frames = model_params['context_frames'])
      tf.summary.scalar('MSE_loss', loss)
      
    
  
    # Manual regulaizer -force all the trainbale weights to be small
    regularizer_scale = args.regularizer_scale
    regularizer_alpha = args.regularizer_alpha
    params_to_be_regulaized = []
    params = tf.trainable_variables()
    for layer in params:
      #only add regulaizer to layers with weights and bias. Exclude the bacth norm layers
      if ( ('weight' in layer.name)  or ('bias' in layer.name) ) :
        params_to_be_regulaized.append(layer)

    regularizer = tf.contrib.layers.l2_regularizer(regularizer_scale)
    reg_term = tf.contrib.layers.apply_regularization(regularizer,params_to_be_regulaized)

    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_losses= sum(reg_losses)
    tf.summary.scalar('reg_losses', reg_losses  )
    loss =  loss + regularizer_alpha * reg_losses
    tf.summary.scalar('total_loss', loss)
    
    # Calculate accuracy tensor
    accuracy = get_acurracy_tensor(pred, labels_pl,BATCH_SIZE,SEQ_LENGTH, NUM_POINTS, context_frames = model_params['context_frames'])
    tf.summary.scalar('accuracy', accuracy)

    # Get training operator
    if args.lr_scheduler == 0: 
      learning_rate = BASE_LEARNING_RATE
    if args.lr_scheduler == 1:
      learning_rate = get_learning_rate(batch) 
    if args.lr_scheduler == 2:
      learning_rate = get_ReduceOnPlateu_learning_rate(lr_patience)       
    if (is_training_pl == False): learning_rate = 0.0
    
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('lr_patience', lr_patience)
    
  
    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
    clipped_gradients = gradients # no gradient cliping
    train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=batch)    

    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours = 5)
    saver_best = tf.train.Saver()
    
    """
    print(" Trainable paramenters: ")
    params = tf.trainable_variables()
    c =0
    for layer in params:
      print("[",c,"] ", layer)
      c= c +1
    """
    
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session( config =config)
    
    # Add summary writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    # Init variables
    #init = tf.initialize_all_variables()
    early_stop_count = 0
    best_validation_loss = np.inf
    best_test_acurracy =  0.0



    if args.restore_training == 0:
      init = tf.global_variables_initializer()
      sess.run(init)
      ckpt_number = 0 
    if args.restore_training == 1:
      # Restore Session from last check-point  
      checkpoint_path_automatic = tf.train.latest_checkpoint(LOG_DIR)
      ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
      restore_checkpoint_path = checkpoint_path_automatic
      ckpt_number=ckpt_number[11:]
      ckpt_number=int(ckpt_number)

      print("ckpt_number:", ckpt_number)  
      print ("\n** Restore from checkpoint ***: ", restore_checkpoint_path)

      saver.restore(sess, restore_checkpoint_path)
            
    if args.restore_training == 2:      
      # Get the last checkpoint number
      checkpoint_path_automatic = tf.train.latest_checkpoint(LOG_DIR)
      ckpt_number = os.path.basename(os.path.normpath(checkpoint_path_automatic))
      ckpt_number=ckpt_number[11:]
      ckpt_number=int(ckpt_number)
      # change random seed
      np.random.seed(ckpt_number)
      tf.set_random_seed(ckpt_number)
      # Restore best checkpoint
      restore_checkpoint_path = os.path.join(LOG_DIR, "best_model.ckpt")
      print("ckpt_number:", ckpt_number)  
      print ("\n** Restore from checkpoint ***: ", restore_checkpoint_path)

      saver.restore(sess, restore_checkpoint_path)
      
      # change random seed
      np.random.seed(ckpt_number)
      tf.set_random_seed(ckpt_number)
          
      #assign_op = tf.assign(lr_patience, 10.0) # to delete
      #sess.run(assign_op)  # to delete
      
    
    ops = {'pointclouds_pl': pointclouds_pl,
  	   'labels_pl': labels_pl,
  	   'is_training_pl': is_training_pl,
  	   'pred': pred,
  	   'loss': loss,
       'reg_losses': reg_losses,
  	   'acc':accuracy,
  	   'train_op': train_op,
  	   'params': params,
  	   'merged': merged,
       'lr_patience': lr_patience, 
  	   'step': batch}

        
    for epoch in range(ckpt_number, args.num_iters):
      
      #  Test Data Val 
      if  (epoch > 2 and (epoch % 1 == 0 or epoch ==ckpt_number) ):        
        # Save Checkpoint
        save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step = epoch)
        print("Model saved in file: %s" % save_path)
      
        #Evaluate Validation
        print(" **  Evalutate VAL Data ** ")
        val_loss = eval_one_epoch(sess, ops, test_writer, epoch)
        
        # Early stopping
        if val_loss < best_validation_loss:
          best_validation_loss = val_loss
          print("[Lowest loss]:",best_validation_loss )
          print("[Patience count]:", early_stop_count)
          early_stop_count = 0
        else:
          early_stop_count = early_stop_count + 1
          print("[Lowest loss]:",best_validation_loss )
          print("[Patience count]:", early_stop_count)
          
        # Restore Checkpoint
        """  BUG! In some modules the weights are updated during evaluation """
        ckpt_number = os.path.basename(os.path.normpath(tf.train.latest_checkpoint(LOG_DIR)))
        print ("\n** Restore from checkpoint ***: ", tf.train.latest_checkpoint(LOG_DIR))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        ckpt_number= int( ckpt_number[11:] )
        np.random.seed(ckpt_number)
        tf.set_random_seed(ckpt_number)  
        ##restore patience-level

        # Saved the Best Model
        if val_loss == best_validation_loss:
          print("Save this as best model")
          #save model
          best_save_path = saver_best.save(sess, os.path.join(LOG_DIR, "best_model.ckpt") )
          print("Best Model saved in file: %s" % best_save_path)
          #Save Again in normal path
          save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step = epoch)
          #Save patience-level
          
      if (early_stop_count%20 == 0 and early_stop_count != 0 ): # Each time the patience reaches the value limit the learning rate decreases.
        """
        When early stop count reache 10 the patience is increased by 1
        When the patience is 10 the model exits training
        """
        #print("Type of lr_patience:", type(lr_patience))
        increment_lr_patience = lr_patience.assign_add(1.0)
        #print("Type of lr_patience:", type(lr_patience))
        sess.run(increment_lr_patience)
        ops['lr_patience'] = lr_patience
        early_stop_count = 0       
        

      # Train one epoch
      if (epoch % 1 == 0):
        train_one_epoch(sess, ops,train_writer, epoch)
        
      sys.stdout.flush() 
      
      # Reload Dataset
      if (epoch % 10 == 0 and epoch != 0):
        print("[Dataset Reload/Augumented] ")   
        train_dataset = Dataset_mmW(root=args.data_dir,
                              seq_length=args.seq_length,
                              num_points=args.num_points,
                              split_number = DATA_SPLIT,
                              train=True)
        
          
""" ------------------   """

def train_one_epoch(sess,ops,train_writer, epoch):
    """ Train one epoch of training data """
    is_training = True
  
    # for adaptative learning rate
    batch = epoch
    
    #Calculate how many batches are needed to do "see" the full training data
    total_frames = int(168960/2) #0 
    #training_size = len(train_dataset)
    #for j in range(0, training_size ): total_frames = total_frames +  np.shape(train_dataset.data[j])[0]
    nr_batches_in_a_epoch = int(total_frames/ ( BATCH_SIZE * SEQ_LENGTH) )
    #nr_batches_in_a_epoch = int(440/3) #440
    if args.data_split == -1: #Fast Debug
      nr_batches_in_a_epoch =4
    
    avg_epoch_loss =0 
    avg_epoch_accuracy = 0
    avg_regu_loss =0
    for batch_idx in tqdm (range(0,nr_batches_in_a_epoch) ):
      # Load Batch Data at Random 
      batch_data = get_batch(dataset=train_dataset, batch_size=args.batch_size) 
      batch = np.array(batch_data)
      if (FLAG_VEL_INT == 0):
        input_point_clouds = batch[:,:,:,0:3]
        input_labels = batch[:,:,:,3:4]
      if (FLAG_VEL_INT == 1):
        input_point_clouds = batch[:,:,:,:]
        input_labels = batch[:,:,:,5:6]        
        
      feed_dict = {ops['pointclouds_pl']: input_point_clouds, ops['labels_pl']: input_labels, ops['is_training_pl']: is_training}
      pred, lr_patience, summary, step, train_op, loss, regu_loss, accuracy =  sess.run([ops['pred'], ops['lr_patience'], ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['reg_losses'], ops['acc']], feed_dict=feed_dict)
      avg_regu_loss = avg_regu_loss + regu_loss
      avg_epoch_loss = avg_epoch_loss + loss
      avg_epoch_accuracy = avg_epoch_accuracy + accuracy
      
      train_writer.add_summary(summary, step)
    
    print("\n lr_patience", lr_patience,'\n')
    avg_epoch_loss = avg_epoch_loss/nr_batches_in_a_epoch
    avg_epoch_accuracy =avg_epoch_accuracy/nr_batches_in_a_epoch
    avg_regu_loss = avg_regu_loss/nr_batches_in_a_epoch
    print("[ %s  epoch:%03d ] Loss: %f\t Regu Loss %f\t  Accuracy: %f\t"%( str( args.model + '_' + args.version) ,  epoch, avg_epoch_loss, avg_regu_loss, avg_epoch_accuracy) )
   
   
    #Got acess to lr_patience value 
    if lr_patience > PATIENCE_LIMIT:
      #Early Stop
      log_string("\n\n---- [EARLY STOP ] No more Patience -----\n\n")
      exit()                     
    
def eval_one_epoch(sess,ops,test_writer, epoch):
    """ Eval all sequences of test dataset """
    is_training = False
    
    nr_tests = len(test_dataset) 
    num_batches = nr_tests // BATCH_SIZE

    
    x = [i for i in range(1, nr_tests+1) if nr_tests % i == 0]
    if (BATCH_SIZE not in x): print("[NOT LOADING ALL TEST DATA] - To test the full test data: select a batch size:", x)

    total_accuracy =0
    total_loss = 0
    Tp =0 #true positives total
    Tn =0
    Fp =0
    Fn =0
    
    for batch_idx in tqdm ( range(num_batches) ):
      start_idx = batch_idx * BATCH_SIZE
      end_idx = (batch_idx+1) * BATCH_SIZE
      cur_batch_size = end_idx - start_idx
      input_point_clouds =[]
      input_labels =[]
      
      for idx  in range(start_idx,end_idx): #sequences to be tested
        test_seq = test_dataset[idx]   
        test_seq =np.array(test_seq)
        if (FLAG_VEL_INT == 0):
          point_clouds = test_seq[:,:,0:3]
          labels = test_seq[:,:,3:4]
          input_point_clouds.append(point_clouds)
          input_labels.append(labels)
        if (FLAG_VEL_INT == 1):
          point_clouds = test_seq[:,:,:]
          labels = test_seq[:,:,5:6] 
          input_point_clouds.append(point_clouds)
          input_labels.append(labels)             
      
      input_point_clouds = np.array(input_point_clouds)
      input_labels = np.array(input_labels)
      
      # Send to model to be evaluated
      feed_dict = {ops['pointclouds_pl']: input_point_clouds, ops['labels_pl']: input_labels, ops['is_training_pl']: is_training}
      pred, summary, step, train_op, loss, accuracy, params =  sess.run([ops['pred'], ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['acc'], ops['params'] ], feed_dict=feed_dict) 
      test_writer.add_summary(summary, step)  
      
      total_accuracy = total_accuracy + accuracy
      total_loss = total_loss + loss
      accuracy, true_positives, false_positives, true_negatives,false_negatives = get_classification_metrics(pred, input_labels, args.batch_size, args.seq_length,args.num_points, args.context_frames )
      Tp = Tp + (true_positives) 
      Fp = Fp + (false_positives)
      Tn = Tn + (true_negatives)
      Fn = Fn + (false_negatives)
  
                   

    
    mean_loss = total_loss/ num_batches
    mean_accuracy = total_accuracy/ num_batches
    precision = Tp / ( Tp+Fp)
    recall = Tp/(Tp+Fn)
    f1_score =2 * ( (precision * recall)/(precision+recall) )
    
    print('**** EVAL: %03d  %s ****' % (epoch, str( args.model + '_' + args.version) ) )
    print("[VALIDATION] Loss   %f\t  Accuracy: %f\t"%( mean_loss, mean_accuracy) )
    print("Precision: ", precision, "\nRecall: ", recall, "\nF1 Score:", f1_score)
    print(' -- ')   
    
    """ Visualize weights in terminal """
    #print_weights(sess, params, 1)
    #print_weights(sess, params, 9)
    #print_weights(sess, params, 6) 
    #exit()
    #print_weights(sess, params, 30)
    #print_weights(sess, params, layer_nr=57)
    #print_weights(sess, params, layer_nr=61)

    # Write to File
    #log_string('****  %03d ****' % (epoch))
    date_string = str(datetime.now().hour) +':'+ str(datetime.now().minute) + '   -' +str(datetime.now().day)+'/'+str(datetime.now().month)
    log_string('%03d  eval mean loss, accuracy: %f \t  %f \t %s' % (epoch, mean_loss , mean_accuracy, date_string))
    if not np.isnan(precision) and not np.isnan(recall) and not np.isnan(f1_score):
    	log_string('Precision %f Recall, F1 Score: %f \t  %f \t ]' % (precision, recall , f1_score))
     
    return mean_loss        
                
                  
if __name__ == "__main__":
    train()
    LOG_FOUT.close()        




