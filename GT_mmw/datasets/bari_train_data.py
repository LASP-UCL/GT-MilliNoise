import numpy as np
import random
import os

def rotate_translate_jitter_pc(pc, angle, x,y,z):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians. (but any value works)
    """
    for p in range (pc.shape[0]):
                
	    ox, oy, oz = [0,0,0]
	    px, py, pz = pc[p,0:3]
	    
	    # Do via Matrix mutiplication istead
	    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy) + x + (np.random.rand() * (0.01 * 2) - 0.01 )
	    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy) + y + (np.random.rand() * (0.01 * 2) - 0.01 )
	    qz = pz + z  + (np.random.rand() * (0.01 * 2) - 0.01 )
	    pc[p,0:3] = qx, qy, pz
	   
    return pc
    
def shuffle_pc(pc):

  idx = np.arange(len(pc))
  np.random.shuffle(idx)
  pc =pc[idx]
  
  return pc

def get_dataset_split(split_number):
    
    print("split number: ",split_number )
    if (split_number == -1): # For Fast Debug
            test_npy_files =  [3,7,49,55,57,68,70,4,8,53,56,62,67,69,6,9,51,58,59,63,66,65]

    if (split_number == 11): # Split 11
        test_npy_files =  [69,73,3,55, 53,67,74,80,88,96,61,6,64,92,70,85,50,56,57] # missing 9 test + val
  

    if (split_number == 13): # Split 13
        #test_npy_files =  [68,77,84,95,87, 98,93,86,75,65,62,4,49,7,51,72] # test + val
        test_npy_files =  [3,4,6,7,8,49,50,51,53,54,55] # test + val

    if (split_number == 14): # Split 14
        test_npy_files =  [3,4,6,7,8,49,50,51,53,54,55] # test + val
        
    if (split_number == 15): # Split 14
        test_npy_files =  [71,72,73,74,75,76,77,79,63] # test + val
        
    if (split_number == 16): # Split 16
        test_npy_files =  [71,72,73,74,75,76,77,79] # test + val
        
    if (split_number == 4): # Split 16
        test_npy_files =  [57,58,59,61,62,63,64,65,66,67,69,70]  
        
    if (split_number == 17): # Split 14
        test_npy_files =  [3,4,6,7,8,49,50,51,53,54,55,56] # test + val
        
                                        
                            
    test_npy_files = [ 'labels_run_' + str(num)+ '.npy' for num in test_npy_files]

    return test_npy_files

class MMW(object):
    def __init__(
        self,
        root="/home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset",
        seq_length=100,
        num_points=200,
        train=True,
        split_number =0, 
    ):

        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []
        
        
        log_nr = 0
        root = root + '/' +str(num_points) + '/all_runs_final'
        if train:

            # load all files
            all_npy_files = os.listdir(root)

                        
            # Select the split of dataset
            test_npy_files = get_dataset_split(split_number)

 
            # Remove test data from training set
            npy_files = [string for string in all_npy_files if string not in test_npy_files]
            
            
            if (split_number == -1): # For Fast Debug
                npy_files =  ['labels_run_56.npy']
            
            print("npy_files:", npy_files)
            
            for run in npy_files:
                file_path = os.path.join(root, run)
                self.data.append(file_path)
            
            print("Train  data", np.shape(self.data) ) # Nr of runs 
         

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):


        #print(" ---- loading item ---- at random")
        nr_seq = len(self.data)
        idx1  = np.random.randint(0, nr_seq)
        
        
        log_data_path = self.data[idx1]
        npy_run = np.load(log_data_path)
        npy_run = npy_run[0]
        
        total_lenght = npy_run.shape[0]
        
        start_limit = total_lenght - (self.seq_length)
        start = np.random.randint(0, start_limit-1)
        end = start + ( self.seq_length )
        cloud_sequence = []
        

        for i in range(start,end):
            pc = npy_run[i]
            #print("pc", pc.shape)
            #exit()
            pc = shuffle_pc(pc)
            cloud_sequence.append(pc)
        points = np.stack(cloud_sequence, axis=0)
        
        direction = 1
        # I can flip the direction of the point cloud
        if np.random.rand() < 0.25: direction = -1
        if (direction == -1):
            # Reverse Array order
            points = np.flip(points, axis = 0)
            
        return points
        


