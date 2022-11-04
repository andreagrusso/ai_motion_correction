# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 09:12:41 2022

@author:  Andrea Gerardo Russo, BME, PhD
University of Campania "Luigi Vanvitelli", Naples, Italy

@email: andreagerardo.russo@unicampania.it

"""
import torch,os 
import torchio as tio

from dicom_processing import mosaic_to_mat



#%%


class Create_test_dataset(torch.utils.data.Dataset):
    
    """Generates data for Pytorch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    
    def __init__(self, pairs_list,dims):
        
        self.pairs = pairs_list #each pair contains movable and corresponding target
        self.list_IDs = range(len(pairs_list)) #indices of the list
        self.dims = dims #list of the desidered dims for zero padding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        

    def __len__(self):
        """Denotes the number training data
        :return: number of training data
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        
        """ Generates one sample of data """


        # Select sample
        ID = self.list_IDs[index]


        # Load data and get label
        pair = self.pairs[ID]
      

        trg_mat = mosaic_to_mat(pair[1])
        mov_mat = mosaic_to_mat(pair[0])
        #the affine is stored in the tensor!!!!
        
       
        #create tensor 
        mov_tensor = tio.ScalarImage(mov_mat)
        trg_tensor = tio.ScalarImage(trg_mat)
        
        #store the original matrix size
        orig_dim = mov_tensor.shape[1:]    
        
        #IT LOOKS LIKE THAT THE RETURNED DATA ARE NOT ANYMORE TORCHIO OBJS
        # THEREFORE WE LOSE EVERY FUNCTIONS AND WE NEED TO STORE THE AFFINE AFTER 
        # THE TO_CANONICAL TRANSFORMATION
        tc = tio.transforms.ToCanonical() #bring the image in RAS+
        mov_tensor = tc(mov_tensor)
        trg_tensor = tc(trg_tensor)
        orig_mov_affine = mov_tensor['affine']
        orig_trg_affine = mov_tensor['affine']

        #compose transformation
        transform = tio.Compose([
            tio.transforms.RescaleIntensity(out_min_max=(0, 1)), #MinMaxscaling
            tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.50)), #masking
            tio.transforms.CropOrPad((128,128,128)) #padding
        ])
        
        
        mov_tensor = transform(mov_tensor)
        trg_tensor = transform(trg_tensor)
        
        #swap the affine to have the one with true image size in RAS+
        mov_tensor['affine'] = orig_mov_affine
        trg_tensor['affine'] = orig_trg_affine

    
        #remove nifti file from disk
        if trg_mat == mov_mat:
            os.remove(trg_mat)
        else:        
            os.remove(trg_mat)
            os.remove(mov_mat)
        
        return trg_tensor, mov_tensor, orig_dim






class Create_train_dataset(torch.utils.data.Dataset):
    
    """Generates data for Pytorch
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    
    def __init__(self, pairs_list,dims):
        
        self.pairs = pairs_list #each pair contains movable and corresponding target
        self.list_IDs = range(len(pairs_list)) #indices of the list
        self.dims = dims #list of the desidered dims for zero padding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        

    def __len__(self):
        """Denotes the number training data
        :return: number of training data
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        
        """ Generates one sample of data """


        # Select sample
        ID = self.list_IDs[index]


        # Load data and get label
        pair = self.pairs[ID]
        
        trg_mat = mosaic_to_mat(pair[1])
        mov_mat = mosaic_to_mat(pair[0])
        #the affine is stored in the tensor!!!!
        
        #create tensor 
        mov_tensor = tio.ScalarImage(mov_mat)
        trg_tensor = tio.ScalarImage(trg_mat)
        
        #store the original matrix size
        orig_dim = mov_tensor.shape[1:]       

        #compose transformation
        transform_trg = tio.Compose([
            tio.transforms.ToCanonical(), #bring the image in RAS+
            tio.transforms.RescaleIntensity(out_min_max=(0, 1)), #MinMaxscaling
            tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.50)), #masking
            tio.transforms.CropOrPad((128,128,128)) #padding
        ])
        transform_mov = tio.Compose([
            tio.transforms.ToCanonical(), #bring the image in RAS+
            tio.transforms.RescaleIntensity(out_min_max=(0, 1)), #MinMaxscaling
            tio.transforms.RandomAffine(scales=0,
                                                degrees=[3,3,3],#sampling between -3,3
                                                translation=[3,3,3],#sampling between -3,3
                                                image_interpolation='bspline',
                                                center='origin'),
            tio.transforms.Mask(masking_method=lambda x: x > torch.quantile(x,0.50)), #masking
            tio.transforms.CropOrPad((128,128,128)) #padding
        ])
        
        
        
        mov_tensor = transform_mov(mov_tensor)
        trg_tensor = transform_trg(trg_tensor)

    
        #remove nifti file from disk  
        if trg_mat == mov_mat:
            os.remove(trg_mat)
        else:        
            os.remove(trg_mat)
            os.remove(mov_mat)
       
        return trg_tensor, mov_tensor, orig_dim
        



# # import matplotlib.pyplot as plt   
# # f,ax = plt.subplots(1,2)
# # ax[0].imshow((movable_mat-target_mat)[:,:,32], cmap ='Greys_r')
# # ax[1].imshow((scaled_mov_data-scaled_targ_data)[:,:,32], cmap = 'Greys_r')
# # ax[1,0].imshow((m_scaled_mov_data-m_scaled_targ_data)[:,:,32], cmap='Greys_r')
# # ax[1,1].imshow((z_scaled_mov_data-z_scaled_targ_data)[:,:,32], cmap='Greys_r')