#PIL
import PIL 

#numpy
import numpy as np
#import numpy.ma as ma

#Torch-packages
from torch.utils import data
#from torch.utils.data import Dataset
import torch
from torchvision import transforms as T

class OurFaceDataSet(data.Dataset):
    '''
        Dataset class for the "Facial Landmark Detection by 
        Deep Multi-task Learning"-dataset.
    '''
    
    def __init__(self, dataset_dir, config_filename, head_pose, smile_filter, transform, cropper):
        '''
        Arguments:
            dataset_dir: string stating the directory of the dataset
            config_filename: string stating the name of the config-file 
                        (training.txt, testing.txt. Both contains the 
                         the information of the image-path, the 
                         coordinates of the points of interests and the 
                         image attributes)
            head_pose: int stating a certain class of head pose, which
                       is taken into account
            smile_filter: smiling or nonsmiling faces are taken into account
            transform: T.transform function, which modify the input-images 
        
        '''
        #Initialize and preprocess the dataset.
        self.dataset_dir = dataset_dir
        self.config_filename = config_filename
        self.head_pose = head_pose
        self.smile_filter = smile_filter
        self.transform = transform
        self.cropper = cropper
        self.data_config = []
        self.data_config_mod = []
        self.number_images = []
        

        #Load the data config file (training or testing)
        self.load_data(self.dataset_dir, self.config_filename)

        #Filter the data config file, so that only a certain kind of faces (for example: head pose) is 
        #listed
        self.filter_data(self.data_config, self.head_pose, self.smile_filter)        

        #Get the number of images
        self.number_images = self.data_config_mod.shape[0]
        

        
        
    def load_data(self, dataset_dir, config_filename):
        '''
        Arguments:
            dataset_dir: string stating the directory of the dataset
            config_filename: string stating the name of the config file

        Returns:
            config_data: numpy-array, which contains:
                            path to the images, points of interest, labels
        '''
        #Read the data config file
        config_data = np.genfromtxt(os.path.join(dataset_dir,
                                              config_filename),
                                 delimiter=" ", dtype=(str))
        config_data[:,1:14] = (config_data[:,1:14]).astype(np.float)

        #Modify the path-string
        for idx in range(config_data.shape[0]):
            filename = config_data[idx, 0].split("/")
            config_data[idx, 0] = os.path.join(filename[0], filename[1])

        self.data_config = config_data
        

    def filter_data(self, config_data, filter_number, smile_no):
        '''
        Arguments:
            config_data: array which stores the whole given information of each image
            filter_number: integer of the wanted head pose (if it is 100, it is mixed)
            smile_no: integer, which states, whether smiling faces, non-smiling or mixed faces
                      are taken into account (1: smile, 2:non-smile, 3:mixed) 
        Returns: 
            data_config_mod:  the filtered config_data 
        '''
        config_data_temp = []        
        for i in range(config_data.shape[0]):
            if(float(filter_number) == float(100)):
                if(float(smile_no) < float(3) and float(config_data[i, 12]) == float(smile_no)):
                    config_data_temp.append(config_data[i, :])
                else:
                    config_data_temp.append(config_data[i, :])

            else:
                if(float(smile_no) < float(3) and float(config_data[i, 12]) == float(smile_no) and float(config_data[i, 14]) == float(filter_number)):
                    config_data_temp.append(config_data[i, :])
                elif(float(config_data[i, 14]) == float(filter_number)):
                    config_data_temp.append(config_data[i, :])
                        
            #if(int(smile_no) == 0)
            #if(int(config_data[i, 14]) == filter_number and float(config_data[i, 12]) == 1.0):
            #if(int(config_data[i, 14]) == filter_number):
            #    config_data_temp.append(config_data[i, :])
        
        self.data_config_mod = np.array(config_data_temp)
        print("1: Length of the config-file:")        
        print(self.data_config_mod.shape)     
      

    def __getitem__(self, index):
        #Return one image and its corresponding attribute label
        dataset = self.data_config_mod
        filename = os.path.join(self.dataset_dir, dataset[index, 0])
        label = torch.from_numpy(np.reshape(np.array(dataset[index, 12], dtype=np.float) - 1, (1,1)))
        label = label.float()
        poi_nose = torch.from_numpy(np.array((dataset[index, 3], dataset[index, 8]), dtype=np.float))
        poi_nose = poi_nose.float()
        poi_left_mouth_co = torch.from_numpy(np.array((dataset[index, 4], dataset[index, 9]), dtype=np.float))
        poi_left_mouth_co = poi_left_mouth_co.float()
        poi_right_mouth_co = torch.from_numpy(np.array((dataset[index, 5], dataset[index, 10]), dtype=np.float))
        poi_right_mouth_co = poi_right_mouth_co.float()
        image = PIL.Image.open(filename)
        cropped_image = self.transform(self.cropper(image, poi_nose, poi_left_mouth_co, poi_right_mouth_co))
        return cropped_image, torch.FloatTensor(label)
        
    def __len__(self):
        #Return the number of images
        return self.number_images
        


#Function, which processes the image (crops the ellipse)
def our_image_cropper(image, poi_nose, poi_left_mouth_co, poi_right_mouth_co):
    #Size of the image
    width, height = image.size
    #Calculate the ROI
    f_1 = poi_left_mouth_co.numpy()
    f_2 = poi_right_mouth_co.numpy()
    f_3 = poi_nose.numpy()
    f2_f1 = (f_2 - f_1)
     
    f_0 = (np.linalg.norm(f2_f1)/2) * (f2_f1/np.linalg.norm(f2_f1)) + f_1
    f2_f0 = (f_2 - f_0)
    
    b = f_0 - f_3
    e_a = f2_f0
    a = np.sqrt(np.dot(e_a,e_a) + np.dot(b, b))
    a_vec = ((f2_f0)/np.linalg.norm(f2_f0)) * a
    
    #Calculate top left and bottom right point
    p_t = f_0 - b - a_vec
    p_b = f_0 + b + a_vec
    
    #Proof, if those points are in the image
    if((p_t[1]-p_b[1])<=0):
        if(p_t[0]<0):
            p_t[0]=0
        if(p_t[1]<0):
            p_t[1]=0
        if(p_b[0]>width):
            p_b[0]=width
        if(p_b[1]>height):
            p_b[1]=height
    else:
        if(p_t[0]<0):
            p_t[0]=0
        if(p_t[1]<height):
            p_t[1]=height
        if(p_b[0]>width):
            p_b[0]=width
        if(p_b[1]<0):
            p_b[1]=0
    
    cropped = image.crop((p_t[0], p_t[1],
                p_b[0], p_b[1]))
    
    return cropped    
    
#Returns a data loader
def get_loader(dataset_dir, head_pose, smile_filter, img_size=64, batch_size=16,
               training=True, num_workers=1):
    #This builds a data loader and returns it
    transform = []
    
    if training:
        transform.append(T.RandomHorizontalFlip())
        
    transform.append(T.Resize((img_size, img_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    #Decide between testing or training
    config_filename = 0
    if training:
        config_filename = "training.txt"
    else:
        config_filename = "testing.txt"
        
    
    #Create OurFaceDataSet
    dataset = OurFaceDataSet(dataset_dir, config_filename, head_pose, smile_filter,
                             transform, our_image_cropper)
    print("2: Length of the Dataset: ")    
    print(len(dataset))

    #Create data loader
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=training,
                                  num_workers=num_workers)
    
    return data_loader

