import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from git import Repo
import cv2
import tensorflow as tf
from scipy import io
from pathlib import Path

class readData():
    '''
    Read data from github
    ----------

    Returns
    -------
    self.images:
        input image data
    self.masks:
        Output masks of images

    '''
    def __init__(self, cwd_path = None,repo_link =  "https://github.com/bearpaw/clothing-co-parsing.git", existing_dir = True ):
#     def __init__(self, *args, **kwargs):
        
        self.cwd_path = os.getcwd()
        self.repo_link =  repo_link
        self.exdir = existing_dir
        
        self.data_path = ""
        self.images = []
        self.masks = []
    
        
#         self.final_set,self.labels = self.build_data()
    def CloneGit(self):
        '''
        Read Data from git
        ----------
        
        Returns
        -------
        Dataframe representation of the csv file
        '''
        

        self.cwd_path = str(Path(self.cwd_path).parents[1])
        self.data_path = self.cwd_path+"/clone/"
        self.exdir = os.path.isdir(self.cwd_path + '/clone') 
        if self.exdir == False:
            Repo.clone_from(self.repo_link, self.data_path)
        

        
    def readImage(self):
        images = []
        for i in range(1,1001):
            url = self.data_path +'photos/%04d.jpg'%(i)
            img = cv2.imread(url)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(tf.convert_to_tensor(img))
    def readMask(self):
    
        for i in range(1,1001):
            url = self.data_path +'annotations/pixel-level/%04d.mat'%(i)
            file = io.loadmat(url)
            mask = tf.convert_to_tensor(file['groundtruth'])
            self.masks.append(mask)
                   
            

    ## address the missing information
    def readImageData(self):
        '''
        Replace the missing value with the zero.
        ----------
        
        Returns
        -------
        Dataframe with replaced missing value.
        '''
        self.CloneGit()
        self.readImage()
        self.readMask()
        
        
        ### Print how many sample we have
        print((len(self.images), len(self.masks)))
        return self.images,self.masks