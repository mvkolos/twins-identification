import numpy as np
import cv2 
import pandas as pd
import os

from torch.utils.data import Dataset


def imread(path):
    img = cv2.imread(os.path.join(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_keypoints(kp_raw):
    data = list(kp_raw.apply(eval))
    return np.array(data)
    
class TwinPairsDataset(Dataset):
    def __init__(self, dataroot, df_pairs, df_views, transform, keypoints=False, id_columns=['id_1', 'id_2']):
        '''
        dataroot: path to folder with items
        df_pairs: pd dataframe containing pairs of ids and a correspodind label:
                    'Same', 'Fraternal', 'Identical', 'UnknownTwinType',
                    'IdenticalMirror', 'Sibling', 'IdenticalTriplet'
        df_views: pd dataframe containing list of available for each id in the dataset                    
        transform: torchvision transform
        '''
        self.dataroot = dataroot
        self.df_pairs = df_pairs
        self.df_views = df_views
        self.transform = transform
        self.keypoints = keypoints
        self.id_columns = id_columns
        self.n = len(id_columns)
        
    def __getitem__(self, index):
        def get_img_path(person_id, view):
            path = os.path.join(self.dataroot, person_id, view)
            return imread(path)
        
        ids = self.df_pairs.iloc[index][self.id_columns].values
        ids = [str(x) for x in ids]
    
        label = int(self.df_pairs.iloc[index].label=='Same')
        
        if ids[0]==ids[1]:
             views = np.random.choice(self.df_views.loc[ids[0]]['filename'], size=2, replace=False) 
        else:
            views = [np.random.choice(self.df_views.loc[ids[i]]['filename']) for i in range(self.n)]

        paths = [os.path.join(self.dataroot, ids[i], views[i]) for i in range(self.n)]
        
        images = [imread(path) for path in paths]
        
        sample = dict([(f'image{i}',images[i]) for i in range(self.n)])
        
        if self.keypoints:
            kp = [pd.read_csv(os.path.join(self.dataroot, ids[i], 'keypoints.csv')) for i in range(self.n)]
            keypoints = [parse_keypoints(kp[i][views[i]]) for i in range(self.n)]
            
            sample.update(dict([(f'keypoints{i}',keypoints[i]) for i in range(self.n)]))
            
        if self.transform:
            samples = [{'image':image} for image in images]
            
            if self.keypoints:
                for i in range(self.n):
                    samples[i]['keypoints'] = keypoints[i]
                
            augs = [self.transform(**sample) for sample in samples]
            
            sample = dict([(f'image{i}', augs[i]['image']) for i in range(self.n)])

            if self.keypoints:
                sample.update(dict([(f'keypoints{i}',np.array(augs[i]['keypoints'])) for i in range(self.n)]))
        
        sample['label'] = label   
        return sample
    
    def __len__(self):
        return self.df_pairs.shape[0]
    
    
class ClassificationDataset(Dataset):
    def __init__(self, dataroot, df_views, transform, keypoints=False):
        '''
        dataroot: path to folder with items
        df_views: pd dataframe containing list of available for each id in the dataset                    
        transform: torchvision transform
        '''
        self.dataroot = dataroot
        self.df_views = df_views
        self.transform = transform
        self.keypoints = keypoints
        
    def __getitem__(self, index):
        def get_img_path(person_id, view):
            path = os.path.join(self.dataroot, person_id, view)
            return imread(path)
        
        views, subject_id, label  = self.df_views.iloc[index].values
        subject_id = str(subject_id)
        view = np.random.choice(views) 

        path = os.path.join(self.dataroot, subject_id, view)
        
        img = imread(path)       
            
        sample = {'image': img, 'label': label}
        
        if self.keypoints:
            kp = pd.read_csv(os.path.join(self.dataroot, subject_id, 'keypoints.csv'))
            sample['keypoints'] = parse_keypoints(kp[view])
        
        if self.transform:            
            sample = self.transform(**sample)
            if self.keypoints:
                sample['keypoints'] = np.array(sample['keypoints'])
           
        return sample
    
    def __len__(self):
        return self.df_views.shape[0]
    