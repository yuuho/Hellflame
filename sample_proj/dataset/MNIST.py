
import random

import torch
import torchvision


# torchvision uses classname as directory name
#   eg) MLDATA/torchvision_dataset/MNIST/
#           - processed
#           - raw
class MNIST(torchvision.datasets.MNIST):

    def __init__(self, datasetDB_root_path, torchvision_path, 
                    mode, eval_rate)->None:
        # if the dataset does not exist, the dataset is automatically downloaded.
        super().__init__(   root=str(datasetDB_root_path/torchvision_path),
                            train={'train':True,'eval':True,'test':False}[mode],
                            download=True)
        
        # シャッフルしてから train,eval に分割して取り出し
        num_data = len(self.data)
        random.seed(0)
        pointer = random.sample(list(range(num_data)), num_data)
        partition = int(num_data*(1.0-eval_rate))
        sl = {  'train': slice(None,        partition   ),
                'eval' : slice(partition,   None        ),
                'test' : slice(None,        None        ) }[mode]
        self.data = self.data[pointer][sl]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index] # (H,W) uint8
        normed = img / 255.0 *2.0 - 1.0
        return normed

Dataset = MNIST
