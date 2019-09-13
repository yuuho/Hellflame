import torch
import os


def _set_device(self):
    # GPUの設定
    if self.config.env['cuda_string']=='cpu':
        self.device = torch.device('cpu')
    ## 何も設定されていないときはすべてのGPUを数え上げて使用
    elif self.config.env['cuda_string'] is None:
        num_devices = torch.cuda.device_count()
        if num_devices>0:
            self.config.env['cuda_string'] = ','.join([str(i) for i in range(num_devices)])
            self.device = torch.device('cuda:'+self.config.env['cuda_string'])
            torch.backends.cudnn.benchmark = True
        else:
            self.config.env['cuda_string'] = 'cpu'
            self.device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.env['cuda_string']
    ## 何か設定されていても実際にはtorchから見えるすべてのGPUを使用することとなる
    else:
        self.device = torch.device('cuda:'+','.join([str(i) for i in range(torch.cuda.device_count())]))
        torch.backends.cudnn.benchmark = True


'''
Trainerクラスに書いている
```
class Trainer:
    def _set_device(self):
        hogehoge
```
を

``from hellfire.snippets.torch_gpu import _set_device``
した上で
```
class Trainer:
    _set_device = _set_device
```
という感じにクラスプロパティ的に挿入するだけで良い．
'''