import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url


# Inception構造の各ブランチをマージするクラスの作成
class _CAT(nn.Module):
    def __init__(self,mods):
        super().__init__()
        self.branch = nn.ModuleList(mods)
    def forward(self,x):
        return torch.cat([m(x) for m in self.branch],1)


# torch.flattenをnn.Module化
class _Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return torch.flatten(x,1)


class FID_InceptionV3(nn.Module):

    structure = {
        'first':[   ['conv_3x3_down',   3,  32],['BN',  32],['relu'],
                    ['conv_3x3',       32,  32],['BN',  32],['relu'],
                    ['conv_3x3_flat',  32,  64],['BN',  64],['relu'],
                    ['max_pool_3x3']                                ],
        'second':[  ['conv_1x1',       64,  80],['BN',  80],['relu'],
                    ['conv_3x3',       80, 192],['BN', 192],['relu'],
                    ['max_pool_3x3']                                ],
        'inception_A':[['CAT',lambda in_ch,out3_ch:{
            0:[ ['conv_1x1',     in_ch,     64],['BN',  64],['relu']],
            1:[ ['conv_1x1',     in_ch,     48],['BN',  48],['relu'],
                ['conv_5x5_flat',   48,     64],['BN',  64],['relu']],
            2:[ ['conv_1x1',     in_ch,     64],['BN',  64],['relu'],
                ['conv_3x3_flat',   64,     96],['BN',  96],['relu'],
                ['conv_3x3_flat',   96,     96],['BN',  96],['relu']],
            3:[ ['avg_pool_3x3'],
                ['conv_1x1',     in_ch,out3_ch],['BN',out3_ch],['relu']]      }]],
        'inception_B':[['CAT',lambda:{
            0:[ ['conv_3x3_down', 288, 384],['BN', 384],['relu']],
            1:[ ['conv_1x1',      288,  64],['BN',  64],['relu'],
                ['conv_3x3_flat',  64,  96],['BN',  96],['relu'],
                ['conv_3x3_down',  96,  96],['BN',  96],['relu']],
            2:[ ['max_pool_3x3']]                                       }]],
        'inception_C':[['CAT',lambda mid12_ch:{
            0:[ ['conv_1x1',          768,     192],['BN',     192],['relu']],
            1:[ ['conv_1x1',          768,mid12_ch],['BN',mid12_ch],['relu'],
                ['conv_1x7_flat',mid12_ch,mid12_ch],['BN',mid12_ch],['relu'],
                ['conv_7x1_flat',mid12_ch,     192],['BN',     192],['relu']],
            2:[ ['conv_1x1',          768,mid12_ch],['BN',mid12_ch],['relu'],
                ['conv_7x1_flat',mid12_ch,mid12_ch],['BN',mid12_ch],['relu'],
                ['conv_1x7_flat',mid12_ch,mid12_ch],['BN',mid12_ch],['relu'],
                ['conv_7x1_flat',mid12_ch,mid12_ch],['BN',mid12_ch],['relu'],
                ['conv_1x7_flat',mid12_ch,     192],['BN',     192],['relu']],
            3:[ ['avg_pool_3x3'],
                ['conv_1x1',          768,     192],['BN',     192],['relu']]      }]],
        #'aux':[ ['avg_pool_5x5'],
        #        ['conv_1x1', 768, 128],['BN', 128],['relu'],
        #        ['conv_5x5', 128, 768],['BN', 768],['relu'],
        #        ['ada_avg_pool'],['flatten'],['fc', 768,1000]           ],
        'inception_D':[['CAT',lambda:{
            0:[ ['conv_1x1',      768, 192],['BN', 192],['relu'],
                ['conv_3x3_down', 192, 320],['BN', 320],['relu']],
            1:[ ['conv_1x1',      768, 192],['BN', 192],['relu'],
                ['conv_1x7_flat', 192, 192],['BN', 192],['relu'],
                ['conv_7x1_flat', 192, 192],['BN', 192],['relu'],
                ['conv_3x3_down', 192, 192],['BN', 192],['relu']],
            2:[ ['max_pool_3x3']                                ] }]],
        #'inception_E':[['CAT',lambda in_ch:{
        'inception_E':[['CAT',lambda in_ch,mode:{
            0:[ ['conv_1x1',     in_ch, 320],['BN', 320],['relu']],
            1:[ ['conv_1x1',     in_ch, 384],['BN', 384],['relu'],['CAT',lambda *_:{
                    0:[ ['conv_1x3_flat', 384, 384],['BN', 384],['relu']],
                    1:[ ['conv_3x1_flat', 384, 384],['BN', 384],['relu']],  }]],
            2:[ ['conv_1x1',     in_ch, 448],['BN', 448],['relu'],
                ['conv_3x3_flat',  448, 384],['BN', 384],['relu'],['CAT',lambda *_:{
                    0:[ ['conv_1x3_flat', 384, 384],['BN', 384],['relu']],
                    1:[ ['conv_3x1_flat', 384, 384],['BN', 384],['relu']],  }]],
            3:[ [mode+'_pool_3x3'],
                ['conv_1x1',     in_ch, 192],['BN', 192],['relu']]      }]],
        #'last': [['ada_avg_pool'],['dropout'],['flatten'],['fc',2048,1000]]
        'last1': [['ada_avg_pool']],
        'last2': [['dropout'],['flatten'],['fc',2048,1008]]
    }

    def __init__(self):
        super().__init__()
        self.first = self._make_block('first')
        self.second = self._make_block('second')
        self.third = nn.Sequential(*[self._make_block(*arg) for arg in
                            [('inception_A',192,32),('inception_A',256,64),('inception_A',288,64),('inception_B',),
                            ('inception_C',128),('inception_C',160),('inception_C',160),('inception_C',192)]])
        #self.aux = self._make_block('aux')
        self.fourth = nn.Sequential(*[self._make_block(*arg) for arg in
                            [('inception_D',),('inception_E',1280,'avg'),('inception_E',2048,'Xmax')]])
        self.last1 = self._make_block('last1')
        self.last2 = self._make_block('last2')

    def _make_block(self,key,*args):
        # convは全部 no_bias, paddingがある場合はzero padding, dilation/groupは行わない
        definition = {
            'conv_3x3':     lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=3, stride=1, padding=0,
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_3x3_down':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=3, stride=2, padding=0,
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_3x3_flat':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=3, stride=1, padding=1,
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_1x3_flat':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=(1,3), stride=1, padding=(0,1),
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_3x1_flat':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=(3,1), stride=1, padding=(1,0),
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_1x1':     lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=1, stride=1, padding=0,
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_5x5':     lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=5, stride=1, padding=0,
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_5x5_flat':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=5, stride=1, padding=2,
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_1x7_flat':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=(1,7), stride=1, padding=(0,3),
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'conv_7x1_flat':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=(7,1), stride=1, padding=(3,0),
                            dilation=1, groups=1, bias=False, padding_mode='zeros'),
            'BN':           lambda *config: nn.BatchNorm2d(
                            num_features=config[0], eps=0.001,
                            momentum=0.1, affine=True, track_running_stats=True),
            'relu':         lambda *config: nn.ReLU(inplace=True),
            'dropout':      lambda *config: nn.Dropout(p=0.5, inplace=False),
            'max_pool_3x3': lambda *config: nn.MaxPool2d(kernel_size=3, stride=2,
                            padding=0, dilation=1, return_indices=False, ceil_mode=False),
            'Xmax_pool_3x3':lambda *config: nn.MaxPool2d(kernel_size=3, stride=1,
                            padding=1, dilation=1, return_indices=False, ceil_mode=False),
            #'avg_pool_3x3': lambda *config: nn.AvgPool2d(kernel_size=3, stride=1,
            #                padding=1, ceil_mode=False, count_include_pad=True, divisor_override=None),
            'avg_pool_3x3': lambda *config: nn.AvgPool2d(kernel_size=3, stride=1,
                            padding=1, ceil_mode=False, count_include_pad=False, divisor_override=None),
            'avg_pool_5x5': lambda *config: nn.AvgPool2d(kernel_size=5, stride=3,
                            padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None),
            'ada_avg_pool': lambda *config: nn.AdaptiveAvgPool2d(output_size=(1,1)),
            'flatten':      lambda *config: _Flatten(),
            'fc':           lambda *config: nn.Linear(
                            in_features=config[0], out_features=config[1], bias=True)
        }
        cat = lambda d: _CAT([seq(llist) for _,llist in sorted(d(*args).items(),key=lambda x:x[0])])
        seq = lambda layer_list: nn.Sequential(*[ definition[k](*cfg) if k!='CAT' else cat(*cfg) for k,*cfg in layer_list])
        return seq(self.structure[key])

    def forward(self,x):
        # input: (N,3,299,299) RGB range[0,1]
        y1 = self.first(x)
        y2 = self.second(y1)
        y3 = self.third(y2)
        y4 = self.fourth(y3)
        y4 = self.last1(y4)
        #result = self.last(y4)
        #result,aux = self.last(y4),self.aux(y3)
        return (y1,y2,y3,y4)#(y1,y2,y3,y4,result,aux)


def get_inception_fid_model():
    model = FID_InceptionV3()

    FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
    pretrained = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)

    src = pretrained
    dst = model.state_dict()
    #print(len(src),len(dst))

    for d,s in zip(dst.keys(),key_mapper()):
        #print('o' if dst[d].shape==src[s].shape else 'x',dst[d].shape, src[s].shape)
        dst[d] = src[s]

    model.load_state_dict(dst)
    return model


def key_mapper():
    CB = ['.conv.weight','.bn.weight','.bn.bias','.bn.running_mean',
            '.bn.running_var','.bn.num_batches_tracked']
    tree = {
        'Conv2d_': ['1a_3x3','2a_3x3','2b_3x3','3b_1x1','4a_3x3'],
        'Mixed_5b.branch':['1x1','5x5_1','5x5_2','3x3dbl_1','3x3dbl_2','3x3dbl_3','_pool'], #A
        'Mixed_5c.branch':['1x1','5x5_1','5x5_2','3x3dbl_1','3x3dbl_2','3x3dbl_3','_pool'], #A
        'Mixed_5d.branch':['1x1','5x5_1','5x5_2','3x3dbl_1','3x3dbl_2','3x3dbl_3','_pool'], #A
        'Mixed_6a.branch':['3x3','3x3dbl_1','3x3dbl_2','3x3dbl_3'], #B
        'Mixed_6b.branch':['1x1','7x7_1','7x7_2','7x7_3','7x7dbl_1','7x7dbl_2','7x7dbl_3','7x7dbl_4','7x7dbl_5','_pool'], #C
        'Mixed_6c.branch':['1x1','7x7_1','7x7_2','7x7_3','7x7dbl_1','7x7dbl_2','7x7dbl_3','7x7dbl_4','7x7dbl_5','_pool'], #C
        'Mixed_6d.branch':['1x1','7x7_1','7x7_2','7x7_3','7x7dbl_1','7x7dbl_2','7x7dbl_3','7x7dbl_4','7x7dbl_5','_pool'], #C
        'Mixed_6e.branch':['1x1','7x7_1','7x7_2','7x7_3','7x7dbl_1','7x7dbl_2','7x7dbl_3','7x7dbl_4','7x7dbl_5','_pool'], #C
        'Mixed_6b.branch':['1x1','7x7_1','7x7_2','7x7_3','7x7dbl_1','7x7dbl_2','7x7dbl_3','7x7dbl_4','7x7dbl_5','_pool'], #C
        'Mixed_7a.branch':['3x3_1','3x3_2','7x7x3_1','7x7x3_2','7x7x3_3','7x7x3_4'], #D
        'Mixed_7b.branch':['1x1','3x3_1','3x3_2a','3x3_2b','3x3dbl_1','3x3dbl_2','3x3dbl_3a','3x3dbl_3b','_pool'], #E
        'Mixed_7c.branch':['1x1','3x3_1','3x3_2a','3x3_2b','3x3dbl_1','3x3dbl_2','3x3dbl_3a','3x3dbl_3b','_pool'], #E
    }

    key_list = []
    for k,ll in tree.items():
        for l in ll:
            for s in CB:
                key_list += [k+l+s]
    key_list += ['fc.weight','fc.bias']

    return key_list
