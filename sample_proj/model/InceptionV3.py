import torch
import torch.nn as nn
import torch.nn.functional as F


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


class InceptionV3(nn.Module):

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
        'aux':[ ['avg_pool_5x5'],
                ['conv_1x1', 768, 128],['BN', 128],['relu'],
                ['conv_5x5', 128, 768],['BN', 768],['relu'],
                ['ada_avg_pool'],['flatten'],['fc', 768,1000]           ],
        'inception_D':[['CAT',lambda:{
            0:[ ['conv_1x1',      768, 192],['BN', 192],['relu'],
                ['conv_3x3_down', 192, 320],['BN', 320],['relu']],
            1:[ ['conv_1x1',      768, 192],['BN', 192],['relu'],
                ['conv_1x7_flat', 192, 192],['BN', 192],['relu'],
                ['conv_7x1_flat', 192, 192],['BN', 192],['relu'],
                ['conv_3x3_down', 192, 192],['BN', 192],['relu']],
            2:[ ['max_pool_3x3']                                ] }]],
        'inception_E':[['CAT',lambda in_ch:{
            0:[ ['conv_1x1',     in_ch, 320],['BN', 320],['relu']],
            1:[ ['conv_1x1',     in_ch, 384],['BN', 384],['relu'],['CAT',lambda _:{
                    0:[ ['conv_1x3_flat', 384, 384],['BN', 384],['relu']],
                    1:[ ['conv_3x1_flat', 384, 384],['BN', 384],['relu']],  }]],
            2:[ ['conv_1x1',     in_ch, 448],['BN', 448],['relu'],
                ['conv_3x3_flat',  448, 384],['BN', 384],['relu'],['CAT',lambda _:{
                    0:[ ['conv_1x3_flat', 384, 384],['BN', 384],['relu']],
                    1:[ ['conv_3x1_flat', 384, 384],['BN', 384],['relu']],  }]],
            3:[ ['avg_pool_3x3'],
                ['conv_1x1',     in_ch, 192],['BN', 192],['relu']]      }]],
        'last': [['ada_avg_pool'],['dropout'],['flatten'],['fc',2048,1000]]
    }

    def __init__(self):
        super().__init__()
        self.first = self._make_block('first')
        self.second = self._make_block('second')
        self.third = nn.Sequential(*[self._make_block(*arg) for arg in
                            [('inception_A',192,32),('inception_A',256,64),('inception_A',288,64),('inception_B',),
                            ('inception_C',128),('inception_C',160),('inception_C',160),('inception_C',192)]])
        self.aux = self._make_block('aux')
        self.fourth = nn.Sequential(*[self._make_block(*arg) for arg in
                            [('inception_D',),('inception_E',1280),('inception_E',2048),('last',)]])

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
            'avg_pool_3x3': lambda *config: nn.AvgPool2d(kernel_size=3, stride=1,
                            padding=1, ceil_mode=False, count_include_pad=True, divisor_override=None),
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
        y1 = self.first(x)
        y2 = self.second(y1)
        y3 = self.third(y2)
        y4,aux = self.fourth(y3),self.aux(y3)
        return (y1,y2,y3,y4,aux)


def calc_inception_score():

    length = 500 # num images

    data = np.zeros((length,1000),dtype=np.float32)

    for i,im in enumerate(images):
        # im is (1,C,299,299) Tensor [0.0,+1.0]
        _1,_2,_3,pred,_4 = inceptionv3(im)
        logits = F.softmax(pred,dim=1)
        data[i] = logits[0].cpu().numpy()
    
    py = data.mean(axis=0)[None,:]
    # assert py.shape==(1,1000)
    dkl = (data*np.log(data/py)).sum(axis=1)
    # assert dkl.shape==(length,)
    inception_score      = np.exp(dkl.mean())

    return inception_score




if __name__ == '__main__':

    # python3.7は辞書挿入順が保存される
    # ここではその機能を利用しているため3.7でないと動かない
    dst_model = InceptionV3()
    dst_dict = dst_model.state_dict()
    dst_keys = list(dst_dict.keys())

    # このモデルを読み込む 'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
    from torchvision.models.inception import inception_v3
    
    # transform_inputは使用しない
    src_model = inception_v3(pretrained=True,transform_input=False)
    src_dict = src_model.state_dict()
    src_keys = list(src_dict.keys())

    # 転送
    for k1,k2 in zip(dst_keys,src_keys):
        #print('%s%s%s'%(k1,' '*(52-len(k1)),k2))
        dst_dict[k1] = src_dict[k2]
    dst_model.load_state_dict(dst_dict)
    
    print('model constructed.')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dst_model.to(device)
    src_model.to(device)
    

    print('>>> eval mode output analysis >>>')

    # trainingのときはランダムなDropoutのせいで正確に出力を比較することはできない
    dst_model.eval()
    src_model.eval()

    # 同じ入力を用意して比較する (torchvision版はバッチサイズ1では動作しない)
    s1 = torch.randn((4,3,299,299),dtype=torch.float32)
    s1 = s1 - s1.min()
    s1 = s1 / s1.max()
    print(s1.min(),s1.max())
    d1 = s1.clone()
    s1 = s1.to(device)
    d1 = d1.to(device)

    # 出力値の比較
    *_,D,daux = dst_model(d1)
    S = src_model(s1)
    print('    shape: dst:',D.shape,', src:',S.shape)
    v = D-S
    print('    element-wise sub-value: max:',v.max().item(),', min:',v.min().item())
    print('    soft max or not:',S.sum(axis=1))
    print('    soft max or not2:',F.softmax(S,dim=1).sum(axis=1))
    print('    transform inpu mode:',src_model.transform_input)

    print('>>> train mode output analysis >>>')

    # 学習時は同じ値にはならないので形状のみ比較
    dst_model.train()
    src_model.train()

    s2 = torch.randn((4,3,299,299),dtype=torch.float32)
    d2 = s1.clone()
    s2 = s1.to(device)
    d2 = d1.to(device)

    *_,D,daux = dst_model(d2)
    S,saux = src_model(s2)
    print('    shape: dst_aux:',daux.shape,', src_aux:',saux.shape)
    
