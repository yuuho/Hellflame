'''
このファイルの構成
- FeatureExtractor
- ThinUnetModule
- M2Det
'''

import re

import torch
import torch.nn as nn
import torchvision
from torchvision.models.vgg import model_urls


# VGG16ベース
class FeatureExtractor(nn.Module):

    structure = {
        'vgg16':       [['conv_3x3',3,64],          ['relu'],
                        ['conv_3x3',64,64],         ['relu'],
                        ['maxpool_down'], # 1/2
                        ['conv_3x3',64,128],        ['relu'],
                        ['conv_3x3',128,128],       ['relu'],
                        ['maxpool_down'], # 1/4
                        ['conv_3x3',128,256],       ['relu'],
                        ['conv_3x3',256,256],       ['relu'],
                        ['conv_3x3',256,256],       ['relu'],
                        ['maxpool_down'], # 1/8
                        ['conv_3x3',256,512],       ['relu'],
                        ['conv_3x3',512,512],       ['relu'],
                        ['conv_3x3',512,512],       ['relu']                        ],
        'additional':  [['maxpool_down'], # 1/16
                        ['conv_3x3',512,512],       ['relu'],
                        ['conv_3x3',512,512],       ['relu'],
                        ['conv_3x3',512,512],       ['relu'], # ここまでVGG16
                        ['maxpool_flat'], # 1/16
                        ['conv_dilation',512,1024], ['relu'],
                        ['conv_1x1',1024,1024],     ['relu'],
                        ['conv_1x1',1024,512],      ['batch_norm',512], ['relu'],
                        ['upsample']                                                ], # 1/8
        'reducer':     [['conv_3x3',512,256],       ['batch_norm',256], ['relu']    ]}

    # 設定通りにBlock(Sequential)を作る
    def _make_block(self,key):
        definition = {
            'conv_3x3':     lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=3, stride=1, padding=1)
            'conv_1x1':     lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=1, stride=1, padding=0)
            'conv_dilation':lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=3, stride=1, padding=6, dilation=6)
            'batch_norm':   lambda *config: nn.BatchNorm2d(
                            num_features=config[0],
                            eps=1e-5, momentum=0.01, affine=True)
            'relu':         lambda *config: nn.ReLU(inplace=True)
            'upsample':     lambda *config: nn.Upsample(scale_factor=2, mode='nearest')
            'maxpool_down': lambda *config: nn.MaxPool2d(kernel_size=2, stride=2)
            'maxpool_flat': lambda *config: nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])

    def load_pretrained(self,checkpoint_dir_path):
        # 事前学習済みモデルの保存場所を作る
        checkpoint_dir_path.mkdir(parents=True,exist_ok=True)

        # ダウンロード
        vgg_dict = torch.hub.load_state_dict_from_url(model_urls['vgg16'],
                            model_dir=str(checkpoint_dir_path.resolve()))

        # 事前学習済みの重みの一部を挿入
        my_dict = self.state_dict()
        for k,v in filter(lambda kv: 'features.' in kv[0], vgg_dict.items()):
            num = int(re.sub(r'\D', '',k))
            key = k.replace('features','vgg') if num<24 \
                    else 'additional.'+str(num-23)+k[11:]
            my_dict[key] = v
        self.load_state_dict(my_dict)

    def __init__(self):
        super().__init__()
        self.vgg = self._make_block('vgg16')
        self.reducer = self._make_block('reducer')
        self.additional = self._make_block('additional')

    def forward(self,x):
        base = self.vgg(x)
        y = torch.cat([self.reducer(base), self.additional(base)],1)
        return y


# UNet dyamic
class ThinUnetModule(nn.Module):

    # 各塊の定義
    structure = {
        '1st_encode_conv': [['conv',        'input_channel','input_channel',3,2,1   ],
                            ['batch_norm',  'input_channel'                         ],
                            ['relu'                                                 ]],
        'encode_conv':     [['conv',        'input_channel','input_channel',3,2,1   ],
                            ['batch_norm',  'input_channel'                         ],
                            ['relu'                                                 ]],
        'decode_conv':     [['conv',        'input_channel','input_channel',3,1,1   ],
                            ['batch_norm',  'input_channel'                         ],
                            ['relu'                                                 ],
                            ['upsample'                                             ]],
        'smooth_conv':     [['conv',        'input_channel','out_channel',1,1,0     ],
                            ['batch_norm',  'out_channel'                           ],
                            ['relu'                                                 ]]
    }

    # 設定通りにBlock(Sequential)を作る
    def _make_block(self,key):
        definition = {
            'conv'      : lambda *config: nn.Conv2d(
                            in_channels=getattr(self,config[0]), out_channels=getattr(self,config[1]),
                            kernel_size=config[2], stride=config[3], padding=config[4]),
            'batch_norm': lambda *config: nn.BatchNorm2d(
                            num_features=getattr(self,config[0]), eps=1e-5, momentum=0.01, affine=True),
            'relu'      : lambda *config: nn.ReLU(inplace=True),
            'upsample'  : lambda *config: nn.Upsample(scale_factor=2, mode='nearest')
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])

    # 指定されたとおりにモジュールを登録していく
    def __init__(self, in_channels=128, out_channels=256, scales=6):# calc_channels=256, scales=6):
        '''
        引数:
            - in_channels: 入力する特徴マップのチャンネル数
            - out_channels: 出力する特徴マップのチャンネル数
            - scales: 出力する特徴マップの解像度の種類の個数 6なら(1/1, 1/2, 1/4, 1/8, 1/16, 1/32)の6つ
        制約
            - 入力する特徴マップの短辺は 2^(scales-1) 以上
        '''
        super().__init__()

        # チャンネルの情報と画像サイズの個数
        self.input_channel = in_channels
        #self.calc_channel = calc_channels
        self.out_channel = out_channels
        self.scales = scales

        # ネットワーク構造の定義
        self.encoder = nn.ModuleList([self._make_block(key)
                            for key in ['1st_encode_conv']+['encode_conv']*(scales-2)])
        self.decoder = nn.ModuleList([self._make_block(key)
                            for key in ['decode_conv']*(scales-1)])
        self.smoother = nn.ModuleList([self._make_block(key)
                            for key in ['smooth_conv']*scales])

    def forward(self, x):
        # 論文版での実装
        e = [x] + [(l.append(f(l[-1])),l[-1])[-1]
                        for l in [[x]] for f in self.encoder]
        d = e[-1:] + [(l.append(f(l[-1])+s),l[-1])[-1]
                        for l in [[e[-1]]] for f,s in zip(self.decoder,e[-2::-1])]
        s = [sm(v) for sm,v in zip(self.smoother,d)]
        return s


class M2Detector(nn.Module):
    '''
    M2Det:

    概要:
        1. 特徴抽出をする
        2. 直列繋ぎの複数UNetのデコーダ部分の出力を画像サイズごとにすべて保存
        3. サイズごとに連結
        4. チャンネルAttentionですべてのサイズのものを合体
        5. SSDと同じく各デフォルトボックスに対してクラス確率とボックス回帰パラメータを算出

    image(N,3,256,256) -> FeatureExtractor -> feature(N,768,64,64)
    feature -> leacher
    '''

    def __init__(self):
        super().__init__()

        # 各種設定値
        ## 出力次元に影響するパラメータ : 座標次元数，クラス数，アスペクト比の個数，スケールの個数 =(P,L,A,S)
        self.num_location_parameters, self.num_classes = 4, 2
        self.num_aspect_ratios, self.num_scales = 6, 6
        ## ネットワーク計算過程のパラメータ : 特徴抽出器の出力チャンネル数，TUMの個数，TUMの計算過程チャンネル数
        ext_out_ch = 768 # const
        tum_num_levels = 8
        tum_in_ch = 256
        tum_out_ch = tum_in_ch//2

        # ネットワーク構造の定義

        ## 特徴抽出器
        self.feature_extractor = FeatureExtractor()

        ## MLFPN: Multi-Level Feature Pyramid Network
        self.preprocesser = nn.ModuleList([ nn.Conv2d(
            in_channels=ext_out_ch, out_channels=ch,
            kernel_size=1,stride=1, padding=0)
            for ch in [tum_in_ch]+[tum_out_ch]*(tum_num_levels-1) ])
        self.concatinator = [lambda x,y:x]+[lambda x,y:torch.cat([x,y],1)]*(tum_num_levels-1)
        self.thin_unets = nn.ModuleList([ThinUnetModule(
            in_channels=tum_in_ch, out_channels=tum_out_ch, scales=self.num_scales)
            for i in range(tum_num_levels)])

        ## SFAM: Scale-wise Feature Aggregation Module
        self.squeeze_seg = None

        ## 出力層
        self.location_convs = nn.ModuleList([ nn.Conv2d(
            tum_out_ch*tum_num_levels, self.num_aspect_ratios*self.num_location_parameters,
            kernel_size=3,stride=1,padding=1) for i in range(self.num_scales) ])
        self.config_convs = nn.ModuleList([ nn.Conv2d(
            tum_out_ch*tum_num_levels, self.num_aspect_ratios*self.num_classes,
            kernel_size=3,stride=1,padding=1) for i in range(self.num_scales) ])


    def forward(self,x):
        # 初期形状
        B, C, H, W = x.shape

        # ネットワークの入力に対する制限
        ## H == 8 * 2**(self.num_scales-1)ならギリギリ動きはする
        ## H == 8 * 2**self.num_scales が望ましい．(UNetの最下層でも2x2のサイズは欲しいので)
        _limit = 8*(2**self.num_scales)
        assert H%_limit==0, 'invalid Height : H may be multiple of %d'%(_limit)
        assert W%_limit==0, 'invalid Width : W may be multiple of %d'%(_limit)

        # 特徴抽出をする (B,C_3,H,W) -> (B,C_768,H/8,W/8)
        feature = self.feature_extractor(x)

        # pyramidsは2次元listで各要素はTensor
        pyramids = [(l.append( u( c( p(feature),l[-1][-1] ) ) ),l[-1])[1]
                    for l in [[[None]]]
                    for p,c,u in zip(self.preprocesser,self.concatinator,self.thin_unets)]

        # 転置して連結 level,scale -> scale,level -> scales
        scales = [torch.cat(list(x),1) for x in zip(*pyramids)]

        ##### (TODO) SFAM #####

        '''
        B: バッチサイズ = 4
        L: クラス数(ラベル数) = 81
        P: ボックスパラメータ = 4
        A: アスペクト比 = 6
        S: スケール個数 = 6
        '''
        P, L = self.num_location_parameters, self.num_classes
        # ボックス回帰パラメータ算出
        ## [(B,C,H,W) -> (B,AP,H,W)]xS
        locs = [f(x) for f,x in zip(self.location_convs,scales)]
        ## [(B,AP,H,W)->(B,H,W,AP)->(B,HWA,P)]xS -> (B,S_HWA,P)
        loc = torch.cat([x.permute(0,2,3,1).contiguous().view(B,-1,P) for x in locs],1)
        # クラス確率算出
        ## [(B,C,H,W) -> (B,AL,H,W)]xS
        confs = [f(x) for f,x in zip(self.config_convs,scales)]
        ## [(B,AL,H,W)->(B,H,W,AL)->(B,HWA,L)]xS -> (B,S_HWA,L)
        conf = torch.cat([x.permute(0,2,3,1).contiguous().view(B,-1,L) for x in confs],1)

        return (loc, conf)


def Model(data_dir_path,**params):
    vgg_dir_path = data_dir_path / params['pretrained_dir']

    model = M2Detector()
    model.feature_extractor.load_pretrained(vgg_dir_path)

    return model
