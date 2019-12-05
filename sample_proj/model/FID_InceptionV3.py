import torch
import torch.nn as nn
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


# FID用InceptionV3モデルの定義
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
        #        ['ada_avg_pool'],['flatten'],['fc', 768,1000]           ], FIDの計算をするだけなので Auxiliary Classifier はいらない
        'inception_D':[['CAT',lambda:{
            0:[ ['conv_1x1',      768, 192],['BN', 192],['relu'],
                ['conv_3x3_down', 192, 320],['BN', 320],['relu']],
            1:[ ['conv_1x1',      768, 192],['BN', 192],['relu'],
                ['conv_1x7_flat', 192, 192],['BN', 192],['relu'],
                ['conv_7x1_flat', 192, 192],['BN', 192],['relu'],
                ['conv_3x3_down', 192, 192],['BN', 192],['relu']],
            2:[ ['max_pool_3x3']                                ] }]],
        #'inception_E':[['CAT',lambda in_ch:{                           ↓ FID用にpooling層の変更が出来るようにした
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
        'last': [['ada_avg_pool'],['dropout'],['flatten'],['fc',2048,1008]] # <- FID用は 1008クラス
    }

    def __init__(self):
        super().__init__()
        self.first = self._make_block('first')
        self.second = self._make_block('second')
        self.third = nn.Sequential(*[self._make_block(*arg) for arg in
                            [('inception_A',192,32),('inception_A',256,64),('inception_A',288,64),('inception_B',),
                            ('inception_C',128),('inception_C',160),('inception_C',160),('inception_C',192)]])
        #self.aux = self._make_block('aux')     ← FIDの計算をするだけなので Auxiliary Classifier はいらない
        self.fourth = nn.Sequential(*[self._make_block(*arg) for arg in
                            # [('inception_D',),('inception_E',1280),('inception_E',2048)]])    ↓ FID用にpooling層の変更が出来るようにした
                            [('inception_D',),('inception_E',1280,'avg'),('inception_E',2048,'Xmax')]])
        self.last = self._make_block('last')

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
                            padding=1, dilation=1, return_indices=False, ceil_mode=False),  # FID用は最後のInceptionEのpoolはmax pool
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
        '''
        input:
            - shape: (N,3,299,299), range: [-1.0, +1.0], ch_order: 'RGB'
        output:
            - shape: (N, 2048, 1, 1)
        '''
        y1 = self.first(x)
        y2 = self.second(y1)
        y3 = self.third(y2)
        y4 = self.fourth(y3)
        result = self.last[0](y4) # adaptive average pool後のベクトルを利用する
        #result,aux = self.last(y4),self.aux(y3) # auxiliary classifierは利用しない
        return result #(result, aux)


# モデルの構築と事前学習済みの重みの読み込みを行う
def get_inception_fid_model():
    model = FID_InceptionV3()

    FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
    pretrained = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)

    dst_key_list = model.state_dict().keys()
    assert len(pretrained.keys())==len(dst_key_list) # 元のモデルとここで定義されているモデルは完全に一致する

    new_dict = {dst: pretrained[src] for dst,src in zip(dst_key_list, ordered_src_key_list())}
    model.load_state_dict(new_dict)
    return model


# 読み込み元のキーでここで定義されているモデルの順に並べる
def ordered_src_key_list():
    # 畳み込みブロックのpostfixリスト
    postfixes = ['.conv.weight','.bn.weight','.bn.bias','.bn.running_mean','.bn.running_var','.bn.num_batches_tracked']
    tree = {
        'Conv2d_'        :['1a_3x3','2a_3x3','2b_3x3','3b_1x1','4a_3x3'], # feature extractor
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
    return [k+l+s for k,ll in tree.items() for l in ll for s in postfixes] + ['fc.weight','fc.bias']



'''
# README:

PyTorch版FID計測プログラム

## 使用方法
画像の入ったディレクトリのパスを渡すだけ．
```
python eval_fid.py hoge/ fuga/
```
- 渡すディレクトリは二つ (順不同)
    - 本物画像の入ったディレクトリ
    - 偽物画像の入ったディレクトリ
- 利用される画像
    - ディレクトリの配下にある全画像 (再帰的に探索される)
    - ファイル名の拡張子が .png .jpg .PNG .jpeg .JPG のいずれかである画像
- その他オプション
    - ``-b, --batch_size`` Inceptionモデルに通すバッチサイズ
    - ``-p, --process`` データ読み込みで並列に使うプロセス数

# ****************** 注意: FID計測用のInceptionV3は通常のInceptionV3と一部異なる ***************************
FID計測の際のInceptionV3は公式実装と少し異なる部分がある．
- Szegedyらが提案したInceptionV3モデル
- HeuselらがFIDを提案したときに実装したInceptionV3モデル (公式実装(TensorFlow)がGitHubで公開されている)

本来のモデル・FID用モデルの両方共PyTorchで再現実装が公開されている．
PyTorch版実装をみてわかったHeuselらが間違っていた箇所について記しておく．
論文などにFIDを記載する際は Heusel らの実装に従って計測したものを載せる必要がある．

## 比較したモデル
- PyTorch公式で画像分類用に公開されている InceptionV3 のPyTorchモデル
    [リンク - torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py)
- 第三者がFID計測用に公開している InceptionV3 のPyTorchモデル
    [リンク - pytorch-fid](https://github.com/mseitzer/pytorch-fid)
    FID提案論文の公式実装(https://github.com/bioinf-jku/TTUR)と同じ重みを利用していて，
    誤差が非常に少ないのでスコアを論文に載せるのはおそらく大丈夫．

## InceptionV3公式実装からの変更点
### 根本的な変更点
- torchvision版モデルはミニバッチ毎にRGB各チャンネルを
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] で
    正規化するが，FID用は分散での正規化を行わない
- 通常のInceptionV3は入力のrangeが[0,1]だがFID用は[-1,+1]

参考) 本来の実装のrangeについて言及されている
    https://pytorch.org/docs/stable/torchvision/models.html
参考) FID用実装のrangeについて言及されている
    https://github.com/mseitzer/pytorch-fid/issues/17#issuecomment-498538838


### 層単位の変更
1. 最後の inception_E のPoolingが average pool から max poolに
2. 3x3 average pooling の count_include_pad は TrueからFalseに
3. PyTorch Hubのものは出力が1000クラスだがこちらは1008クラス

参考1) https://github.com/mseitzer/pytorch-fid/blob/master/inception.py#L302-L305
参考2) https://github.com/mseitzer/pytorch-fid/blob/master/inception.py#L208-L209
参考2) https://github.com/mseitzer/pytorch-fid/blob/master/inception.py#L269-L270
参考3) https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py#L67
参考3) https://github.com/mseitzer/pytorch-fid/blob/master/inception.py#L175

　
### 使用しないブロック
- Auxiliary Classifier

'''

if __name__ == '__main__':

    import re
    import argparse
    from pathlib import Path

    import tqdm
    import numpy as np
    import cv2
    import scipy.linalg

    # コマンドライン引数の受け取り
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('path', type=str, nargs=2, help=('画像の入ったディレクトリへのパス2つ'))
        parser.add_argument('-b','--batch_size', type=int, default=64, help='バッチサイズ')
        parser.add_argument('-p','--processes', type=int, default=0, help='データ読み込みの並列数')
        return parser.parse_args()

    # 画像データセット
    class Images(torch.utils.data.Dataset):
        def __init__(self, dir_path):
            super().__init__()
            self.dir_path = dir_path
            self.image_paths = [item for item in sorted(dir_path.glob('*'))
                        if re.search(r'\.(png|PNG|jpeg|jpg|JPG)$',str(item)) and not item.is_dir()]

        def __getitem__(self, idx):
            img = cv2.imread(str(self.image_paths[idx]),cv2.IMREAD_COLOR)[:,:,::-1]
            resized = cv2.resize(img,(299,299))
            return torch.from_numpy(resized.transpose(2,0,1).astype(np.float32) / 255.0 * 2.0 -1.0)

        def __len__(self):
            return len(self.image_paths)

        def has_run_cache(self):
            return (self.dir_path/'fid.npz').exists()
        def get_run_cache(self):
            result_set = np.load(str(self.dir_path/'fid.npz'))
            return result_set['mu'], result_set['sigma']
        def set_run_cache(self,mu,sigma):
            np.savez(str(self.dir_path/'fid.npz'),mu=mu,sigma=sigma)


    # 画像データセットについて特徴量の平均分散を計算する
    def calc_mu_sigma(model, device, img_iter):
        # 計算済み結果があればそれを返す
        if img_iter.dataset.has_run_cache():
            return img_iter.dataset.get_run_cache()
        # なければ特徴量計算して計算
        model.eval()
        with torch.no_grad():
            features = torch.cat([model(minibatch.to(device)).view(-1,2048).cpu()
                                for minibatch in tqdm.tqdm(img_iter)]).numpy()
        mu, sigma = np.mean(features, axis=0), np.cov(features, rowvar=False)
        img_iter.dataset.set_run_cache(mu,sigma)
        return mu, sigma

    # コマンドライン引数の受け取り
    args = parse_args()

    # モデルの構築・学習済みの重みの読み込み
    model = get_inception_fid_model()
    print('model constructed')

    # GPUの設定
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:',device)
    model.to(device)

    # 画像データの読み込み
    img_iterR = torch.utils.data.DataLoader( Images(Path(args.path[0])),
                        batch_size=args.batch_size, shuffle=False,num_workers=args.processes)
    img_iterF = torch.utils.data.DataLoader( Images(Path(args.path[1])),
                        batch_size=args.batch_size, shuffle=False,num_workers=args.processes)

    # FIDの計算
    muR, sigmaR = calc_mu_sigma(model, device, img_iterR)
    muF, sigmaF = calc_mu_sigma(model, device, img_iterF)
    del model
    assert muR.shape==(2048,), sigmaR.shape==(2048,2048)
    assert muF.shape==(2048,), sigmaF.shape==(2048,2048)

    eps = 1e-6
    diff = muR-muF

    # 行列の平方根の計算
    covmean, _ = scipy.linalg.sqrtm(sigmaR@sigmaF, disp=False)

    # nanやinfがあるなら
    if not np.isfinite(covmean).all():
        print('非正則行列なので %lf を対角要素に足した'% eps)
        # 対角要素にオフセットを付加して計算し直す
        offset = np.eye(sigmaR.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigmaR+offset)@(sigmaF+offset))

    # 複素数が含まれているなら
    if np.iscomplexobj(covmean):
        # 対角要素の虚数成分が全て0なのはエラー
        assert not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3), \
                    'Imaginary component {}'.format(np.max(np.abs(covmean.imag)))
        # 実成分を取り出す
        covmean = covmean.real

    fid_score = diff@diff + np.trace(sigmaR) + np.trace(sigmaF) - 2*np.trace(covmean)

    print(fid_score)

'''
# 元の実装と出力に違いがないか確認する方法


同じディレクトリに以下の3つのファイルを置いて python test_output.py とする．
- このファイルを右の名前で ``eval_fid.py``
- 元の実装のファイル ``inception.py`` (https://github.com/mseitzer/pytorch-fid/blob/master/inception.py)
- 以下を書いたファイル ``test_output.py``
```
import torch
import numpy as np

from inception import InceptionV3
from eval_fid import get_inception_fid_model


if __name__=='__main__':

    origin_model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    print('origin')
    my_model = get_inception_fid_model()
    print('mine')

    xo = torch.randn((4,3,299,299))
    xm = xo.clone()

    print('input same: ',np.allclose(xo.numpy(),xm.numpy()))
    print('forward')
    with torch.no_grad():
        yo = origin_model(xo)[0]
        ym = my_model(xm*2-1)

    print('output same: ',np.allclose(yo.numpy(),ym.numpy()))
    print('input same: ',np.allclose(xo.numpy(),xm.numpy()))

    print(origin_model.resize_input)
    print(origin_model.normalize_input)
```
'''
