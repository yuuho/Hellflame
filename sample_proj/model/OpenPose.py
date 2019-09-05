import torch
import torch.nn as nn


class Model(nn.Module):
    structure = {
        'VGG19_half': [ ['conv3x3',  3, 64],    ['relu'],
                        ['conv3x3', 64, 64],    ['relu'],
                        ['pool1/2'],
                        ['conv3x3', 64,128],    ['relu'],
                        ['conv3x3',128,128],    ['relu'],
                        ['pool1/2'],
                        ['conv3x3',128,256],    ['relu'],
                        ['conv3x3',256,256],    ['relu'],
                        ['conv3x3',256,256],    ['relu'],
                        ['conv3x3',256,256],    ['relu'],
                        ['pool1/2'],
                        ['conv3x3',256,512],    ['relu'],
                        ['conv3x3',512,512],    ['relu']    ],
        'stage0_MAP': [ ['conv3x3',512,256],    ['relu'],
                        ['conv3x3',256,128],    ['relu']    ],
        'stage1_PAF': [ ['conv3x3',128,128],    ['relu'],
                        ['conv3x3',128,128],    ['relu'],
                        ['conv3x3',128,128],    ['relu'],
                        ['conv1x1',128,512],    ['relu'],
                        ['conv1x1',512, 38],                ],
        'stage1_PCM': [ ['conv3x3',128,128],    ['relu'],
                        ['conv3x3',128,128],    ['relu'],
                        ['conv3x3',128,128],    ['relu'],
                        ['conv1x1',128,512],    ['relu'],
                        ['conv1x1',512, 19],                ],
        'stageN_PAF': [ ['conv7x7',185,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv1x1',128,128],    ['relu'],
                        ['conv1x1',128, 38],                ],
        'stageN_PCM': [ ['conv7x7',185,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv7x7',128,128],    ['relu'],
                        ['conv1x1',128,128],    ['relu'],
                        ['conv1x1',128, 19],                ]
    }

    def _make_layer(self,key):
        definition = {
            'pool1/2':  lambda *config: nn.MaxPool2d(
                            kernel_size=2, stride=2, padding=0),
            'conv1x1':  lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=1, stride=1, padding=0),
            'conv3x3':  lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=3, stride=1, padding=1),
            'conv7x7':  lambda *config: nn.Conv2d(
                            in_channels=config[0], out_channels=config[1],
                            kernel_size=7, stride=1, padding=3),
            'relu':     lambda *config: nn.ReLU(inplace=True)
        }
        return nn.Sequential(*[ definition[k](*cfg) for k,*cfg in self.structure[key]])

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            self._make_layer('VGG19_half'),
            self._make_layer('stage0_MAP')  )
        self.paf_branch = nn.ModuleList([
            self._make_layer(key) for key in ['stage1_PAF']+['stageN_PAF']*5 ])
        self.pcm_branch = nn.ModuleList([
            self._make_layer(key) for key in ['stage1_PCM']+['stageN_PCM']*5 ])

    def forward(self, x):
        base = self.feature_extractor(x)

        pafs, pcms = [], []
        for i,( m_paf, m_pcm) in enumerate(zip(self.paf_branch, self.pcm_branch)):
            f = base if i==0 else torch.cat([pafs[-1],pcms[-1],base],1)
            pafs += [m_paf(f)]
            pcms += [m_pcm(f)]

        return pafs, pcms

