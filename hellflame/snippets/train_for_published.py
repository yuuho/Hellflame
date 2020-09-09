
'''
# 必要なもの
requirements (third party):
    - pyyaml
        ``conda install pyyaml`` or ``pip install pyyaml``


# 実行方法
cd train
python train.py config.yml

# 設定
config.ymlのenv/experimentsとenv/expdir
env
'''

import sys
import os
import signal
import argparse
from importlib import import_module
from pathlib import Path

import yaml


# コマンドライン引数の取得
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',type=str,
                            help='config file path, 設定ファイルへのパス')
    return parser.parse_args()


if __name__ == '__main__':
    # コマンドライン引数の取得
    args = parse_args()

    # 設定ファイルの読み込み
    with Path(args.config).open('r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config['env']['savedir'] = config['env']['log']['exp'] / config['env']['exp_name']

    # GPUの設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        config['env']['cuda_string'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    else:
        config['env']['cuda_string'] = None

    # Trainerの書かれたpythonファイルからTrainerを取得
    sys.path.append(str(config['env']['prog']))
    Trainer = getattr(import_module('trainer.'+config['trainer']['name']),'Trainer')

    # 設定
    print('\033[36m>>> train preparation...\033[0m')
    trainer = Trainer(config)
    
    # 学習
    print('\033[36m>>> train start\033[0m')
    trainer.train()
    print('\033[36m>>> train end\033[0m')
    
    # 強制終了
    os.kill(os.getpid(), signal.SIGTERM)

