
'''
exp_nameを指定して出力

含まれるファイル
- publish.zip/
    - train/
        - config.yml
        - train.py
        - program/

実行方法
cd train
python train.py config.yml
'''

import re
import os
from pathlib import Path
from datetime import datetime, timedelta
import zipfile

import yaml

from hellfire.services.Service import Service
from hellfire.snippets.train_for_published import __file__ as train_file


class PublishService(Service):
    command_name = 'publish'
    help_text = '設定ファイルの読み取り内容を書き出す'

    def __init__(self,subparsers):
        super().__init__(subparsers)

    # subparserの登録
    def register_parser(self,parser):
        # 必須項目
        parser.add_argument('exp_name',type=str,
                            help='実験名')
        # コマンドライン引数からパスを指定する場合
        parser.add_argument('--experiments','-e', type=str,
                            default=None,
                            dest='exp', help='実験データ置き場')

    # エントリーポイント
    def handler_function(self,args):
        print('\033[36m::: >>> Enter: PublishService\033[0m')

        # 実験データベースのディレクトリ特定
        exp_path = args.exp if args.exp is not None else os.environ.get('MLEXP')
        if (exp_path is None) or (not Path(exp_path).exists()):
            raise Exception('experiment directory is not exist : '+str(exp_path))
        exp_path = Path(exp_path)

        # 設定ファイルの存在確認と読み込み
        config_paths = [item for item in sorted((exp_path/args.exp_name).glob('*'))
                if re.search(r'hellfire_config_.*\.yml$',str(item)) and item.is_file()]
        if len(config_paths) == 0:
            raise Exception('there is no config file : '+str(exp_path/args.exp_name))
        with config_paths[-1].open('r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)

        savedir = config['env']['savedir']
        # 設定ファルに読み込んだ内容を追加
        config['env'] = {
            'prog' : Path('program'),               # プログラムのルートディレクトリ
            'data' : config['env']['data'],         # データセットのルートディレクトリ
            'tmp'  : config['env']['tmp'],          # 計算キャッシュやglobal_writerに使う
            #'savedir' : config['env']['savedir'],   # この実験の保存ディレクトリ
            'is_continue': False,                   # 続きからかどうか
            'exp_name': args.exp_name,                   # 実験の名前
            #'machine': machine,             # マシン名
            #'cuda_string': cuda_string,     # CUDA_VISIBLE_DEVICESに設定された文字列
            'log': {
                'exp'  : config['env']['log']['exp'],
            #    'config': config_path,     # 設定ファイルのパス
            }
        }
        del config['environ']

        # 設定ファイルの保存
        timestamp = (datetime.now()+timedelta(milliseconds=1)).strftime('%Y%m%d_%H%M_%S.%f')[:-3]
        publish_file_path = savedir/('hellfire_publish_config_%s.yml'%(timestamp))
        with publish_file_path.open('w') as f:             # 読み取り結果の保存
            yaml.dump(config,f)

        # zipファイルにまとめる
        zip_file_path = savedir/('hellfire_publish_%s.zip'%(timestamp))
        prefix = 'train/'
        with zipfile.ZipFile(str(zip_file_path), 'w') as new_zip:
            # 設定ファイル
            new_zip.write(str(publish_file_path), arcname=prefix+'config.yml')
            
            # プログラムファイル
            scripts = [item for item in list((savedir/'program').rglob('*'))
                        if re.search(r'\.py$',str(item)) and item.is_file()]
            for script in scripts:
                arcname = str(script.resolve()).replace(str(savedir),'')
                new_zip.write(str(script), arcname=prefix+arcname)

            # 学習コード
            new_zip.write(train_file, arcname=prefix+'train.py')
        print('    publish : ',str(zip_file_path))

        # 設定ファイルを削除
        publish_file_path.unlink()

        print('\033[36m::: <<< Exit: PublishService\033[0m')
        return 0

