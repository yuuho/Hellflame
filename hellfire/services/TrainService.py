
import sys
import os
import shutil
from argparse import Namespace
from pathlib import Path
from importlib import import_module
from datetime import datetime, timedelta

import yaml

from hellfire.services.Service import Service


class TrainService(Service):
    command_name = 'train'
    help_text = '設定ファイルの内容を元に学習を行う'

    def __init__(self,subparsers):
        super().__init__(subparsers)

    # subparserの登録
    def register_parser(self,parser):
        # 必須項目
        parser.add_argument('--config','-c',type=str,
                            required=True,
                            dest='config', help='config file path, 設定ファイルへのパス')
        parser.add_argument('--name','-n',type=str,
                            required=True,
                            dest='name', help='experiment name, 実験名')
        # コマンドライン引数からパスを指定する場合
        parser.add_argument('--programs','-p', type=str,
                            default=None,
                            dest='prog', help='ソースコードのディレクトリ')
        parser.add_argument('--datasets','-d', type=str,
                            default=None,
                            dest='data', help='データセット置き場')
        parser.add_argument('--experiments','-e', type=str,
                            default=None,
                            dest='exp', help='実験データ置き場')
        parser.add_argument('--temporary','-t', type=str,
                            default=None,
                            dest='tmp', help='一時ファイル置き場')
        # 強制的に最初から
        parser.add_argument('--force-clear', action='store_true',
                            dest='clear', help='強制的に最初から')
        # すべて yes オプション
        parser.add_argument('--yes','-y', action='store_true',
                            dest='yes', help='ディレクトリの作成などすべて自動でyesを入力')
        # GPU指定オプション
        parser.add_argument('--gpu','-g', type=str,
                            default=None,
                            dest='gpu', help='GPUを指定する場合')

    # エントリーポイント
    def handler_function(self,args):
        print('\033[36m::: >>> Enter: TrainService\033[0m')

        # 設定ファイルの存在確認と読み込み
        config_path = Path(args.config)
        if not config_path.exists():
            raise Exception('there is not such a config file : '+str(config_path))
        else:
            with config_path.open('r') as f:
                config = yaml.load(f,Loader=yaml.FullLoader)

        # マシンの確認
        if 'machine' in config['environ'] and config['environ']['machine']!=os.uname()[1]:
            raise Exception('    setting name is not this machine : %s'%(config['environ']['machine']))
        machine = os.uname()[1]

        # GPU設定の確認
        cuda_string = self.get_device_settings(args,config) # str, list of int
        # cpu
        # 0,1,2
        # None <- 数え上げるのはtorchの仕事． CUDA_VISIBLE_DEVICESの設定がされていないのですべて扱えるはず

        # パスの設定読み込み
        paths = self.get_paths(args, config) # paths : Namespace, all attr is Path
        # 実験保存ディレクトリの作成
        paths['savedir'] = paths['exp'] / args.name
        if args.clear: # 強制新規作成のときは
            shutil.rmtree(paths['savedir'],ignore_errors=True)
        paths['savedir'].mkdir(parents=True,exist_ok=True)

        # 実験ディレクトリの中身を見てcontinue確認する
        continue_flag = self.continue_check(paths['savedir'])

        # 設定ファルに読み込んだ内容を追加
        config['env'] = {
            'prog' : paths['prog'],         # プログラムのルートディレクトリ
            'data' : paths['data'],         # データセットのルートディレクトリ
            'tmp'  : paths['tmp'],          # 計算キャッシュやglobal_writerに使う
            'savedir' : paths['savedir'],   # この実験の保存ディレクトリ
            'is_continue': continue_flag,   # 続きからかどうか
            'exp_name': args.name,          # 実験の名前
            'machine': machine,             # マシン名
            'cuda_string': cuda_string,     # CUDA_VISIBLE_DEVICESに設定された文字列
            'log': {
                'exp'  : paths['exp'],
                'config': config_path,     # 設定ファイルのパス
            }
        }

        # 設定ファイルの保存
        timestamp = (datetime.now()+timedelta(milliseconds=1)).strftime('%Y%m%d_%H%M_%S.%f')[:-3]
        shutil.copy(config_path, paths['savedir']/('hellfire_raw_config_%s.yml'%(timestamp)) )    # 生のやつの保存
        with (paths['savedir']/('hellfire_config_%s.yml'%(timestamp))).open('w') as f:             # 読み取り結果の保存
            yaml.dump(config,f)

        # Trainerの呼び出し
        sys.path.append(str(config['env']['prog']))
        Trainer = getattr(import_module('trainer.'+config['trainer']['name']),'Trainer')
        print('\033[36m>>> ================ environment construction ================= <<<\033[0m')
        trainer = Trainer(config)
        print('\033[36m>>> ======================= train start ======================= <<<\033[0m')
        try:
            trainer.train()
            # 終了したことを明示
            (paths['savedir']/'hellfire_end_point').touch()
            print('\033[36m>>> ======================== train end ======================== <<<\033[0m')
            del trainer
        except KeyboardInterrupt:
            print('\n\033[36m>>> ====================== catch Ctrl-C ======================= <<<\033[0m')
            del trainer
        print('\033[36m::: <<< Exit: TrainService\033[0m')


    def get_paths(self,args,config):
        """パスの設定

        優先度
            current path < environment variable < config file < command line input
        """

        # 初期化
        path_names = ['prog','data','exp','tmp']
        paths = {k:{'path':None,'src':None} for k in path_names} # パスと情報ソース

        # プログラムだけ初期設定はカレントディレクトリ
        paths['prog'] = {'path':Path(os.getcwd()),'src':'current path'}

        # 環境変数の読み込み
        env_names = [ 'ML'+name.upper() for name in path_names]
        paths = { p:{'path':Path(os.environ.get(e)),'src':'environment variable'}
                    if os.environ.get(e) is not None else paths[p] for p, e in zip(path_names,env_names) }

        # 設定ファイルからパスの読み込み
        if 'path' in config['environ']:
            paths = { p:{'path':Path(config['environ']['path'][e]),'src':'config file'}
                        if e in config['environ']['path'] else paths[p] for p, e in zip(path_names,env_names) }

        # コマンドライン引数からの読み込み
        paths = { p:{'path':Path(vars(args)[p]),'src':'commandline args'}
                    if vars(args)[p] is not None else paths[p] for p, e in zip(path_names,env_names) }

        # 存在確認
        for pname, ename in zip(path_names,env_names):
            # パスが設定されていないとき
            if paths[pname]['path'] is None:
                raise Exception('set path : '+pname+' ($'+ename+')')
            # 設定されたパスが存在しないとき
            if not paths[pname]['path'].exists():
                if pname in ['exp','tmp']: # 書き込み系ディレクトリはディレクトリを作っていいか聞く
                    if args.yes or input('the path is not exist : %s\n'
                             '    read from %s\n'
                             'Do you make the path (y/n)? >> '%(
                                 str(paths[pname]['path']),paths[pname]['src'])) == 'y':
                        paths[pname]['path'].mkdir(parents=True)
                    else:
                        raise Exception('    set accurate path : %s ($%s)'%(pname,ename))
                else:
                    raise Exception('the path is not exist : '+str(paths[pname]['path'])+\
                                    '\n    read from '+paths[pname]['src']+\
                                    '\n    set accurate path : '+pname+' ($'+ename+')')

        result_paths = {k:v['path'] for k,v in paths.items()}
        return result_paths


    def continue_check(self,save_path):
        '''
        実験が続きからかどうか見る
        '''
        start_file = save_path/'hellfire_start_point'
        end_file = save_path/'hellfire_end_point'

        # 既に終了しているとき
        if end_file.exists():
            raise Exception('this experiments has been ended.')

        # スタートファイルがあったら continue = True
        if start_file.exists():
            return True
        # なかったら作る
        else:
            start_file.touch()
            return False


    def get_device_settings(self,args,config):
        # GPU 優先度 CUDA_VISIBLE_DEVICES > argparse > config

        # pci順で考える
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

        ## CUDA_VISIBLE_DEVICES あり
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            cuda_string = os.environ.get('CUDA_VISIBLE_DEVICES')
            return cuda_string

        # hellfire コマンドライン指定あり
        if args.gpu is not None:
            if args.gpu == 'cpu':
                cuda_string = os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'
            elif args.gpu == 'all':
                cuda_string = None
            else:
                cuda_string = os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            return cuda_string

        # 設定ファイルに指定があるとき
        if 'gpu' in config['environ']:
            if config['environ']['gpu'] == 'cpu':
                cuda_string = os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'
            elif config['environ']['gpu'] == 'all':
                cuda_string = None
            else:
                cuda_string = os.environ['CUDA_VISIBLE_DEVICES'] = \
                                        ''.join([str(i) for i in config['environ']['gpu']])
            return cuda_string

        cuda_string = None
        return cuda_string
