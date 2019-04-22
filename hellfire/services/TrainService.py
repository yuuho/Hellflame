
import sys
import os
from argparse import Namespace
from pathlib import Path
from importlib import import_module

import yaml

from hellfire.services.Service import Service


class TrainService(Service):
    command_name = 'train'
    help_text = 'hogehoge'

    def __init__(self,subparsers):
        super().__init__(subparsers)

    # subparserの登録
    def register_parser(self,parser):
        # 必須項目
        parser.add_argument('--config','-c',type=str,
                            required=True,
                            dest='config', help='config file path')
        parser.add_argument('--name','-n',type=str,
                            required=True,
                            dest='name', help='experiment name')
        # 環境変数からも読み込める
        parser.add_argument('--programs','-p', type=str,
                            default=None,
                            dest='prog', help='source code dir path')
        parser.add_argument('--datasets','-d', type=str,
                            default=None,
                            dest='data', help='データセット置き場')
        parser.add_argument('--experiments','-e', type=str,
                            default=None,
                            dest='exp', help='実験データ置き場')
        parser.add_argument('--temporary','-t', type=str,
                            default=None,
                            dest='tmp', help='一時ファイル置き場')


    # エントリーポイント
    def handler_function(self,args):
        # 設定ファイルの存在確認
        paths = self.get_paths(args) # paths : Namespace, all attr is Path

        # 設定ファイルの読み込み
        with open(str(paths.config),'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        config['environ'].update({
            'prog' : paths.prog,
            'data' : paths.data,
            'exp'  : paths.exp,
            'tmp'  : paths.tmp,
            'savedir' : paths.savedir
        })

        ## continue確認する？

        # Trainerの呼び出し
        sys.path.append(str(config['environ']['prog']))
        Trainer = getattr(import_module('trainers.'+config['trainer']['name']),'Trainer')
        trainer = Trainer(config)
        trainer.train()


    def get_paths(self,args):
        """パスの設定

        優先度
            current path < environment variable < config file < command line input
        """

        # 初期化
        path_names = ['prog','data','exp','tmp']
        env_names = [ 'ML'+name.upper() for name in path_names]
        paths = {k:{'path':None,'src':None} for k in path_names}

        # プログラムだけ初期設定はカレントディレクトリ
        paths['prog'] = {'path':Path(os.getcwd()),'src':'current path'}

        # 環境変数の読み込み
        for pname, ename in zip(path_names,env_names):
            if os.environ.get(ename) is not None:
                paths[pname] = {'path':Path(os.environ.get(ename)),'src':'environment variable'}

        # 設定ファイルの存在確認
        config_path = Path(args.config)
        if not config_path.exists():
            raise Exception('there is not such a config file : '+config_path)
        else:
            with open(str(config_path),'r') as f:
                config = yaml.load(f,Loader=yaml.FullLoader)

        # 設定ファイルからパスの読み込み
        if 'path' in config['environ']:
            for pname, ename in zip(path_names,env_names):
                if ename in config['environ']['path']:
                    paths[pname] = {'path':Path(config['environ']['path'][ename]),'src':'config file'}

        # コマンドライン引数からの読み込み
        for pname in path_names:
            if vars(args)[pname] is not None:
                paths[pname] = {'path':Path(vars(args)[pname]),'src':'commandline args'}

        # 存在確認
        for pname, ename in zip(path_names,env_names):
            # パスが設定されていないとき
            if paths[pname]['path'] is None:
                raise Exception('set path : '+pname+' ($'+ename+')')
            # 設定されたパスが存在しないとき
            if not paths[pname]['path'].exists():
                raise Exception('the path is not exist : '+str(paths[pname]['path'])+\
                                '\n    read from '+paths[pname]['src']+\
                                '\n    set accurate path : '+pname+' ($'+ename+')')

        # Namespaceに格納
        result_paths = Namespace()
        result_paths.__dict__.update({k:v['path'] for k,v in paths.items()})

        # 実験結果保存用ディレクトリを作成
        result_paths.savedir = result_paths.exp / args.name
        result_paths.savedir.mkdir(parents=True,exist_ok=True)

        # 設定ファイルのパスを格納
        result_paths.config = config_path

        return result_paths
