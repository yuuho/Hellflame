
import os
from argparse import Namespace
from pathlib import Path

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

        print(paths)

        # 読み込み
        print('here do train')


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
        paths['prog'] = {'paht':Path(os.getcwd()),'src':'current path'}

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
            if paths[pname]['path'] is None:
                raise Exception('set path : '+pname+' ($'+ename+')')
            if not paths[pname]['path'].exists():
                raise Exception('the path is not exist : '+str(paths[pname]['path'])+' read from '+paths[pname]['src']+\
                                'set accurate path : '+pname+' ($'+ename+')')

        # Namespaceに格納
        result_paths = Namespace()
        result_paths.__dict__.update({k:v['path'] for k,v in paths.items()})

        return result_paths

