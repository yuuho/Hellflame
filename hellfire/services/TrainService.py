
import sys
import os
from argparse import Namespace
from pathlib import Path
from importlib import import_module
from datetime import datetime, timedelta

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

        # TODO) 最初から オプション
        # TODO) すべて yes オプション


    # エントリーポイント
    def handler_function(self,args):
        print('::: >>> Enter: TrainService')
        # 設定ファイルの存在確認
        paths = self.get_paths(args) # paths : Namespace, all attr is Path

        # 設定ファイルの読み込み
        with open(str(paths.config),'r') as f:
            config = yaml.load(f,Loader=yaml.FullLoader)

        # continue確認する
        continue_flag = self.continue_check(paths.savedir)
        
        # 設定ファルの準備
        config['environ'].update({
            'prog' : paths.prog,
            'data' : paths.data,
            'exp'  : paths.exp,
            'tmp'  : paths.tmp,
            'savedir' : paths.savedir,
            'config': paths.config,
            'is_continue': continue_flag
        })

        # 設定ファイルの保存
        timestamp = (datetime.now()+timedelta(milliseconds=1)).strftime('%Y%m%d_%H%M_%S.%f')[:-3]
        with (paths.savedir/('hellfire_config_%s.yml'%(timestamp))).open('w') as f:
            yaml.dump(config,f)

        # Trainerの呼び出し
        sys.path.append(str(config['environ']['prog']))
        Trainer = getattr(import_module('trainer.'+config['trainer']['name']),'Trainer')
        print('>>> ================ environment construction ================= <<<')
        trainer = Trainer(config)
        print('>>> ======================= train start ======================= <<<')
        try:
            trainer.train()
            # 終了したことを明示
            (paths.savedir/'hellfire_end_point').touch()
        except KeyboardInterrupt:
            print('\n>>> ====================== catch Ctrl-C ======================= <<<')
            print('::: <<< Exit: TrainService')


    def get_paths(self,args):
        """パスの設定

        優先度
            current path < environment variable < config file < command line input
        """

        # 初期化
        path_names = ['prog','data','exp','tmp']
        env_names = [ 'ML'+name.upper() for name in path_names]
        paths = {k:{'path':None,'src':None} for k in path_names} # パスと情報ソース

        # プログラムだけ初期設定はカレントディレクトリ
        paths['prog'] = {'path':Path(os.getcwd()),'src':'current path'}

        # 環境変数の読み込み
        for pname, ename in zip(path_names,env_names):
            if os.environ.get(ename) is not None:
                paths[pname] = {'path':Path(os.environ.get(ename)),'src':'environment variable'}

        # 設定ファイルの存在確認
        config_path = Path(args.config)
        if not config_path.exists():
            raise Exception('there is not such a config file : '+str(config_path))
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
                if pname in ['exp','tmp']: # 書き込み系ディレクトリはディレクトリを作っていいか聞く
                    if input('the path is not exist : %s\n'
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

        # Namespaceに格納
        result_paths = Namespace()
        result_paths.__dict__.update({k:v['path'] for k,v in paths.items()})

        # 実験結果保存用ディレクトリを作成
        result_paths.savedir = result_paths.exp / args.name
        result_paths.savedir.mkdir(parents=True,exist_ok=True)

        # 設定ファイルのパスを格納
        result_paths.config = config_path

        return result_paths


    def continue_check(self,save_path):
        start_file = save_path/'hellfire_start_point'
        end_file = save_path/'hellfire_end_point'

        # 既に終了しているとき
        if end_file.exists():
            # TODO) 最初からオプションを考慮する
            raise Exception('this experiments has been ended.')

        if start_file.exists():
            return True
        else:
            start_file.touch()
            return False