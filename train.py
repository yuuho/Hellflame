import os
import sys
import socket
import filecmp
import shutil
from pathlib import Path
from importlib import import_module
import argparse

import yaml


# コマンドライン入力の受け取り
def parse_args():
    parser = argparse.ArgumentParser()
    # 必須項目
    parser.add_argument('--config','-c',type=str,
                        default=None,
                        dest='config', help='config file path')
    parser.add_argument('--name','-n',type=str,
                        default=None,
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
    args = parser.parse_args()
    return args


def valid_required(args):
    paths = argparse.Namespace()

    # エラー終了する処理
    def err_end(cond,text):
        if cond:
            print(text)
            exit()

    # 必須項目の存在確認
    err_end(args.config == None,'set config file')
    err_end(args.name == None,  'set experiment name')

    # 設定ファイルの存在確認
    paths.config = Path(args.config)
    err_end(not paths.config.exists(),'there is not such a config file')

    return paths


# パスの読み込みと存在確認
def read_paths(args,paths):
    # まずは実行しているディレクトリを登録
    paths.prog = Path(os.getcwd())

    # 環境変数からの読み込み
    paths.prog = Path(os.environ.get('MLPROG')) if os.environ.get('MLPROG') is not None else paths.prog
    paths.data = Path(os.environ.get('MLDATA')) if os.environ.get('MLDATA') is not None else None
    paths.exp = Path(os.environ.get('MLEXP')) if os.environ.get('MLEXP') is not None else None
    paths.tmp = Path(os.environ.get('MLTMP')) if os.environ.get('MLTMP') is not None else None

    # 設定ファイルからパスの読み込み
    '''
    with open(str(paths.config),'r') as f:
        config = yaml.load(f)

    # 設定ファイルからパスの読み込み
    if 'environ' in config:
        env = config['environ']
        print(env)
        #paths.prog = config
    '''

    # コマンドライン引数からパスの読み込み
    paths.prog = Path(args.prog) if args.prog is not None else paths.prog
    paths.data = Path(args.data) if args.data is not None else paths.data
    paths.exp = Path(args.exp) if args.exp is not None else paths.exp
    paths.tmp = Path(args.tmp) if args.tmp is not None else paths.tmp



    # 最後にディレクトリの存在確認
    if paths.prog is None or not paths.prog.exists():
        print('programs directory is not exists')
        exit()

    if paths.data is None or not paths.data.exists():
        print('dataset directory is not exists')
        exit()

    if paths.exp is None or not paths.exp.exists():
        print('experiment directory is not exists')
        exit()

    if paths.tmp is None or not paths.tmp.exists():
        print('temporary directory is not exists')
        exit()


    paths.savedir = paths.exp / args.name # どうするべきか

    return paths


if __name__ == '__main__':

    print('===================================================================')
    print('| Hellfire Start                                                  |')
    print('===================================================================')

    # コマンドライン入力の受け取り
    args = parse_args()
    # パスの設定と存在確認
    paths = valid_required(args)
    paths = read_paths(args,paths)

    # 設定ファイルの読み込み
    with open(str(paths.config),'r') as f:
        config = yaml.load(f)
    config['environ'].update({
        'prog' : paths.prog,
        'data' : paths.data,
        'exp'  : paths.exp,
        'tmp'  : paths.tmp,
        'savedir' : paths.savedir
    })

    print(paths)

    # モジュールの呼び出しと起動
    sys.path.append(str(config['environ']['prog']))
    Trainer = getattr(import_module('trainers.'+config['trainer']['name']),'Trainer')
    trainer = Trainer(config)
    trainer.train()

    print('===================================================================')
    print('| Hellfire End                                                    |')
    print('===================================================================')
