import argparse
from importlib import import_module


__version__ = '0.4.3'


class Hellflame:

    service_names = ['train','doctor','list','publish','warming']

    def __init__(self):
        print('\033[33m===================================================================\n' \
                      '| Hellflame Start  %s ( ver. %s )                           |\n' \
                      '===================================================================\033[0m'%(' '*(10-len(__version__)),__version__))

        # クラス変数の読み込み
        self.service_names = Hellflame.service_names

        # コマンド入力パーサーの作成
        self.parser = argparse.ArgumentParser(description='Hellflame')
        # 各サービスの実体を保持する領域の作成
        self.services = {}
        # 全てのサービスを読み込み
        self.read_services()


    def run(self):
        # コマンド入力の受け取り
        args = self.parser.parse_args()
        # 実行 or エラーヘルプ
        result = args.handler(args) if hasattr(args,'handler') else self.parser.print_help()
        return result


    def __del__(self):
        print('\033[33m===================================================================\n' \
                      '| Hellflame End    %s ( ver. %s )                           |\n' \
                      '===================================================================\033[0m'%(' '*(10-len(__version__)),__version__))


    # 存在するすべてのサービスへの案内を準備
    def read_services(self):
        # subparserの入れ物を準備
        subparsers = self.parser.add_subparsers()
        # 全てのsubparserを登録
        for service_name in self.service_names:
            # クラスの名前に変換
            class_name = service_name.capitalize()+'Service'
            # クラスのモジュールをロード
            class_module = getattr(import_module('hellflame.services.'+class_name),class_name)
            # サービスを追加
            self.services[service_name] = class_module(subparsers)


def main():
    hellflame = Hellflame()
    result = hellflame.run()
    del hellflame

    if result is not None and result>0:
        import os
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
