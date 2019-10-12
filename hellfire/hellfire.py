import argparse
from importlib import import_module


__version__ = '0.3.2'


class Hellfire:

    service_names = ['train','doctor','list']

    def __init__(self):
        print('\033[33m===================================================================\n' \
                      '| Hellfire Start  %s ( ver. %s )                            |\n' \
                      '===================================================================\033[0m'%(' '*(10-len(__version__)),__version__))

        # クラス変数の読み込み
        self.service_names = Hellfire.service_names

        # コマンド入力パーサーの作成
        self.parser = argparse.ArgumentParser(description='Hellfire')
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
                      '| Hellfire End    %s ( ver. %s )                            |\n' \
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
            class_module = getattr(import_module('hellfire.services.'+class_name),class_name)
            # サービスを追加
            self.services[service_name] = class_module(subparsers)


def main():
    hellfire = Hellfire()
    result = hellfire.run()
    del hellfire

    if result>0:
        import os
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
