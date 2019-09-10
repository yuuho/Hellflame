import argparse
from importlib import import_module


__version__ = '0.2.5'


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

        # コマンド入力の受け取り
        args = self.parser.parse_args()
        # 実行 or エラーヘルプ
        result = args.handler(args) if hasattr(args,'handler') else self.parser.print_help()

        print('\033[33m===================================================================\n' \
                      '| Hellfire End                                                    |\n' \
                      '===================================================================\033[0m')


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


#if __name__ == '__main__':
def main():
    hellfire = Hellfire()
