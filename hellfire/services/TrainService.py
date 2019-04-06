
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

    # エントリーポイント
    def handler_function(self,args):

        # argsのvalidation

        # 読み込み

        print('here do train')

    def validate_args(self,args):
        """コマンドライン引数が正しいか確認"""
        return args
