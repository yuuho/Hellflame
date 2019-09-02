
class Service:
    # クラス変数として定義しておく
    command_name = 'template'
    help_text = 'here write command discription'

    # クラス変数をロードするための関数
    @classmethod
    def load_class_parameters(cls):
        return {
            'command_name' : cls.command_name,
            'help_text' : cls.help_text,
        }

    def __init__(self,subparsers):
        # クラス変数を読み込んでインスタンスに利用できるように
        class_parameters = self.load_class_parameters()
        self.command_name = class_parameters['command_name']
        self.help_text = class_parameters['help_text']

        # サブパーサーを作成して設定
        subparser = subparsers.add_parser(self.command_name,help=self.help_text)
        self.register_parser(subparser)
        subparser.set_defaults(handler=self.handler_function)

        #print(self.command_name+' is loaded')


    def register_parser(self,subparser):
        pass

    def handler_function(self,args):
        pass
