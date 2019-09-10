
import os

from hellfire.services.Service import Service


class DoctorService(Service):
    command_name = 'doctor'
    help_text = 'see information of environment'

    def __init__(self,subparsers):
        super().__init__(subparsers)

    # subparserの登録
    def register_parser(self,parser):
        pass

    # エントリーポイント
    def handler_function(self,args):
        print('::: >>> Enter: DoctorService')
        envs = ['MLPROG','MLDATA','MLEXP','MLTMP']
        print('>>> ======================= doctor start ====================== <<<')
        err_log = '\n'
        for env in envs:
            if os.environ.get(env) is not None:
                print(' o | environment variable ',env,' is ',os.environ.get(env))
            else:
                print(' x | environment variable ',env,' does not exist')
                err_log +=  '--- set the environment variable ---\n' \
                            'export %s="/path/to/dir"\n'%(env)
        print(err_log)
        print('>>> ======================= doctor end ======================== <<<')
        print('::: <<< Exit: DoctorService')

