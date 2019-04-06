
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
        print('=== Hellfire doctor ===')
        envs = ['MLPROG','MLDATA','MLEXP','MLTMP']
        for env in envs:
            if os.environ.get(env) is not None:
                print(' - environment variable ',env,' is ',os.environ.get(env))
            else:
                print(' - environment variable ',env,' does not exist')
                print('    set the environment variable\n---\nexport '+env+'="/path/to/dir"\n---')

