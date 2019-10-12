
import os
from pathlib import Path

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
        print('\033[36m::: >>> Enter: DoctorService\033[0m')
        envs = ['MLPROG','MLDATA','MLEXP','MLTMP']
        print('\033[36m>>> ======================= doctor start ====================== <<<\033[0m')

        # 環境変数の設定状況
        print('# setted environment paths:')
        err_log = '\n'
        err_flag = False
        for env in envs:
            if os.environ.get(env) is not None:
                print(' o | environment variable ',env,' is ',os.environ.get(env))
            else:
                if err_flag==False:
                    err_log += '--- set the environment variable ---\n'
                    err_flag = True
                print(' x | environment variable ',env,' does not exist')
                err_log += 'export %s="/path/to/dir"\n'%(env)
        if err_flag:
            print(err_log)

        # 設定されたパスの存在状況
        print('# paths existence:')
        err_log = '\n'
        err_flag = False
        for env in envs:
            if os.environ.get(env) is not None:
                p = Path(os.environ.get(env)).resolve()
                if p.exists():
                    print(' o | environment variable ',env,':',p,' is exist')
                else:
                    if err_flag==False:
                        err_log += '--- make the directory ---\n'
                        err_flag = True
                    print(' x | environment variable ',env,':',p,' does not exist')
                    err_log += 'mkdir -p %s\n'%(str(p))
        if err_flag:
            print(err_log)



        print('\033[36m>>> ======================= doctor end ======================== <<<\033[0m')
        print('\033[36m::: <<< Exit: DoctorService\033[0m')
        return 0

