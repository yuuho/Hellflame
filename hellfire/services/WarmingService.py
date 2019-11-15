
import os
import time

from hellfire.services.Service import Service


def warmup(index):
    import torch
    assert torch.cuda.device_count()==1, 'set one GPU'
    print(index,':',torch.cuda.get_device_name(0))
    torch.no_grad()

    allocated = []
    count_giga_byte = 0
    
    # メモリの確保
    while True:
        try:
            allocated += [torch.randn(256,1024,1024).cuda()]
            count_giga_byte+=1
            print('%3d,'%count_giga_byte,end='',flush=True)
        except:
            allocated.pop()
            print('\ntotally allocated %d GB'%(count_giga_byte-1))
            break
    
    if count_giga_byte==0:
        print('GPU memory allocation failure...')
        return

    print('warming up start...')
    while True:
        for tensor in allocated:
            tensor = tensor + tensor
            time.sleep(0.03)
    

class WarmingService(Service):
    command_name = 'warming'
    help_text = 'allocate a gpu and warm up it'

    def __init__(self,subparsers):
        super().__init__(subparsers)

    # subparserの登録
    def register_parser(self,parser):
        parser.add_argument('index',type=int,
                            help='index of gpu (nvidia-smi), GPUの番号')

    # エントリーポイント
    def handler_function(self,args):
        print('\033[36m::: >>> Enter: WarmingService\033[0m')
        print('\033[36m>>> ======================= warming up start ====================== <<<\033[0m')

        print('\033[36mGPU ID = %d\033[0m'%args.index)

        try:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.index)
            warmup(args.index)
        except KeyboardInterrupt:
            print('\n\033[36m>>> ====================== catch Ctrl-C ======================= <<<\033[0m')

        print('\033[36m>>> ======================= warming up end ======================== <<<\033[0m')
        print('\033[36m::: <<< Exit: WarmingService\033[0m')
        return 1

