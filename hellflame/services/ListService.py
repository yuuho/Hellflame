
import os
from pathlib import Path
import re

from hellflame.services.Service import Service


# 全体で実験結果の個数をカウントするためのオブジェクト
class Counter:
    def __init__(self):
        self.count = 0

    def getc(self):
        return self.count

    def increment(self):
        self.count += 1

# ベースとなるツリー
class Tree:
    def __init__(self,nodes,counter,trace,size_check):
        self.counter = counter
        self.title = nodes[0]
        self.children = []
        self.size_check = size_check
        self.trace = trace / self.title

        if nodes[1] in ['hellfire_start_point','hellflame_start_point']:
            self.nodetype = 'end'
        else:
            self.nodetype = 'inner'
            self.children.append(Tree(nodes[1:],self.counter,self.trace,size_check=self.size_check))

    def add(self,nodes):
        exist_flag = False

        for child in self.children:
            if nodes[0] == child.title:
                child.add(nodes[1:])
                exist_flag = True
                break

        if not exist_flag:
            self.children.append(Tree(nodes,self.counter,self.trace,size_check=self.size_check))

    def render(self,indent):
        if self.nodetype == 'inner':
            num = '   '
        else:
            num = '%3d'%(self.counter.getc())
            self.counter.increment()

        is_end = '\033[32mDONE\033[0m' if (self.trace/'hellflame_end_point').exists() or \
                   (self.trace/'hellfire_end_point').exists() else '\033[31mWIP\033[0m'
        end = '/' if self.nodetype == 'inner' else '    @[%s]'%(is_end)
        # サイズ情報が必要なとき
        if self.nodetype!='inner' and self.size_check:
            all_file = [item for item in list(self.trace.rglob('*'))
                            if re.search(r'.*',str(item)) and item.is_file()]
            total = 0
            for f in all_file:
                total += os.path.getsize(str(f))
            if total>(10**9):
                end += ' %.1lf GB'%(total/(10**9))
            elif total>(10**6):
                end += ' %.1lf MB'%(total/(10**6))
            else:
                end += ' %.1lf kb'%(total/(10**3))
        print('%s : %s%s%s'%(num,indent*'   ',self.title,end))
        for c in self.children:
            c.render(indent+1)


# 一番上だけ別の動作
class RootTree(Tree):
    def __init__(self,root,leaves,size_check):
        self.title = root.name
        self.counter = Counter()
        self.size_check = size_check
        self.trace = root # 絶対パス

        start = len(root.parts)
        leaves = sorted(leaves,key=lambda x:str(x))
        lleaves = [list(l.parts[start:]) for l in leaves]

        self.children = []
        for leaf in lleaves:
            self.add(leaf)

    def render(self):
        num = self.counter.getc()
        self.counter.increment()

        print('    : %s/'%(self.title))
        for c in self.children:
            c.render(1)



class ListService(Service):
    command_name = 'list'
    help_text = '実験項目を見る'

    def __init__(self,subparsers):
        super().__init__(subparsers)

    # subparserの登録
    def register_parser(self,parser):
        parser.add_argument('--size','-s', action='store_true',
                            dest='size_check', help='ファイルサイズも測定する')

    # エントリーポイント
    def handler_function(self,args):
        print('\033[36m::: >>> Enter: ListService\033[0m')

        exp_path = Path(os.environ.get('MLEXP')).resolve()
        print('MLEXP : ',exp_path)

        # 実験スタートファイルのみ
        exps = [item for item in sorted(exp_path.rglob('*'))
                if re.search(r'\/hell(flame|fire)_start_point$',str(item)) and item.is_file()]
        print('\033[36m>>> ======================= list start ======================== <<<\033[0m')
        tree = RootTree(exp_path,exps,size_check=args.size_check)
        tree.render()
        print('\033[36m>>> ======================= list end ========================== <<<\033[0m')

        print('\033[36m::: <<< Exit: ListService\033[0m')
        return 0
