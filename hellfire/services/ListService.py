
import os
from pathlib import Path
import re

from hellfire.services.Service import Service


class Counter:
    def __init__(self):
        self.count = 0
    
    def getc(self):
        return self.count

    def increment(self):
        self.count += 1

class Tree:
    def __init__(self,nodes,counter):
        self.counter = counter
        self.title = nodes[0]
        self.children = []

        if nodes[1]=='hellfire_start_point':
            self.nodetype = 'end'
        else:
            self.nodetype = 'inner'
            self.children.append(Tree(nodes[1:],self.counter))

    def add(self,nodes):
        exist_flag = False

        for child in self.children:
            if nodes[0] == child.title:
                child.add(nodes[1:])
                exist_flag = True
                break
        
        if not exist_flag:
            self.children.append(Tree(nodes,self.counter))

    def render(self,indent):
        if self.nodetype == 'inner':
            num = '   '
        else:
            num = '%3d'%(self.counter.getc())
            self.counter.increment()

        end = '/' if self.nodetype == 'inner' else ' @'
        print('%s : %s%s%s'%(num,indent*'   ',self.title,end))
        for c in self.children:
            c.render(indent+1)


class RootTree(Tree):
    def __init__(self,root,leaves):
        self.title = root.name
        self.counter = Counter()

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
        pass

    # エントリーポイント
    def handler_function(self,args):
        print('::: >>> Enter: ListService')

        exp_path = Path(os.environ.get('MLEXP')).resolve()
        print('MLEXP : ',exp_path)

        exps = [item for item in sorted(exp_path.rglob('*'))
                if re.search(r'\/hellfire_start_point$',str(item)) and item.is_file()]
        print('>>> ======================= list start ======================== <<<')
        tree = RootTree(exp_path,exps)
        tree.render()
        print('>>> ======================= list end ========================== <<<')

        print('::: <<< Exit: ListService')
