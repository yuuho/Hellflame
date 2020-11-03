from pathlib import Path
from importlib.machinery import SourceFileLoader

def load_object(path, key):
    mod_file = lambda p: str(( Path(__file__).parent / path ).resolve())
    mod_name = lambda p: '_mod_'+(hash(p)>=0 and 'p' or 'n')+str(abs(hash(p)))
    mod = SourceFileLoader( mod_name(path), mod_file(path) ).load_module()
    return getattr(mod,key)


'''
Hoge = load_object('../../foo/bar/baz.py', 'Hoge')
Fuga = load_object('../../foo/bar/baz.py', 'Fuga')
'''
