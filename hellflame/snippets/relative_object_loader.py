from pathlib import Path
from importlib.machinery import SourceFileLoader
import inspect

def load_object(path, key):
    mod_file = str(( Path( inspect.stack()[1].filename ).parent / path ).resolve())
    mod_name = '_mod_'+str(hash(path)).replace('-','n')
    mod = SourceFileLoader( mod_name, mod_file ).load_module()
    return getattr(mod,key)


'''
Hoge = load_object('../../foo/bar/baz.py', 'Hoge')
Fuga = load_object('../../foo/bar/baz.py', 'Fuga')
'''
