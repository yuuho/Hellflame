from pathlib import Path
from importlib.machinery import SourceFileLoader
import inspect

def load_object(path, key):
    mod_file = str(( Path( inspect.stack()[1].filename ).parent / path ).resolve())
    mod_name = '_mod_'+str(hash(path)).replace('-','n')
    mod = SourceFileLoader( mod_name, mod_file ).load_module()
    return getattr(mod,key)


'''
from hellflame.snippets.relative_object_loader import load_object

# 相対パスでも動くし

Hoge = load_object('../../foo/bar/baz.py', 'Hoge')
Fuga = load_object('../../foo/bar/baz.py', 'Fuga')


# 絶対パスでも正しく動く

Hoge = load_object('/home/hoge/foo/bar/baz.py', 'Hoge')
Fuga = load_object('/home/hoge/foo/bar/baz.py', 'Fuga')

'''
