# ドキュメントはコード末尾
from importlib.machinery import SourceFileLoader
from argparse import Namespace


def _load_modules(self):

    # 読み込んだモジュール
    self.modules = Namespace()

    # 読み込み対象
    to_load = { k:v for k,v in self.config.items() if not k in ['env','environ'] }

    for key,struct in to_load.items():
        # 形式の判定
        ## list形式のもの 'list' : list
        ## ひとつだけのもの 'one' : dict, has 'name', dict['name'] == str
        ## 辞書形式のもの 'dict' : dict, has or dont has 'name', dict['name'] == dict
        mode =  'list' if (type(struct)==list) else \
                'one' if (type(struct)==dict) and ('name' in struct.keys()) and (type(struct['name'])==str) else \
                'dict'
        # .pyファイルへの相対パス
        relat_path_str = {
            'list': lambda key,st: [key+'/'+v['name']+'.py' for v in st],
            'one':  lambda key,st: key+'/'+st['name']+'.py',
            'dict': lambda key,st: {k:key+'/'+v['name']+'.py' for k,v in st.items()}
            }[mode](key,struct)
        # 相対パスにプログラムディレクトリをsuffixとして追加
        mod_path = {
            'list': lambda relat: [self.config['env']['prog']/p for p in relat],
            'one':  lambda relat: self.config['env']['prog']/relat,
            'dict': lambda relat: {k:self.config['env']['prog']/p for k,p in relat.items()}
            }[mode](relat_path_str)
        # 読み込み
        load = lambda dummy,path,objname: getattr(SourceFileLoader(dummy,path).load_module(),objname)
        cap = lambda x: x.capitalize()
        append = {
            'list': lambda key,path: { cap(key)+'s':
                                        [load('_'+cap(key)+'s',   str(p),    cap(key)) for p in path]           },
            'one':  lambda key,path: { cap(key):
                                        load('_'+cap(key),        str(path), cap(key))                          },
            'dict': lambda key,path: { cap(key)+cap(k):
                                        load('_'+cap(key)+cap(k), str(p),    cap(key)) for k,p in path.items()  }
            }[mode](key,mod_path)
        # 追加
        self.modules.__dict__.update(append)



'''
# 概要
設定ファイルどおりにファイルから読み込み，
実行したファイルのコピーを作らない．


# 使い方
Trainerクラスに書いている
```
class Trainer:
    def _load_modules(self):
        hogehoge
```
を

``from hellflame.snippets.config_loader import _load_modules``
した上で
```
class Trainer:
    _load_modules = _load_modules
```
という感じにクラスプロパティ的に挿入するだけで良い．
'''
