# ドキュメントはコード末尾
import shutil
from importlib.machinery import SourceFileLoader
from argparse import Namespace


def _load_modules(self):

    # プログラムを保存するディレクトリ
    prog_save_dir = self.config['env']['savedir']/'program'
    self.modules = Namespace()

    # 読み込み対象
    to_load = { k:v for k,v in self.config.items() if not k in ['env','environ'] }

    for key,struct in to_load.items():

        # loggerなどリスト形式で指定されたもの
        if type(struct) == list:
            # 読み込み
            relat_path_strs = [key+'/'+v['name']+'.py' for v in struct ]
            mod_paths = [self.config['env']['prog']/p for p in relat_path_strs]
            mods = [getattr(  SourceFileLoader('_'+key.capitalize()+'s',str(p)).load_module(),
                            key.capitalize()    ) for p in mod_paths ]
            # コピー
            dst_paths = [prog_save_dir/p for p in relat_path_strs]
            [(dp.parent.mkdir(parents=True,exist_ok=True), shutil.copy(mp, dp))
                        for mp,dp in zip(mod_paths,dst_paths)]
            self.modules.__dict__.update({key.capitalize()+'s':mods})
        
        # 普通のやつ trainerなど
        elif type(struct)==dict and 'name' in struct.keys():
            # 読み込み
            relat_path_str = key+'/'+struct['name']+'.py'
            mod_path = self.config['env']['prog']/relat_path_str
            mod = getattr(  SourceFileLoader('_'+key.capitalize(),str(mod_path)).load_module(),
                            key.capitalize()                                    )
            self.modules.__dict__.update({key.capitalize():mod})
            # コピー
            dst_path = prog_save_dir/relat_path_str
            dst_path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy(mod_path, dst_path)
        
        # dictでネストされたもの model.generatorなど
        else:
            # 読み込み
            relat_path_strs = {k:key+'/'+v['name']+'.py' for k,v in struct.items()}
            mod_paths = {k:self.config['env']['prog']/p for k,p in relat_path_strs.items()}
            mods = {    key.capitalize()+k.capitalize():
                            getattr(  SourceFileLoader('_'+key.capitalize()+k.capitalize(),str(p)).load_module(),
                            key.capitalize()    ) for k,p in mod_paths.items()}
            dst_paths = {k:prog_save_dir/p for k,p in relat_path_strs.items()}
            [(dp.parent.mkdir(parents=True,exist_ok=True), shutil.copy(mod_paths[k], dp))
                        for k,dp in dst_paths.items()]
            self.modules.__dict__.update(mods)



'''
# 概要
設定ファイルどおりにファイルから読み込み，
実行したファイルのコピーを実験結果ディレクトリに作る．


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