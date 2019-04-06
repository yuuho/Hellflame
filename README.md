# Hellfire

GPUを燃やすための地獄の学習システム．


## インストール
このディレクトリで以下のコマンドを実行するとhellfireというコマンドが実行できるようになる．
```
pip install -e .
```

## アンインストール
```
python setup.py install --record files.txt
cat files.txt | xargs rm -rf
```