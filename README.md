# Hellfire

PyTorchで実験を管理するためのシステム．  
*特徴*
- 実験結果，プログラム，データセットの場所を設定ファイルで管理する．それぞれのサーバーで設定ファイルを準備しておくことで同じ様に実験することが可能．
- 様々なモジュールの動的importをする．プログラムを書き変えずに作成して動作させるため実験結果の再現性が担保される．
- (TODO)実験結果の状況を確認して途中から実行をできるようにする．
- 自由に改変可能なサンプルプログラムを用意
- (TODO)ユーティリティ関数を用意

## インストール
```
pip install git+https://github.com/yuuho/Hellfire
```

## アンインストール
```
pip uninstall hellfire
```

## Requirements/依存関係
hellfireはconda環境に追加して入れることを前提としている．  
以下のようにしてcondaパッケージをあらかじめ追加しておく．
```
conda install pyyaml
```

## サブコマンド
``hellfire``の後に続けて使う
- ``train`` : 学習システムの起動
- ``doctor`` : hellfire環境のチェック


## ドキュメント
[document](.doc/index.md)
