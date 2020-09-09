# Hellflame

hellflameはモジュール間の結合をメタ的に行うため，
PyTorch本来の持つクラスと違ったクラスのimportなどは必要にならない．

モジュールの結合を設定ファイルで表す．
モジュールの読み込み部分も自分で実装するため，
あらゆる状況にフレキシブルに対応可能．(つまり実質的に何も行なっていない)

面倒な環境チェックの作業を行う部分を提供する．

## メモ
```
- hellflame/ : hellflame本体
    - services/ : サブコマンドの実行ファイルが入っている
        - Service.py : サブコマンドクラスの親クラス
        - TrainService.py : 学習システム
        - DoctorService.py : 調査システム
    - snippets/ : 有用なメモのようなコードが入っている．コピーして使うこと前提だが面倒な時は参照して利用できる．
- sample_config/ : サンプルプロジェクトの動作
- sample_proj/ : hellflameプロジェクトの書き方の例．model zoo的な使い方もする．
```

## 実験時に保存すべきこと？
- コード？
- 実行時環境のログ？
- 実験結果


## 設定ファイルの記述方法
Hellflame TrainServiceが設定ファイルを読み込み，
その指定通りに実験を行う．

設定ファイルはyaml形式のテキストファイルである．
yamlはjsonなどのように階層的にテキスト形式で情報を保存できるフォーマットであり，
配列やハッシュ(辞書)などのデータ構造を用いることが出来る．

Hellflame TrainServiceが設定ファイルに対して要求する記述はenvironというキーのハッシュと
trainerというキーのハッシュのみである．

以下に最小構成のファイルを掲載する．
```
environ: {}
trainer:
    name: hoge
    params: {}
```
この状態ではprogramディレクトリとして設定したディレクトリの中の
trainer/hoge.py モジュールにある Trainer オブジェクトが
``trainer = Trainer(config,**params)``という形で呼ばれ，
``trainer.train()``される．

したがって，``Trainer``は呼び出し可能なオブジェクトであり，
その返り値のオブジェクトは``train()``というメソッドを持っている必要がある．

yaml内のその他の記述に関してはTrainerの扱うところであり，自由である．
``trainer = Trainer(config,**params)``におけるconfigは
この設定ファイルを読み込んだ内容とTrainServiceが付加した実験情報でありpythonのdict形式である，
paramsは``trainer: params:``に記述した内容であり，pythonのdict形式である．

現状では設定ファイルのenvには以下の内容が記載可能である．
```
environ:
    paths:
        MLEXP: hoge
        MLTMP: fuga
        MLDATA: piyo
        MLPROG: foo
```

## configに付加される情報
```
config['env'] = {
    'prog' : paths['prog'],        # プログラムのルートディレクトリ
    'data' : paths['data'],        # データセットのルートディレクトリ
    'tmp'  : paths['tmp'],         # 計算キャッシュやglobal_writerに使う
    'savedir' : paths['savedir'],  # この実験の保存ディレクトリ
    'is_continue': continue_flag,   # 続きからかどうか
    'exp_name': args.name,          # 実験の名前
    'machine': 'piyo',              # マシンの名前
    'gpu':  [0,1,2],                # GPU
    'log': {
        'exp'  : paths['exp'],
        'config': config_path,     # 設定ファイルのパス
    }
}
```
