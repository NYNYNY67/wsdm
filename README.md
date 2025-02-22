# 環境構築
vast.ai, NVIDIA CUDA templateでインスタンスを作成


## pyenvのinstall

```
curl https://pyenv.run | bash
```

- `~/.bashrc`に下記を転記してpathを通す

```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

- 設定を反映

```
source ~/.bashrc
```

[参考](https://github.com/pyenv/pyenv)


## pyenvを利用したpythonのinstall

- そのまま新しいversionをinstallするとpythonのインストール時にエラーが出るので、依存packageをinstallしておく

```
apt install -y libedit-dev libssl-dev libffi-dev libreadline-dev sqlite3 liblzma-dev libsqlite3-dev libbz2-dev
```

- python3.10のインストール

```
pyenv install 3.13
pyenv global 3.13
```

## pipxのinstall
- poetryのinstallのため、pipxをまずinstallする

```
apt install -y pipx
pipx ensurepath
```

[参考](https://pipx.pypa.io/stable/installation/)

## poetryのinstall

```
pipx install poetry
source ~/.bashrc
poetry self add poetry-plugin-shell
```

[参考](https://python-poetry.org/docs/)

## kaggle API認証情報
- 下記からkaggle.jsonに認証情報を書き込む
- デフォルトpathはインスタンスに依存するので一度適当なコマンドで認証エラーを出して参照先を確認する

```
vim path_to_kaggle_json_dir/kaggle.json
```

## コンペデータのdownload

```
kaggle competitions download -c wsdm-cup-multilingual-chatbot-arena
```

## git pull, pushできない時の対処法

```
export GIT_ASKPASS=
```

- 上記を実行してからpull, pushすると通常通りusernameとpassを聞かれるので入力する

[github token発行](https://github.com/settings/tokens)

## gitで毎回認証情報を求められないようにする

```
git config credential.helper store
```

[参考](https://qiita.com/Kamo123/items/c92b03278b6302c641e3)

# gitのidentity設定
```
git config --global user.email "yuhi.nagatsuma1996@gmail.com"
git config --global user.name "nynyny67"
```

## kaggle dataset作成

- datasetの初期化

```
kaggle datasets init -p path_to_dataset
```

- `path_to_dataset/dataset-metadata.json`ができるので内容を編集
- kaggleにdatsetを作成

```
kaggle datasets create -p path_to_dataset
```

- datasetの更新

```
kaggle datasets version -p path_to_dataset -m "default comment" --dir-mode zip
```

# openai

[参考](https://platform.openai.com/docs/quickstart)

```
# .bashrc
export OPENAI_API_KEY="hoge"
```

# GPU使用率の監視

```
watch nvidia-smi
```

# flash-attentionのinstall
[参考](https://github.com/Dao-AILab/flash-attention)

```
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

# git config

```
git config --global user.email "nynyny67@example.com"
git config --global user.name "nynyny67"
```

永続的に認証情報を保存
```
git config --global credential.helper store
```