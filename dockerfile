# ベースイメージとしてPythonのオフィシャルイメージを使用
FROM python:3.9

# 作業ディレクトリを設定
WORKDIR /usr/src/app

# 必要なPythonライブラリをファイルでコピー
COPY requirements.txt ./

# pipのアップグレード
RUN pip install --upgrade pip

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# requirements.txtにリストされたPythonライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのソースコードをコピー
COPY . .

# コンテナ起動時にデフォルトで実行されるコマンド
# この例では、Pythonの対話モードを起動する（パッケージに合わせて適宜変更）
CMD ["python"]
