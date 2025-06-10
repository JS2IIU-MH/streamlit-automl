# Streamlit x AutoML sample

## AutoML on Streamlit

このコードは、**Streamlit** を使って作成された **自動機械学習（AutoML）Webアプリケーション** です。

ユーザーがCSVファイルをアップロードするだけで、以下の機械学習の一連のプロセスを自動で実行し、結果を可視化してくれます。

1.  **データの前処理**
2.  **特徴量の選択**
3.  **複数モデルの学習と評価**
4.  **最適なモデルの特定と詳細な分析**

### **プログラムの主な流れ**

このアプリケーションは、大きく分けて以下のステップで動作します。

#### **1. 初期設定と画面構成**
* `st.set_page_config()`: Webページのタイトルやレイアウトを設定します。
* `st.title()`: アプリケーションのメインタイトルを表示します。
* `st.session_state`: ユーザーの操作状況（データ処理が完了したか、モデル学習が完了したかなど）を保存するための変数を初期化します。これにより、ページの再読み込み後も状態を維持できます。
* **サイドバー (`st.sidebar`)**: ファイルのアップロードや設定項目をここに配置しています。

#### **2. データのアップロードと基本分析**
* `st.sidebar.file_uploader`: ユーザーがCSVファイルをアップロードするためのウィジェットです。
* ファイルがアップロードされると、`pandas`でデータを読み込み、以下の情報を表示します。
    * **データプレビュー**: `st.dataframe(df.head())`でデータの先頭5行を表示。
    * **基本統計量**: 数値データの平均、標準偏差、最小値、最大値などを表示。
    * **データ型と欠損値**: 各列のデータ型、欠損値の数、欠損率を一覧で表示。

#### **3. ユーザーによる設定**
* `st.selectbox`, `st.multiselect`: ユーザーが以下の項目を選択します。
    * **ターゲット列**: 予測したい目的変数（例：「売上」「解約したかどうか」）。
    * **特徴量列**: 予測に使う説明変数（例：「広告費」「顧客の年齢」）。
    * **問題タイプ**: 「回帰」（数値を予測）か「分類」（カテゴリを予測）かを選択。

#### **4. データ処理 (`st.button("🚀 データ処理を開始")`)**
このボタンが押されると、以下の前処理が自動で実行されます。

* **欠損値の処理**: `dropna()` を使い、欠損値を含む行を単純に削除しています。
* **カテゴリカル変数の処理**: 文字列のデータ（カテゴリカル変数）を機械学習モデルが扱えるように、**ワンホットエンコーディング** (`pd.get_dummies`) を使って数値に変換します。
* **データの正規化**: `StandardScaler` を使い、各特徴量のスケール（単位や大きさ）を揃えます。これにより、モデルの学習が安定しやすくなります。
* **特徴量重要度分析**:
    * `SelectKBest` を使い、各特徴量がターゲットの予測にどれだけ重要かをスコア化します。
    * 回帰問題では `f_regression`、分類問題では `f_classif` という統計手法を用いてスコアを計算します。
    * `Plotly` を使って、スコアの高い上位15個の特徴量をグラフで可視化します。
    * **スコアの高い上位70%の特徴量を自動で選択**し、それらを使ってモデルを学習させます。
* **データ分割**: 処理済みのデータを、モデルの学習用（80%）と性能評価用のテストデータ（20%）に分割します。

#### **5. モデル学習 (`st.button("🎯 モデル学習を開始")`)**
データ処理が完了すると、このボタンが押せるようになります。

* **モデルの定義**:
    * 回帰・分類それぞれに適した複数の機械学習モデルをあらかじめ定義しています。
        * **回帰**: `RandomForestRegressor`, `GradientBoostingRegressor`, `LinearRegression`
        * **分類**: `RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`
    * `GridSearchCV` を使い、各モデルで最も性能が高くなるような**ハイパーパラメータ**（モデルの挙動を決める設定値）を自動で探索・最適化します。
* **学習と評価**:
    * 定義された全てのモデルを順番に学習させます。
    * 学習済みモデルを使ってテストデータを予測し、性能を評価します。
        * **回帰**: R2スコア（決定係数）とMSE（平均二乗誤差）を計算。
        * **分類**: Accuracy（正解率）を計算。
* **最適モデルの選択**: 全モデルの中で最も評価スコアが高かったものを「最適モデル」として特定します。

#### **6. 結果の表示と可視化**
モデル学習が完了すると、以下の評価結果が表示されます。

* **モデル比較表**: 全てのモデルの性能スコアと、最適化されたハイパーパラメータを一覧表で表示します。
* **最適モデルの詳細評価**:
    * **回帰の場合**:
        * **散布図**: 実際の値とモデルの予測値をプロット。赤い点線（理想的な予測線）に近いほど精度が高いことを示します。
        * **残差プロット**: 予測値と実際の値の誤差（残差）をプロット。誤差が0付近に均等に分布しているのが理想です。
    * **分類の場合**:
        * **混同行列 (Confusion Matrix)**: モデルがどのカテゴリをどのカテゴリと間違えたかを行列で示します。
        * **ROC曲線**: 二値分類の場合に表示され、モデルの性能を視覚的に評価する指標です。左上の角に近づくほど高性能です。
        * **分類レポート**: Accuracy（正解率）に加えて、Precision（適合率）、Recall（再現率）、F1スコアといった、より詳細な評価指標を表示します。

## Sample data generator

- [`generate_test_data.py`](src/generate_test_data.py)：テスト用データを生成します。
- 生成済みデータ
  - [`customer_churn_dataset.csv`](src/customer_churn_dataset.csv)：顧客解約。二値分類問題。
    ```plaintext
      形状: (800, 8)
      カラム: ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'contract_type', 'payment_method', 'internet_service', 'churn']
      ターゲット変数の分布:
      churn
      No     517
      Yes    283
      Name: count, dtype: int64
      欠損値:
      tenure_months      16
      monthly_charges    24
      dtype: int64
    ```
  - [`loan_approval_dataset.csv`](src/loan_approval_dataset.csv)：ローン承認。分類問題。
    ```plaintext
      形状: (1000, 11)
      カラム: ['age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'loan_amount', 'employment_years', 'employment_type', 'education_level', 'marital_status', 'property_ownership', 'loan_approved']
      ターゲット変数の分布:
      loan_approved
      Rejected    651
      Approved    349
      Name: count, dtype: int64
      欠損値:
      annual_income       40
      credit_score        20
      employment_years    30
      dtype: int64
    ```
  - [`salary_prediction_dataset.csv`](src/salary_prediction_dataset.csv)：給与予測。回帰問題。
    ```plaintext
      形状: (1000, 9)
      カラム: ['age', 'experience_years', 'education_years', 'working_hours_per_week', 'department', 'city', 'gender', 'company_size', 'annual_salary']
      ターゲット変数の統計:
      平均給与: 9,609,657 円
      給与範囲: 6,401,162 - 13,553,932 円
      欠損値:
      age                       50
      experience_years          30
      working_hours_per_week    20
      dtype: int64
    ```
