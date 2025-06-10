import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(page_title="AutoML App", page_icon="🤖", layout="wide")

# タイトル
st.title("🤖 AutoML Application")
st.markdown("---")

# セッション状態の初期化
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# サイドバー
st.sidebar.header("📋 Settings")

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # データ読み込み
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ データが正常に読み込まれました: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # データプレビュー
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        # 基本統計量
        st.subheader("📈 Basic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**数値データの統計量:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.write("数値データがありません")
        
        with col2:
            st.write("**データ型と欠損値:**")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df)
        
        # 特徴量とラベルの選択
        st.subheader("🎯 Feature and Target Selection")
        
        # ターゲット列の選択
        target_col = st.selectbox("ターゲット列を選択してください:", df.columns.tolist())
        
        # 特徴量列の選択（ターゲット列以外）
        feature_cols = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "特徴量を選択してください:",
            feature_cols,
            default=feature_cols
        )
        
        # 問題タイプの選択
        problem_type = st.selectbox(
            "問題タイプを選択してください:",
            ["回帰", "分類"]
        )
        
        if st.button("🚀 データ処理を開始"):
            if len(selected_features) == 0:
                st.error("❌ 最低1つの特徴量を選択してください")
            else:
                with st.spinner("データを処理中..."):
                    # データの準備
                    X = df[selected_features].copy()
                    y = df[target_col].copy()
                    
                    # 欠損値の処理
                    initial_rows = len(X)
                    X = X.dropna()
                    y = y[X.index]
                    removed_rows = initial_rows - len(X)
                    
                    if removed_rows > 0:
                        st.warning(f"⚠️ {removed_rows} 行の欠損値を含むレコードを削除しました")
                    
                    # カテゴリカル変数の処理
                    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        st.info(f"📝 カテゴリカル変数をワンホットエンコーディング: {list(categorical_cols)}")
                        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    
                    # ターゲット変数の処理（分類の場合）
                    if problem_type == "分類" and y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                        st.session_state.label_encoder = le
                    
                    # データの正規化判定
                    numeric_features = X.select_dtypes(include=[np.number]).columns
                    need_scaling = False
                    
                    if len(numeric_features) > 0:
                        # スケールの差が大きい場合は正規化
                        scales = X[numeric_features].std()
                        if scales.max() / scales.min() > 10:
                            need_scaling = True
                            scaler = StandardScaler()
                            X[numeric_features] = scaler.fit_transform(X[numeric_features])
                            st.info("📊 データの正規化を実行しました")
                            st.session_state.scaler = scaler
                    
                    # 特徴量重要度分析
                    st.subheader("🎯 Feature Importance Analysis")
                    
                    if problem_type == "回帰":
                        selector = SelectKBest(score_func=f_regression, k='all')
                    else:
                        selector = SelectKBest(score_func=f_classif, k='all')
                    
                    X_scored = selector.fit_transform(X, y)
                    feature_scores = pd.DataFrame({
                        'Feature': X.columns,
                        'Score': selector.scores_
                    }).sort_values('Score', ascending=False)
                    
                    # 特徴量重要度の可視化
                    fig = px.bar(
                        feature_scores.head(15), 
                        x='Score', 
                        y='Feature',
                        orientation='h',
                        title="Top 15 Feature Importance Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 重要な特徴量の自動選択（上位70%）
                    n_features = max(1, int(len(X.columns) * 0.7))
                    top_features = feature_scores.head(n_features)['Feature'].tolist()
                    X_selected = X[top_features]
                    
                    st.info(f"🎯 上位 {n_features} 個の特徴量を自動選択しました")
                    st.write("選択された特徴量:", top_features)
                    
                    # データ分割
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_selected, y, test_size=0.2, random_state=42
                    )
                    
                    # セッション状態に保存
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.problem_type = problem_type
                    st.session_state.data_processed = True
                    
                    st.success("✅ データ処理が完了しました！")
                    st.info(f"訓練データ: {X_train.shape[0]} 行, テストデータ: {X_test.shape[0]} 行")
        
        # モデル学習セクション
        if st.session_state.data_processed:
            st.markdown("---")
            st.subheader("🤖 Model Training")
            
            if st.button("🎯 モデル学習を開始"):
                X_train = st.session_state.X_train
                X_test = st.session_state.X_test
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                problem_type = st.session_state.problem_type
                
                # モデルの定義とハイパーパラメータ
                if problem_type == "回帰":
                    models = {
                        'Random Forest': {
                            'model': RandomForestRegressor(random_state=42),
                            'params': {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [5, 10, None],
                                'min_samples_split': [2, 5]
                            }
                        },
                        'Gradient Boosting': {
                            'model': GradientBoostingRegressor(random_state=42),
                            'params': {
                                'n_estimators': [50, 100],
                                'learning_rate': [0.01, 0.1],
                                'max_depth': [3, 5]
                            }
                        },
                        'Linear Regression': {
                            'model': LinearRegression(),
                            'params': {}
                        }
                    }
                else:
                    models = {
                        'Random Forest': {
                            'model': RandomForestClassifier(random_state=42),
                            'params': {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [5, 10, None],
                                'min_samples_split': [2, 5]
                            }
                        },
                        'Gradient Boosting': {
                            'model': GradientBoostingClassifier(random_state=42),
                            'params': {
                                'n_estimators': [50, 100],
                                'learning_rate': [0.01, 0.1],
                                'max_depth': [3, 5]
                            }
                        },
                        'Logistic Regression': {
                            'model': LogisticRegression(random_state=42, max_iter=1000),
                            'params': {
                                'C': [0.1, 1, 10]
                            }
                        }
                    }
                
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, config) in enumerate(models.items()):
                    status_text.text(f"🔄 {name} を学習中...")
                    
                    try:
                        if config['params']:
                            # ハイパーパラメータ調整
                            grid_search = GridSearchCV(
                                config['model'], 
                                config['params'], 
                                cv=3, 
                                scoring='neg_mean_squared_error' if problem_type == "回帰" else 'accuracy',
                                n_jobs=-1
                            )
                            grid_search.fit(X_train, y_train)
                            best_model = grid_search.best_estimator_
                            best_params = grid_search.best_params_
                        else:
                            best_model = config['model']
                            best_model.fit(X_train, y_train)
                            best_params = {}
                        
                        # 予測
                        y_pred = best_model.predict(X_test)
                        
                        # 評価
                        if problem_type == "回帰":
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            results[name] = {
                                'model': best_model,
                                'params': best_params,
                                'predictions': y_pred,
                                'mse': mse,
                                'r2': r2,
                                'score': r2
                            }
                        else:
                            accuracy = accuracy_score(y_test, y_pred)
                            results[name] = {
                                'model': best_model,
                                'params': best_params,
                                'predictions': y_pred,
                                'accuracy': accuracy,
                                'score': accuracy
                            }
                    
                    except Exception as e:
                        st.error(f"❌ {name} の学習中にエラーが発生しました: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(models))
                
                status_text.text("✅ 全モデルの学習が完了しました！")
                
                # 結果の保存
                st.session_state.results = results
                st.session_state.models_trained = True
                
                # 最適モデルの選択
                best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
                st.session_state.best_model_name = best_model_name
                
                st.success(f"🏆 最適モデル: {best_model_name}")
        
        # 結果表示セクション
        if st.session_state.models_trained:
            st.markdown("---")
            st.subheader("📊 Model Evaluation Results")
            
            results = st.session_state.results
            y_test = st.session_state.y_test
            problem_type = st.session_state.problem_type
            
            # モデル比較表
            comparison_data = []
            for name, result in results.items():
                if problem_type == "回帰":
                    comparison_data.append({
                        'Model': name,
                        'R² Score': f"{result['r2']:.4f}",
                        'MSE': f"{result['mse']:.4f}",
                        'Best Parameters': str(result['params'])
                    })
                else:
                    comparison_data.append({
                        'Model': name,
                        'Accuracy': f"{result['accuracy']:.4f}",
                        'Best Parameters': str(result['params'])
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # 最適モデルの詳細評価
            best_model_name = st.session_state.best_model_name
            best_result = results[best_model_name]
            y_pred = best_result['predictions']
            
            st.subheader(f"🏆 Best Model: {best_model_name}")
            
            if problem_type == "回帰":
                # 散布図
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(
                        x=y_test, y=y_pred,
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title="Actual vs Predicted"
                    )
                    # 理想的な予測線を追加
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], 
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 残差プロット
                    residuals = y_test - y_pred
                    fig = px.scatter(
                        x=y_pred, y=residuals,
                        labels={'x': 'Predicted', 'y': 'Residuals'},
                        title="Residual Plot"
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                # 分類の場合
                col1, col2 = st.columns(2)
                
                with col1:
                    # 混同行列
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(
                        cm, 
                        text_auto=True, 
                        aspect="auto",
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # ROC曲線（バイナリ分類の場合）
                    if len(np.unique(y_test)) == 2:
                        y_pred_proba = best_result['model'].predict_proba(st.session_state.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC = {roc_auc:.2f})'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(dash='dash'),
                            name='Random Classifier'
                        ))
                        fig.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ROC曲線は二値分類のみ表示されます")
                
                # 分類レポート
                st.subheader("📋 Classification Report")
                if 'label_encoder' in st.session_state:
                    target_names = st.session_state.label_encoder.classes_
                    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
                else:
                    report = classification_report(y_test, y_pred, output_dict=True)
                
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ ファイルの読み込み中にエラーが発生しました: {str(e)}")

else:
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください")
    
    # サンプルデータの説明
    st.subheader("🔍 使い方")
    st.markdown("""
    1. **CSVファイルをアップロード**: 左のサイドバーからCSVファイルを選択
    2. **データ確認**: アップロードされたデータの基本統計量を確認
    3. **特徴量とターゲットを選択**: 予測に使用する特徴量と予測対象を指定
    4. **問題タイプを選択**: 回帰問題か分類問題かを選択
    5. **データ処理**: 欠損値処理、特徴量エンジニアリング、重要度分析を実行
    6. **モデル学習**: 複数のモデルを自動学習し、ハイパーパラメータを最適化
    7. **結果確認**: 各モデルの性能を比較し、最適モデルの詳細評価を確認
    """)
    
    st.subheader("📋 サポートする機能")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **データ前処理**
        - 欠損値の自動削除
        - カテゴリカル変数のワンホットエンコーディング
        - データの自動正規化
        - 特徴量重要度分析
        """)
        
    with col2:
        st.markdown("""
        **機械学習**
        - 回帰/分類問題の自動判定
        - 複数モデルの自動学習
        - ハイパーパラメータ自動最適化
        - 詳細な性能評価と可視化
        """)