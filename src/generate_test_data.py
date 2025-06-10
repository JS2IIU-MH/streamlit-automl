import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# シード値設定（再現性のため）
np.random.seed(42)
random.seed(42)

def generate_regression_data(n_samples=1000):
    """回帰問題用のテストデータを生成"""
    
    # 基本的な数値特徴量
    age = np.random.randint(22, 65, n_samples)  # 22歳以上に変更
    
    # 経験年数の安全な生成
    max_experience = age - 18  # 18歳から働き始めると仮定
    experience = np.array([
        np.random.randint(0, max(1, max_exp)) for max_exp in max_experience
    ])
    
    education_years = np.random.choice([12, 14, 16, 18, 20], n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05])
    
    # 勤務時間（週）
    working_hours = np.random.normal(40, 8, n_samples)
    working_hours = np.clip(working_hours, 20, 60)  # 20-60時間の範囲
    
    # カテゴリカル特徴量
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
    department = np.random.choice(departments, n_samples)
    
    cities = ['Tokyo', 'Osaka', 'Kyoto', 'Yokohama', 'Kobe']
    city = np.random.choice(cities, n_samples)
    
    genders = ['Male', 'Female']
    gender = np.random.choice(genders, n_samples)
    
    company_sizes = ['Small', 'Medium', 'Large']
    company_size = np.random.choice(company_sizes, n_samples, p=[0.3, 0.4, 0.3])
    
    # 一部の値を欠損させる
    age_with_missing = age.copy().astype(float)
    experience_with_missing = experience.copy().astype(float)
    working_hours_with_missing = working_hours.copy()
    
    # 5%の欠損値を作成
    missing_indices_age = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    missing_indices_exp = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    missing_indices_hours = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    
    age_with_missing[missing_indices_age] = np.nan
    experience_with_missing[missing_indices_exp] = np.nan
    working_hours_with_missing[missing_indices_hours] = np.nan
    
    # ターゲット変数（給与）を生成 - 特徴量に依存
    salary_base = 3000000  # 基本給与（300万円）
    salary = (salary_base + 
              age * 50000 +  # 年齢効果
              experience * 80000 +  # 経験効果
              education_years * 150000 +  # 学歴効果
              working_hours * 10000 +  # 労働時間効果
              np.where(department == 'Engineering', 800000, 0) +  # エンジニアボーナス
              np.where(department == 'Sales', 400000, 0) +  # セールスボーナス
              np.where(department == 'Finance', 600000, 0) +  # ファイナンスボーナス
              np.where(city == 'Tokyo', 500000, 0) +  # 東京手当
              np.where(city == 'Osaka', 200000, 0) +  # 大阪手当
              np.where(company_size == 'Large', 800000, 0) +  # 大企業ボーナス
              np.where(company_size == 'Medium', 300000, 0) +  # 中企業ボーナス
              np.random.normal(0, 300000, n_samples))  # ノイズ
    
    # 負の給与を防ぐ
    salary = np.maximum(salary, 2000000)  # 最低200万円
    
    df = pd.DataFrame({
        'age': age_with_missing,
        'experience_years': experience_with_missing,
        'education_years': education_years,
        'working_hours_per_week': working_hours_with_missing,
        'department': department,
        'city': city,
        'gender': gender,
        'company_size': company_size,
        'annual_salary': salary.astype(int)
    })
    
    return df

def generate_classification_data(n_samples=1000):
    """分類問題用のテストデータを生成"""
    
    # 基本的な数値特徴量
    age = np.random.randint(20, 70, n_samples)
    income = np.random.normal(5000000, 2000000, n_samples)  # 年収（円）
    income = np.maximum(income, 1000000)  # 最低100万円
    
    credit_score = np.random.randint(300, 850, n_samples)
    debt_ratio = np.random.uniform(0, 1, n_samples)
    
    # ローン申請額
    loan_amount = np.random.uniform(1000000, 50000000, n_samples)  # 100万〜5000万円
    
    # 勤続年数
    employment_years = np.random.randint(0, 30, n_samples)
    
    # カテゴリカル特徴量
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    employment = np.random.choice(employment_types, n_samples, p=[0.6, 0.2, 0.15, 0.05])
    
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = np.random.choice(education_levels, n_samples, p=[0.3, 0.4, 0.25, 0.05])
    
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1])
    
    property_ownership = np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples, p=[0.3, 0.4, 0.3])
    
    # 一部の値を欠損させる
    income_with_missing = income.copy()
    credit_score_with_missing = credit_score.copy().astype(float)
    employment_years_with_missing = employment_years.copy().astype(float)
    
    # 欠損値を作成
    missing_indices_income = np.random.choice(n_samples, size=int(n_samples * 0.04), replace=False)
    missing_indices_credit = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    missing_indices_emp_years = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    
    income_with_missing[missing_indices_income] = np.nan
    credit_score_with_missing[missing_indices_credit] = np.nan
    employment_years_with_missing[missing_indices_emp_years] = np.nan
    
    # ターゲット変数（ローン承認）を生成 - 特徴量に依存
    # 収入対ローン比率
    income_loan_ratio = income / loan_amount
    
    approval_probability = (
        0.1 +  # 基本確率
        (credit_score - 300) / 550 * 0.3 +  # クレジットスコア効果
        np.where(income > 6000000, 0.2, 0) +  # 高収入ボーナス（600万円以上）
        np.where(debt_ratio < 0.3, 0.15, -0.15) +  # 低負債比率ボーナス
        np.where(employment == 'Full-time', 0.15, 0) +  # フルタイム雇用ボーナス
        np.where(employment == 'Unemployed', -0.3, 0) +  # 無職ペナルティ
        np.where(employment_years > 5, 0.1, 0) +  # 長期勤続ボーナス
        np.where(education == 'Bachelor', 0.05, 0) +  # 学士号ボーナス
        np.where(education == 'Master', 0.08, 0) +  # 修士号ボーナス
        np.where(education == 'PhD', 0.1, 0) +  # 博士号ボーナス
        np.where(marital_status == 'Married', 0.08, 0) +  # 既婚ボーナス
        np.where(property_ownership == 'Own', 0.12, 0) +  # 持ち家ボーナス
        np.where(income_loan_ratio > 5, 0.15, 0) +  # 収入対ローン比率ボーナス
        np.where(income_loan_ratio < 1, -0.2, 0)  # 収入対ローン比率ペナルティ
    )
    
    # 確率を0-1の範囲に制限
    approval_probability = np.clip(approval_probability, 0, 1)
    
    # 確率に基づいてローン承認を決定
    loan_approved = np.random.binomial(1, approval_probability, n_samples)
    loan_status = ['Rejected', 'Approved']
    loan_result = [loan_status[x] for x in loan_approved]
    
    df = pd.DataFrame({
        'age': age,
        'annual_income': income_with_missing,
        'credit_score': credit_score_with_missing,
        'debt_to_income_ratio': debt_ratio,
        'loan_amount': loan_amount,
        'employment_years': employment_years_with_missing,
        'employment_type': employment,
        'education_level': education,
        'marital_status': marital_status,
        'property_ownership': property_ownership,
        'loan_approved': loan_result
    })
    
    return df

def generate_binary_classification_data(n_samples=800):
    """二値分類問題用のシンプルなデータセット（顧客解約予測）"""
    
    # 数値特徴量
    age = np.random.randint(18, 80, n_samples)
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = monthly_charges * np.random.uniform(1, 60, n_samples)  # 1-60ヶ月
    
    # サービス利用期間（月）
    tenure_months = np.random.randint(1, 72, n_samples)
    
    # カテゴリカル特徴量
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract = np.random.choice(contract_types, n_samples, p=[0.5, 0.3, 0.2])
    
    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    payment_method = np.random.choice(payment_methods, n_samples)
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
    
    # 欠損値を含む特徴量
    monthly_charges_missing = monthly_charges.copy()
    tenure_missing = tenure_months.copy().astype(float)
    
    missing_idx1 = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    missing_idx2 = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    
    monthly_charges_missing[missing_idx1] = np.nan
    tenure_missing[missing_idx2] = np.nan
    
    # ターゲット変数（解約）の生成
    churn_probability = (
        0.1 +  # 基本確率
        np.where(contract == 'Month-to-month', 0.3, 0) +  # 月契約は解約しやすい
        np.where(contract == 'Two year', -0.15, 0) +  # 2年契約は解約しにくい
        np.where(monthly_charges > 80, 0.2, 0) +  # 高額利用者は解約しやすい
        np.where(tenure_months < 12, 0.25, 0) +  # 短期利用者は解約しやすい
        np.where(tenure_months > 48, -0.2, 0) +  # 長期利用者は解約しにくい
        np.where(payment_method == 'Electronic check', 0.15, 0) +  # 電子チェックは解約しやすい
        np.where(internet_service == 'Fiber optic', 0.1, 0)  # 光ファイバーは解約しやすい
    )
    
    churn_probability = np.clip(churn_probability, 0, 1)
    churn_binary = np.random.binomial(1, churn_probability, n_samples)
    churn_result = ['No', 'Yes']
    churn = [churn_result[x] for x in churn_binary]
    
    df = pd.DataFrame({
        'age': age,
        'tenure_months': tenure_missing,
        'monthly_charges': monthly_charges_missing,
        'total_charges': total_charges,
        'contract_type': contract,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'churn': churn
    })
    
    return df

# データセット生成
print("テスト用CSVファイルを生成中...")

try:
    # 1. 回帰問題用データセット（給与予測）
    print("📊 回帰データセットを生成中...")
    regression_df = generate_regression_data(1000)
    regression_df.to_csv('salary_prediction_dataset.csv', index=False)
    print("✅ salary_prediction_dataset.csv を生成しました")

    # 2. 多クラス分類問題用データセット（ローン承認予測）
    print("📊 多クラス分類データセットを生成中...")
    classification_df = generate_classification_data(1000)
    classification_df.to_csv('loan_approval_dataset.csv', index=False)
    print("✅ loan_approval_dataset.csv を生成しました")

    # 3. 二値分類問題用データセット（顧客解約予測）
    print("📊 二値分類データセットを生成中...")
    binary_df = generate_binary_classification_data(800)
    binary_df.to_csv('customer_churn_dataset.csv', index=False)
    print("✅ customer_churn_dataset.csv を生成しました")

    # データの概要を表示
    print("\n" + "="*60)
    print("📊 回帰データセット（給与予測）の概要:")
    print(f"形状: {regression_df.shape}")
    print("カラム:", list(regression_df.columns))
    print("ターゲット変数の統計:")
    print(f"平均給与: {regression_df['annual_salary'].mean():,.0f} 円")
    print(f"給与範囲: {regression_df['annual_salary'].min():,.0f} - {regression_df['annual_salary'].max():,.0f} 円")
    print("欠損値:")
    print(regression_df.isnull().sum()[regression_df.isnull().sum() > 0])

    print("\n" + "="*60)
    print("📊 分類データセット（ローン承認）の概要:")
    print(f"形状: {classification_df.shape}")
    print("カラム:", list(classification_df.columns))
    print("ターゲット変数の分布:")
    print(classification_df['loan_approved'].value_counts())
    print("欠損値:")
    print(classification_df.isnull().sum()[classification_df.isnull().sum() > 0])

    print("\n" + "="*60)
    print("📊 二値分類データセット（顧客解約）の概要:")
    print(f"形状: {binary_df.shape}")
    print("カラム:", list(binary_df.columns))
    print("ターゲット変数の分布:")
    print(binary_df['churn'].value_counts())
    print("欠損値:")
    print(binary_df.isnull().sum()[binary_df.isnull().sum() > 0])

    print("\n" + "="*60)
    print("🎯 生成完了！以下のファイルがAutoMLアプリで使用できます:")
    print("1. salary_prediction_dataset.csv (回帰問題)")
    print("   - ターゲット: annual_salary")
    print("   - 特徴量: age, experience_years, education_years, working_hours_per_week, department, city, gender, company_size")
    print()
    print("2. loan_approval_dataset.csv (多クラス分類問題)")
    print("   - ターゲット: loan_approved")
    print("   - 特徴量: age, annual_income, credit_score, debt_to_income_ratio, loan_amount, employment_years, employment_type, education_level, marital_status, property_ownership")
    print()
    print("3. customer_churn_dataset.csv (二値分類問題)")
    print("   - ターゲット: churn")
    print("   - 特徴量: age, tenure_months, monthly_charges, total_charges, contract_type, payment_method, internet_service")

except Exception as e:
    print(f"❌ エラーが発生しました: {str(e)}")
    import traceback
    traceback.print_exc()
