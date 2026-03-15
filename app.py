import streamlit as st
import pandas as pd
import numpy as np
import linearmodels as plm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="面板数据智能分析工具", layout="wide")

# 页面标题
st.title("🤖 面板数据智能分析工具（Streamlit Cloud版）")

# 1. 数据上传
st.sidebar.header("📥 数据上传")
uploaded_file = st.sidebar.file_uploader("上传Excel/CSV文件", type=["xlsx", "csv"])

if uploaded_file:
    # 读取数据
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding='gbk')  # 适配中文编码
    st.sidebar.success("✅ 数据上传成功！")
    st.subheader("原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    # 2. 变量配置
    st.sidebar.header("⚙️ 变量配置")
    cols = df.columns.tolist()
    id_col = st.sidebar.selectbox("个体列（如firm_id）", cols)
    time_col = st.sidebar.selectbox("时间列（如year）", cols)
    y_col = st.sidebar.selectbox("被解释变量Y", cols)
    X_cols = st.sidebar.multiselect("解释变量X（多选）", [c for c in cols if c not in [id_col, time_col, y_col]])

    # 3. 一键分析
    if st.sidebar.button("🚀 一键全分析", type="primary"):
        with st.spinner("正在分析，请稍等..."):
            # 数据预处理
            st.subheader("📋 数据预处理报告")
            core_vars = [id_col, time_col, y_col] + X_cols
            df_clean = df[core_vars].copy()
            
            # 缺失值填充
            df_clean = df_clean.groupby(id_col, group_keys=False).apply(lambda g: g.fillna(g.median()))
            # 1%缩尾
            numeric_vars = [y_col] + X_cols
            for var in numeric_vars:
                df_clean[var] = df_clean[var].clip(df_clean[var].quantile(0.01), df_clean[var].quantile(0.99))
            
            # VIF共线性检验
            if len(X_cols)>=2:
                vif_df = pd.DataFrame({
                    '变量': X_cols,
                    'VIF值': [variance_inflation_factor(df_clean[X_cols].values, i) for i in range(len(X_cols))]
                })
                st.write("### 多重共线性检验（VIF）")
                st.dataframe(vif_df.round(2), use_container_width=True)

            # 计量模型
            st.subheader("📊 固定效应回归结果")
            panel_df = df_clean.set_index([id_col, time_col])
            formula = f"{y_col} ~ 1 + {' + '.join(X_cols)}"
            fe_model = plm.PanelOLS.from_formula(formula, panel_df).fit(cov_type='clustered')
            st.text(fe_model.summary.as_text())

            # 机器学习
            st.subheader("🤖 机器学习模型（XGBoost）")
            X_ml = df_clean[X_cols]
            y_ml = df_clean[y_col]
            X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
            xgb = XGBRegressor(random_state=42)
            xgb.fit(X_train, y_train)
            test_r2 = r2_score(y_test, xgb.predict(X_test))
            st.write(f"XGBoost测试集R²：{test_r2:.4f}")

            # 变量重要性
            st.write("### 变量重要性排序")
            imp_df = pd.DataFrame({
                '变量': X_cols,
                '重要性': xgb.feature_importances_
            }).sort_values('重要性', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(imp_df['变量'][:10], imp_df['重要性'][:10], color='#3498db')
            st.pyplot(fig)

            st.success("✅ 全流程分析完成！结果可直接复制到论文/报告中。")
else:
    st.info("请先在左侧侧边栏上传面板数据文件（Excel/CSV格式）")                                                                       
