import streamlit as st
import pandas as pd
import numpy as np
import linearmodels as plm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="Advanced Panel Data Analysis Tool", layout="wide")

# 页面标题
st.title("🤖 进阶版面板数据分析工具（含DID/中介/调节效应）")

# ====================== 新增：数据整合前置环节 ======================
st.sidebar.header("📥 数据预处理与整合")
integrate_mode = st.sidebar.radio("数据来源模式", ["单文件上传", "多文件整合"])

# 多文件整合逻辑
if integrate_mode == "多文件整合":
    st.sidebar.subheader("上传多个Excel/CSV文件（同结构）")
    uploaded_files = st.sidebar.file_uploader(
        "选择多个文件", type=["xlsx", "csv"], accept_multiple_files=True
    )
    if uploaded_files:
        df_list = []
        for file in uploaded_files:
            if file.name.endswith(".xlsx"):
                temp_df = pd.read_excel(file)
            else:
                temp_df = pd.read_csv(file, encoding='gbk')
            df_list.append(temp_df)
        # 数据整合（合并+去重+补全）
        df = pd.concat(df_list, ignore_index=True).drop_duplicates()
        # 缺失值初步提示
        missing_info = df.isnull().sum()[df.isnull().sum() > 0]
        if not missing_info.empty:
            st.sidebar.warning("⚠️ 整合后数据存在缺失值：")
            for col, num in missing_info.items():
                st.sidebar.write(f"- {col}: {num} 个缺失值")
        st.sidebar.success(f"✅ 已整合 {len(uploaded_files)} 个文件，共 {len(df)} 条记录！")
else:
    # 原单文件上传逻辑
    uploaded_file = st.sidebar.file_uploader("上传Excel/CSV文件", type=["xlsx", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, encoding='gbk')
        st.sidebar.success("✅ 数据上传成功！")

# 数据预览（整合/上传后）
if 'df' in locals() and not df.empty:
    st.subheader("📋 整合后原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    # ====================== 变量配置（保留原逻辑+新增DID/中介变量） ======================
    st.sidebar.header("⚙️ 变量配置")
    cols = df.columns.tolist()
    id_col = st.sidebar.selectbox("个体列（如province_id）", cols)
    time_col = st.sidebar.selectbox("时间列（如year）", cols)
    y_col = st.sidebar.selectbox("被解释变量Y", cols)
    X_cols = st.sidebar.multiselect("核心解释变量X（多选）", [c for c in cols if c not in [id_col, time_col, y_col]])
    
    # 新增：DID/中介/调节效应变量配置
    did_treat_col = st.sidebar.selectbox("DID-处理组标识列（1=处理组，0=对照组）", [""] + cols)
    did_policy_col = st.sidebar.selectbox("DID-政策冲击列（1=政策后，0=政策前）", [""] + cols)
    mediator_col = st.sidebar.selectbox("中介效应-中介变量", [""] + cols)
    moderator_col = st.sidebar.selectbox("调节效应-调节变量", [""] + cols)

    # ====================== 一键全分析（新增DID/中介/调节+丰富机器学习） ======================
    if st.sidebar.button("🚀 一键全分析", type="primary"):
        with st.spinner("正在执行深度分析，请稍等..."):
            # 1. 数据预处理（保留原逻辑+优化）
            st.subheader("📊 数据预处理报告")
            core_vars = [id_col, time_col, y_col] + X_cols
            if did_treat_col: core_vars.append(did_treat_col)
            if did_policy_col: core_vars.append(did_policy_col)
            if mediator_col: core_vars.append(mediator_col)
            if moderator_col: core_vars.append(moderator_col)
            
            df_clean = df[core_vars].copy()
            # 缺失值填充（分组中位数）
            df_clean = df_clean.groupby(id_col, group_keys=False).apply(lambda g: g.fillna(g.median()))
            # 1%缩尾处理
            numeric_vars = [y_col] + X_cols
            if mediator_col: numeric_vars.append(mediator_col)
            if moderator_col: numeric_vars.append(moderator_col)
            
            for var in numeric_vars:
                if df_clean[var].dtype in [np.float64, np.int64]:
                    df_clean[var] = df_clean[var].clip(df_clean[var].quantile(0.01), df_clean[var].quantile(0.99))
            
            # 多重共线性检验（VIF）
            if len(X_cols)>=2:
                vif_df = pd.DataFrame({
                    '变量': X_cols,
                    'VIF值': [variance_inflation_factor(df_clean[X_cols].dropna().values, i) for i in range(len(X_cols))]
                })
                st.write("### 多重共线性检验（VIF）")
                st.dataframe(vif_df.round(2), use_container_width=True)

            # 2. 传统计量模型（原固定效应+新增DID）
            st.subheader("📈 计量模型分析")
            panel_df = df_clean.set_index([id_col, time_col])
            
            # 2.1 固定效应回归（原逻辑）
            st.write("#### 固定效应回归结果")
            fe_formula = f"{y_col} ~ 1 + {' + '.join(X_cols)}"
            fe_model = plm.PanelOLS.from_formula(fe_formula, panel_df).fit(cov_type='clustered')
            st.text(fe_model.summary.as_text())
            
            # 2.2 DID双重差分分析（新增）
            if did_treat_col and did_policy_col:
                st.write("#### DID双重差分回归结果")
                # 构造交互项
                panel_df['did_interaction'] = panel_df[did_treat_col] * panel_df[did_policy_col]
                did_formula = f"{y_col} ~ 1 + {did_treat_col} + {did_policy_col} + did_interaction + {' + '.join(X_cols)}"
                did_model = plm.PanelOLS.from_formula(did_formula, panel_df).fit(cov_type='clustered')
                st.text(did_model.summary.as_text())
                # DID核心结果提取
                did_coef = did_model.params['did_interaction']
                did_pval = did_model.pvalues['did_interaction']
                st.write(f"**DID核心结论**：政策处理效应系数 = {did_coef:.4f}，P值 = {did_pval:.4f}")

            # 2.3 中介效应分析（新增：逐步法）
            if mediator_col and len(X_cols)>=1:
                st.write("#### 中介效应分析（逐步回归法）")
                # 步骤1：Y ~ X
                step1_formula = f"{y_col} ~ 1 + {X_cols[0]} + {' + '.join(X_cols[1:])}"
                step1_model = plm.PanelOLS.from_formula(step1_formula, panel_df).fit()
                # 步骤2：M ~ X
                step2_formula = f"{mediator_col} ~ 1 + {X_cols[0]} + {' + '.join(X_cols[1:])}"
                step2_model = plm.PanelOLS.from_formula(step2_formula, panel_df).fit()
                # 步骤3：Y ~ X + M
                step3_formula = f"{y_col} ~ 1 + {X_cols[0]} + {' + '.join(X_cols[1:])} + {mediator_col}"
                step3_model = plm.PanelOLS.from_formula(step3_formula, panel_df).fit()
                
                # 展示关键结果
                mediator_result = pd.DataFrame({
                    '步骤': ['Y~X', 'M~X', 'Y~X+M'],
                    f'{X_cols[0]}系数': [step1_model.params[X_cols[0]], step2_model.params[X_cols[0]], step3_model.params[X_cols[0]]],
                    f'{X_cols[0]}P值': [step1_model.pvalues[X_cols[0]], step2_model.pvalues[X_cols[0]], step3_model.pvalues[X_cols[0]]],
                    f'{mediator_col}系数': [np.nan, np.nan, step3_model.params[mediator_col]],
                    f'{mediator_col}P值': [np.nan, np.nan, step3_model.pvalues[mediator_col]]
                }).round(4)
                st.dataframe(mediator_result, use_container_width=True)
                # 中介效应判断
                if (step1_model.pvalues[X_cols[0]] < 0.05 and 
                    step2_model.pvalues[X_cols[0]] < 0.05 and 
                    step3_model.pvalues[mediator_col] < 0.05):
                    if abs(step3_model.params[X_cols[0]]) < abs(step1_model.params[X_cols[0]]):
                        st.success(f"✅ {mediator_col} 存在部分中介效应")
                    else:
                        st.success(f"✅ {mediator_col} 存在完全中介效应")
                else:
                    st.warning(f"⚠️ {mediator_col} 中介效应不显著")

            # 2.4 调节效应分析（新增）
            if moderator_col and len(X_cols)>=1:
                st.write("#### 调节效应分析")
                # 构造交互项
                panel_df['mod_interaction'] = panel_df[X_cols[0]] * panel_df[moderator_col]
                mod_formula = f"{y_col} ~ 1 + {X_cols[0]} + {moderator_col} + mod_interaction + {' + '.join(X_cols[1:])}"
                mod_model = plm.PanelOLS.from_formula(mod_formula, panel_df).fit(cov_type='clustered')
                st.text(mod_model.summary.as_text())
                # 调节效应判断
                mod_pval = mod_model.pvalues['mod_interaction']
                if mod_pval < 0.05:
                    st.success(f"✅ {moderator_col} 调节效应显著（P值={mod_pval:.4f}）")
                else:
                    st.warning(f"⚠️ {moderator_col} 调节效应不显著（P值={mod_pval:.4f}）")

            # 3. 机器学习模型（丰富模型+对比+可视化）
            st.subheader("🤖 机器学习模型对比分析")
            X_ml = df_clean[X_cols].dropna()
            y_ml = df_clean[y_col].loc[X_ml.index]
            X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
            
            # 定义模型池
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(alpha=0.1, random_state=42),
                "Ridge Regression": Ridge(alpha=0.1, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "XGBoost": XGBRegressor(random_state=42),
                "LightGBM": LGBMRegressor(random_state=42)
            }
            
            # 模型训练与评估
            model_metrics = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                model_metrics.append({
                    "模型": name,
                    "R²": r2,
                    "RMSE": rmse,
                    "MAE": mae
                })
            
            # 展示模型对比结果
            metrics_df = pd.DataFrame(model_metrics).round(4)
            st.dataframe(metrics_df.sort_values("R²", ascending=False), use_container_width=True)
            
            # 最优模型变量重要性
            best_model_name = metrics_df.loc[metrics_df['R²'].idxmax(), '模型']
            best_model = models[best_model_name]
            st.write(f"#### 最优模型（{best_model_name}）变量重要性")
            
            if hasattr(best_model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    '变量': X_cols,
                    '重要性': best_model.feature_importances_
                }).sort_values('重要性', ascending=False)
                
                # 可视化
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(imp_df['变量'][:10], imp_df['重要性'][:10], color='#2ecc71')
                ax.set_xlabel('Importance')
                ax.set_title(f'{best_model_name} Feature Importance (Top 10)')
                st.pyplot(fig)
            else:
                st.write("该模型无变量重要性指标（线性模型可参考系数绝对值）")

            st.success("✅ 深度分析完成！所有结果可直接用于论文/报告撰写。")
else:
    st.info("请先在左侧侧边栏上传/整合面板数据文件（Excel/CSV格式）")
    
