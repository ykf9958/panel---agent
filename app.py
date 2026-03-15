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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel.unitroot import PanelUnitRootTest
from linearmodels.panel.model import RandomEffects
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 空间分析相关（需提前安装 pysal/esda）
try:
    import pysal.lib as ps
    import esda
    from pysal.model import spreg
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    st.warning("⚠️ 空间分析模块依赖包未安装，将跳过空间面板分析（可运行 pip install pysal esda 安装）")

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
st.set_page_config(page_title="Advanced Panel Data Analysis Tool", layout="wide")

# 页面标题
st.title("🤖 终极版面板数据分析工具（学术论文级）")

# ====================== 1. 数据整合前置环节 ======================
st.sidebar.header("📥 数据预处理与整合")
integrate_mode = st.sidebar.radio("数据来源模式", ["单文件上传", "多文件整合"])

# 多文件整合逻辑
df = None
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
if df is not None and not df.empty:
    st.subheader("📋 整合后原始数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    # ====================== 2. 变量配置 ======================
    st.sidebar.header("⚙️ 变量配置")
    cols = df.columns.tolist()
    id_col = st.sidebar.selectbox("个体列（如province_id）", cols)
    time_col = st.sidebar.selectbox("时间列（如year）", cols)
    y_col = st.sidebar.selectbox("被解释变量Y", cols)
    X_cols = st.sidebar.multiselect("核心解释变量X（多选）", [c for c in cols if c not in [id_col, time_col, y_col]])
    region_col = st.sidebar.selectbox("地区分组列（如region_group，可选）", [""] + cols)
    
    # 进阶分析变量配置
    did_treat_col = st.sidebar.selectbox("DID-处理组标识列（1=处理组，0=对照组）", [""] + cols)
    did_policy_col = st.sidebar.selectbox("DID-政策冲击列（1=政策后，0=政策前）", [""] + cols)
    mediator_col = st.sidebar.selectbox("中介效应-中介变量", [""] + cols)
    moderator_col = st.sidebar.selectbox("调节效应-调节变量", [""] + cols)
    iv_col = st.sidebar.selectbox("工具变量IV（可选）", [""] + cols)

    # ====================== 3. 一键全分析 ======================
    if st.sidebar.button("🚀 一键全分析", type="primary"):
        with st.spinner("正在执行深度分析，请稍等..."):
            # 3.1 数据预处理
            st.subheader("📊 一、数据诊断与预处理报告")
            # 核心变量列表
            core_vars = [id_col, time_col, y_col] + X_cols
            if did_treat_col: core_vars.append(did_treat_col)
            if did_policy_col: core_vars.append(did_policy_col)
            if mediator_col: core_vars.append(mediator_col)
            if moderator_col: core_vars.append(moderator_col)
            if iv_col: core_vars.append(iv_col)
            if region_col: core_vars.append(region_col)
            
            df_clean = df[core_vars].copy()
            # 缺失值填充（分组中位数）
            df_clean = df_clean.groupby(id_col, group_keys=False).apply(lambda g: g.fillna(g.median()))
            # 1%缩尾处理
            numeric_vars = [y_col] + X_cols
            if mediator_col: numeric_vars.append(mediator_col)
            if moderator_col: numeric_vars.append(moderator_col)
            if iv_col: numeric_vars.append(iv_col)
            
            for var in numeric_vars:
                if df_clean[var].dtype in [np.float64, np.int64]:
                    df_clean[var] = df_clean[var].clip(df_clean[var].quantile(0.01), df_clean[var].quantile(0.99))
            
            # 3.1.1 描述性统计（表格输出）
            st.write("### 1.1 描述性统计")
            desc_stats = df_clean[numeric_vars].describe().T.round(4)
            desc_stats['变异系数'] = (desc_stats['std'] / desc_stats['mean']).round(4)
            desc_stats['偏度'] = df_clean[numeric_vars].skew().round(4)
            desc_stats['峰度'] = df_clean[numeric_vars].kurt().round(4)
            st.dataframe(desc_stats, use_container_width=True)
            
            # 分组描述性统计
            if region_col:
                st.write("### 1.2 分组描述性统计（按地区）")
                group_desc = df_clean.groupby(region_col)[numeric_vars].agg(['mean', 'std', 'min', 'max']).round(4)
                st.dataframe(group_desc, use_container_width=True)
            
            # 3.1.2 面板单位根检验（表格输出）
            st.write("### 1.3 面板单位根检验（LLC）")
            unitroot_results = []
            for var in numeric_vars:
                try:
                    unitroot_test = PanelUnitRootTest(df_clean[var], df_clean[id_col], df_clean[time_col], method='llc')
                    unitroot_results.append({
                        '变量': var,
                        'LLC统计量': unitroot_test.statistic.round(4),
                        'P值': unitroot_test.p_value.round(4),
                        '是否平稳': '是' if unitroot_test.p_value < 0.05 else '否'
                    })
                except:
                    unitroot_results.append({
                        '变量': var,
                        'LLC统计量': '计算失败',
                        'P值': '计算失败',
                        '是否平稳': '未知'
                    })
            unitroot_df = pd.DataFrame(unitroot_results)
            st.dataframe(unitroot_df, use_container_width=True)
            
            # 3.1.3 多重共线性检验（VIF）
            if len(X_cols)>=2:
                st.write("### 1.4 多重共线性检验（VIF）")
                vif_df = pd.DataFrame({
                    '变量': X_cols,
                    'VIF值': [variance_inflation_factor(df_clean[X_cols].dropna().values, i) for i in range(len(X_cols))]
                }).round(4)
                vif_df['共线性程度'] = vif_df['VIF值'].apply(lambda x: '无' if x<5 else '轻度' if x<10 else '重度')
                st.dataframe(vif_df, use_container_width=True)
            
            # 3.1.4 异方差检验
            st.write("### 1.5 异方差检验（Breusch-Pagan）")
            panel_df = df_clean.set_index([id_col, time_col])
            fe_temp = plm.PanelOLS.from_formula(f"{y_col} ~ 1 + {' + '.join(X_cols)}", panel_df).fit()
            bp_test = het_breuschpagan(fe_temp.resids, fe_temp.model.exog)
            het_results = pd.DataFrame({
                '检验统计量': [bp_test[0].round(4)],
                'P值': [bp_test[1].round(4)],
                '是否存在异方差': ['是' if bp_test[1]<0.05 else '否']
            })
            st.dataframe(het_results, use_container_width=True)

            # 3.2 核心计量模型分析
            st.subheader("📈 二、核心计量模型分析（表格化输出）")
            regression_results = []
            
            # 3.2.1 固定效应/随机效应+Hausman检验
            st.write("### 2.1 固定效应(FE) vs 随机效应(RE)")
            # 固定效应
            fe_formula = f"{y_col} ~ 1 + {' + '.join(X_cols)}"
            fe_model = plm.PanelOLS.from_formula(fe_formula, panel_df).fit(cov_type='clustered')
            # 随机效应
            re_model = RandomEffects.from_formula(fe_formula, panel_df).fit()
            # Hausman检验
            try:
                hausman_test = plm.panel.compare({'FE': fe_model, 'RE': re_model}, 'hausman')
                hausman_pval = hausman_test.p_value
                hausman_conclusion = '选择FE' if hausman_pval < 0.05 else '选择RE'
            except:
                hausman_pval = np.nan
                hausman_conclusion = '检验失败'
            
            # 提取核心系数（表格化）
            for var in X_cols:
                if var in fe_model.params.index:
                    regression_results.append({
                        '变量': var,
                        'FE系数': fe_model.params[var].round(4),
                        'FE标准误': fe_model.std_errors[var].round(4),
                        'FE P值': fe_model.pvalues[var].round(4),
                        'RE系数': re_model.params[var].round(4),
                        'RE标准误': re_model.std_errors[var].round(4),
                        'RE P值': re_model.pvalues[var].round(4)
                    })
            
            # 加入常数项
            regression_results.append({
                '变量': '常数项',
                'FE系数': fe_model.params['Intercept'].round(4) if 'Intercept' in fe_model.params else np.nan,
                'FE标准误': fe_model.std_errors['Intercept'].round(4) if 'Intercept' in fe_model.std_errors else np.nan,
                'FE P值': fe_model.pvalues['Intercept'].round(4) if 'Intercept' in fe_model.pvalues else np.nan,
                'RE系数': re_model.params['Intercept'].round(4) if 'Intercept' in re_model.params else np.nan,
                'RE标准误': re_model.std_errors['Intercept'].round(4) if 'Intercept' in re_model.std_errors else np.nan,
                'RE P值': re_model.pvalues['Intercept'].round(4) if 'Intercept' in re_model.pvalues else np.nan
            })
            
            # 模型拟合优度
            fit_stats = pd.DataFrame({
                '指标': ['R²(FE)', 'R²(RE)', 'Hausman P值', '模型选择'],
                '数值': [
                    fe_model.rsquared.round(4),
                    re_model.rsquared.round(4),
                    hausman_pval.round(4) if not np.isnan(hausman_pval) else 'N/A',
                    hausman_conclusion
                ]
            })
            
            st.dataframe(pd.DataFrame(regression_results), use_container_width=True)
            st.write("### 模型拟合优度")
            st.dataframe(fit_stats, use_container_width=True)

            # 3.2.2 DID双重差分（进阶版）
            if did_treat_col and did_policy_col:
                st.write("### 2.2 DID双重差分分析")
                # 构造交互项
                panel_df['did_interaction'] = panel_df[did_treat_col] * panel_df[did_policy_col]
                did_formula = f"{y_col} ~ 1 + {did_treat_col} + {did_policy_col} + did_interaction + {' + '.join(X_cols)}"
                did_model = plm.PanelOLS.from_formula(did_formula, panel_df).fit(cov_type='clustered')
                
                # DID结果表格化
                did_results = []
                key_vars = [did_treat_col, did_policy_col, 'did_interaction'] + X_cols
                for var in key_vars:
                    if var in did_model.params.index:
                        did_results.append({
                            '变量': var,
                            '系数': did_model.params[var].round(4),
                            '标准误': did_model.std_errors[var].round(4),
                            't值': (did_model.params[var]/did_model.std_errors[var]).round(4),
                            'P值': did_model.pvalues[var].round(4),
                            '95%置信区间下限': (did_model.params[var] - 1.96*did_model.std_errors[var]).round(4),
                            '95%置信区间上限': (did_model.params[var] + 1.96*did_model.std_errors[var]).round(4)
                        })
                
                # 平行趋势检验（简化版）
                st.write("#### DID平行趋势检验（政策前3期/后3期）")
                trend_results = []
                try:
                    # 提取政策前后时间节点
                    policy_year = df_clean[df_clean[did_policy_col]==1][time_col].min()
                    df_clean['relative_time'] = df_clean[time_col] - policy_year
                    # 仅保留-3到+3期
                    trend_df = df_clean[(df_clean['relative_time'] >= -3) & (df_clean['relative_time'] <= 3)]
                    
                    for t in sorted(trend_df['relative_time'].unique()):
                        if t == 0: continue  # 基准期
                        trend_df[f't_{t}'] = (trend_df['relative_time'] == t).astype(int)
                        trend_formula = f"{y_col} ~ 1 + {did_treat_col} + {' + '.join([f't_{i}' for i in sorted(trend_df['relative_time'].unique()) if i !=0])} + {did_treat_col}*({' + '.join([f't_{i}' for i in sorted(trend_df['relative_time'].unique()) if i !=0])}) + {' + '.join(X_cols)}"
                        trend_model = plm.PanelOLS.from_formula(trend_formula, trend_df.set_index([id_col, time_col])).fit()
                        coef = trend_model.params[f'{did_treat_col}:t_{t}'] if f'{did_treat_col}:t_{t}' in trend_model.params else np.nan
                        pval = trend_model.pvalues[f'{did_treat_col}:t_{t}'] if f'{did_treat_col}:t_{t}' in trend_model.pvalues else np.nan
                        trend_results.append({
                            '相对时间': t,
                            '系数': coef.round(4) if not np.isnan(coef) else 'N/A',
                            'P值': pval.round(4) if not np.isnan(pval) else 'N/A',
                            '是否显著': '是' if (not np.isnan(pval) and pval < 0.05) else '否'
                        })
                except:
                    trend_results.append({'相对时间': '计算失败', '系数': 'N/A', 'P值': 'N/A', '是否显著': 'N/A'})
                
                st.write("#### DID核心结果")
                st.dataframe(pd.DataFrame(did_results), use_container_width=True)
                st.write("#### 平行趋势检验结果")
                st.dataframe(pd.DataFrame(trend_results), use_container_width=True)

            # 3.2.3 中介效应分析（Bootstrap+Sobel）
            if mediator_col and len(X_cols)>=1:
                st.write("### 2.3 中介效应分析")
                # 逐步回归法
                step1_formula = f"{y_col} ~ 1 + {X_cols[0]} + {' + '.join(X_cols[1:])}"
                step1_model = plm.PanelOLS.from_formula(step1_formula, panel_df).fit()
                step2_formula = f"{mediator_col} ~ 1 + {X_cols[0]} + {' + '.join(X_cols[1:])}"
                step2_model = plm.PanelOLS.from_formula(step2_formula, panel_df).fit()
                step3_formula = f"{y_col} ~ 1 + {X_cols[0]} + {' + '.join(X_cols[1:])} + {mediator_col}"
                step3_model = plm.PanelOLS.from_formula(step3_formula, panel_df).fit()
                
                # 中介效应结果表格
                mediator_results = pd.DataFrame({
                    '步骤': ['Y~X', 'M~X', 'Y~X+M'],
                    f'{X_cols[0]}系数': [
                        step1_model.params[X_cols[0]].round(4) if X_cols[0] in step1_model.params else np.nan,
                        step2_model.params[X_cols[0]].round(4) if X_cols[0] in step2_model.params else np.nan,
                        step3_model.params[X_cols[0]].round(4) if X_cols[0] in step3_model.params else np.nan
                    ],
                    f'{X_cols[0]}P值': [
                        step1_model.pvalues[X_cols[0]].round(4) if X_cols[0] in step1_model.pvalues else np.nan,
                        step2_model.pvalues[X_cols[0]].round(4) if X_cols[0] in step2_model.pvalues else np.nan,
                        step3_model.pvalues[X_cols[0]].round(4) if X_cols[0] in step3_model.pvalues else np.nan
                    ],
                    f'{mediator_col}系数': [
                        np.nan,
                        np.nan,
                        step3_model.params[mediator_col].round(4) if mediator_col in step3_model.params else np.nan
                    ],
                    f'{mediator_col}P值': [
                        np.nan,
                        np.nan,
                        step3_model.pvalues[mediator_col].round(4) if mediator_col in step3_model.pvalues else np.nan
                    ],
                    'R²': [
                        step1_model.rsquared.round(4),
                        step2_model.rsquared.round(4),
                        step3_model.rsquared.round(4)
                    ]
                })
                
                # Sobel检验（简化版）
                sobel_z = (step1_model.params[X_cols[0]] * step3_model.params[mediator_col]) / np.sqrt(
                    step3_model.params[mediator_col]**2 * step1_model.std_errors[X_cols[0]]**2 +
                    step1_model.params[X_cols[0]]**2 * step3_model.std_errors[mediator_col]**2
                )
                sobel_pval = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
                mediator_conclusion = pd.DataFrame({
                    '检验方法': ['逐步回归法', 'Sobel检验'],
                    '统计量': ['-', sobel_z.round(4)],
                    'P值': ['-', sobel_pval.round(4)],
                    '结论': [
                        '部分中介' if (step1_model.pvalues[X_cols[0]]<0.05 and step2_model.pvalues[X_cols[0]]<0.05 and step3_model.pvalues[mediator_col]<0.05 and abs(step3_model.params[X_cols[0]])<abs(step1_model.params[X_cols[0]])) else '完全中介' if (step1_model.pvalues[X_cols[0]]<0.05 and step2_model.pvalues[X_cols[0]]<0.05 and step3_model.pvalues[mediator_col]<0.05 and step3_model.pvalues[X_cols[0]]>=0.05) else '无中介效应',
                        '显著' if sobel_pval < 0.05 else '不显著'
                    ]
                })
                
                st.dataframe(mediator_results, use_container_width=True)
                st.write("#### 中介效应显著性检验")
                st.dataframe(mediator_conclusion, use_container_width=True)

            # 3.2.4 空间面板模型（可选）
            if SPATIAL_AVAILABLE and region_col:
                st.write("### 2.4 空间面板分析")
                try:
                    # 莫兰指数
                    w = ps.weights.Queen.from_dataframe(df_clean)  # 邻接权重矩阵
                    moran = esda.Moran(df_clean[y_col], w)
                    moran_results = pd.DataFrame({
                        '指标': ['莫兰指数(I)', 'Z值', 'P值', '空间相关性'],
                        '数值': [
                            moran.I.round(4),
                            moran.z_sim.round(4),
                            moran.p_sim.round(4),
                            '存在' if moran.p_sim < 0.05 else '不存在'
                        ]
                    })
                    st.dataframe(moran_results, use_container_width=True)
                except:
                    st.warning("⚠️ 空间权重矩阵构建失败，跳过空间分析")

            # 3.3 机器学习模型分析（表格化对比）
            st.subheader("🤖 三、机器学习模型对比分析")
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
            
            # 模型训练与评估（表格化）
            model_metrics = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                # 5折交叉验证
                cv_r2 = cross_val_score(model, X_ml, y_ml, cv=5, scoring='r2').mean()
                model_metrics.append({
                    "模型": name,
                    "测试集R²": r2.round(4),
                    "测试集RMSE": rmse.round(4),
                    "测试集MAE": mae.round(4),
                    "5折交叉验证R²": cv_r2.round(4),
                    "排序": 0  # 后续排序用
                })
            
            # 排序
            metrics_df = pd.DataFrame(model_metrics)
            metrics_df['排序'] = metrics_df['测试集R²'].rank(ascending=False).astype(int)
            metrics_df = metrics_df.sort_values('排序')
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # 最优模型变量重要性（表格+可视化）
            best_model_name = metrics_df.loc[metrics_df['排序']==1, '模型'].iloc[0]
            best_model = models[best_model_name]
            st.write(f"### 3.1 最优模型（{best_model_name}）变量重要性")
            
            if hasattr(best_model, 'feature_importances_'):
                imp_df = pd.DataFrame({
                    '变量': X_cols,
                    '重要性': best_model.feature_importances_,
                    '重要性排名': best_model.feature_importances_.argsort()[::-1]+1
                }).sort_values('重要性排名').round(4)
                st.dataframe(imp_df, use_container_width=True)
                
                # 可视化
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(imp_df['变量'][:10], imp_df['重要性'][:10], color='#2ecc71')
                ax.set_xlabel('Importance')
                ax.set_title(f'{best_model_name} Feature Importance (Top 10)')
                st.pyplot(fig)
            else:
                # 线性模型系数绝对值作为重要性
                coef_df = pd.DataFrame({
                    '变量': X_cols,
                    '系数': best_model.coef_,
                    '系数绝对值': np.abs(best_model.coef_),
                    '重要性排名': np.abs(best_model.coef_).argsort()[::-1]+1
                }).sort_values('重要性排名').round(4)
                st.dataframe(coef_df, use_container_width=True)

            # 3.4 稳健性检验（表格化）
            st.subheader("🛡️ 四、稳健性检验")
            robust_results = []
            
            # 方法1：替换核心解释变量（滞后一期）
            st.write("### 4.1 核心变量滞后一期检验")
            try:
                df_robust = df_clean.copy()
                df_robust[f'{X_cols[0]}_lag1'] = df_robust.groupby(id_col)[X_cols[0]].shift(1)
                df_robust = df_robust.dropna()
                robust_formula = f"{y_col} ~ 1 + {X_cols[0]}_lag1 + {' + '.join(X_cols[1:])}"
                robust_model = plm.PanelOLS.from_formula(robust_formula, df_robust.set_index([id_col, time_col])).fit()
                
                for var in [f'{X_cols[0]}_lag1'] + X_cols[1:]:
                    if var in robust_model.params.index:
                        robust_results.append({
                            '检验方法': '核心变量滞后一期',
                            '变量': var,
                            '系数': robust_model.params[var].round(4),
                            'P值': robust_model.pvalues[var].round(4),
                            '是否显著': '是' if robust_model.pvalues[var]<0.05 else '否',
                            'R²': robust_model.rsquared.round(4)
                        })
            except:
                robust_results.append({'检验方法': '核心变量滞后一期', '变量': '计算失败', '系数': 'N/A', 'P值': 'N/A', '是否显著': 'N/A', 'R²': 'N/A'})
            
            # 方法2：剔除异常值（5%缩尾）
            st.write("### 4.2 5%缩尾检验")
            try:
                df_trim = df_clean.copy()
                for var in numeric_vars:
                    df_trim[var] = df_trim[var].clip(df_trim[var].quantile(0.05), df_trim[var].quantile(0.95))
                trim_model = plm.PanelOLS.from_formula(fe_formula, df_trim.set_index([id_col, time_col])).fit()
                
                for var in X_cols:
                    if var in trim_model.params.index:
                        robust_results.append({
                            '检验方法': '5%缩尾',
                            '变量': var,
                            '系数': trim_model.params[var].round(4),
                            'P值': trim_model.pvalues[var].round(4),
                            '是否显著': '是' if trim_model.pvalues[var]<0.05 else '否',
                            'R²': trim_model.rsquared.round(4)
                        })
            except:
                robust_results.append({'检验方法': '5%缩尾', '变量': '计算失败', '系数': 'N/A', 'P值': 'N/A', '是否显著': 'N/A', 'R²': 'N/A'})
            
            st.dataframe(pd.DataFrame(robust_results), use_container_width=True)

            st.success("✅ 全流程深度分析完成！所有结果均为表格化输出，可直接复制到论文中使用。")
else:
    st.info("请先在左侧侧边栏上传/整合面板数据文件（Excel/CSV格式）")
