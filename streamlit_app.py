import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from io import BytesIO
import base64
import plotly.graph_objects as go

st.set_page_config(page_title='Telecom Churn Analytics', layout='wide')

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load dataset either from secrets URL or local CSV."""
    url = st.secrets.get('data_url', '')
    try:
        if url:
            df = pd.read_csv(url)
        else:
            df = pd.read_csv('telecom_survey_synthetic.csv')
    except Exception as e:
        st.error(f'Error loading data → {e}')
        df = pd.DataFrame()
    return df

def download_link(df, filename, label):
    """Generate a link allowing the data in a given panda dataframe to be downloaded
    in:  df   --  DataFrame
         filename -- filename and extension of file. e.g. mydata.csv
         label -- link text
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

# ------------------------------------------------------------------
# Load + sidebar filters
# ------------------------------------------------------------------
df = load_data()
if df.empty:
    st.stop()

st.sidebar.header('Global Filters')
age_sel = st.sidebar.multiselect('Age Group', sorted(df.age_group.unique()),
                                 default=sorted(df.age_group.unique()))
loyalty_sel = st.sidebar.multiselect('Loyalty Tier', sorted(df.loyalty_tier.unique()),
                                     default=sorted(df.loyalty_tier.unique()))
df_filt = df[df['age_group'].isin(age_sel) & df['loyalty_tier'].isin(loyalty_sel)]

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_viz, tab_clf, tab_cluster, tab_arm, tab_reg = st.tabs(
    ['📊 Data Visualization', '🤖 Classification', '🧩 Clustering',
     '🔗 Association Rules', '📈 Regression'])

# ------------------------------------------------------------------
# 1. Data Visualization
# ------------------------------------------------------------------
with tab_viz:
    st.subheader('Key Descriptive Insights')

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Survey Rows', len(df_filt))
        fig = px.histogram(df_filt, x='satisfaction_score', nbins=20,
                           title='Satisfaction Score Distribution')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df_filt, x='churn_intent', y='avg_monthly_spend',
                     color='churn_intent',
                     title='Monthly Spend vs. Churn Intent')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(df_filt, x='loyalty_tier',
                     y='avg_monthly_spend',
                     title='Average Spend by Loyalty Tier',
                     color='loyalty_tier')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(df_filt, x='engagement_count',
                         y='satisfaction_score', color='churn_intent',
                         title='Engagement vs. Satisfaction')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        corr = df_filt.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df_filt, x='network_quality_score',
                           color='churn_intent',
                           title='Network Quality vs.Churn Intent',
                           barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(df_filt, x='price_quality_weight',
                           title='Price vs. Quality Preference')
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ NEW: ADDITIONAL DESCRIPTIVE PLOTS (ALL PLOTLY) -----------------------
    st.markdown("### Additional Descriptive Churn Insights")
    # 1. Churn Rate by Tenure Group (Plotly)
    if 'churn_flag' not in df_filt.columns:
        df_filt['churn_flag'] = (df_filt['churn_intent'] >= 4).astype(int)
    df_filt['tenure_group'] = pd.cut(df_filt['tenure_months'], 
                                     bins=[0, 12, 24, 36, 48, 60, 72, 1000], 
                                     labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '73+'])
    churn_by_tenure = (
        df_filt.groupby('tenure_group')['churn_flag'].mean().reset_index()
        .rename(columns={'churn_flag': 'churn_rate'})
    )
    churn_by_tenure['churn_rate'] = churn_by_tenure['churn_rate'] * 100

    fig1 = px.bar(
        churn_by_tenure, x='tenure_group', y='churn_rate',
        labels={'tenure_group': 'Tenure Group (Months)', 'churn_rate': 'Churn Rate (%)'},
        title='Churn Rate by Tenure Group',
        color='churn_rate',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Churn rate is highest among early-tenure customers. Target retention programs accordingly.")

    # 2. Monthly Spend (Charges) vs Churn (Box Plot, Plotly)
    fig2 = px.box(
        df_filt, x='churn_flag', y='avg_monthly_spend',
        color='churn_flag',
        labels={'churn_flag': 'Churn Status', 'avg_monthly_spend': 'Avg Monthly Spend'},
        title='Monthly Spend by Churn Status',
        color_discrete_map={0: 'mediumseagreen', 1: 'salmon'}
    )
    fig2.update_xaxes(tickvals=[0, 1], ticktext=['Stayed', 'Churned'])
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Analyze if high or low spenders are more likely to churn. Enables better pricing interventions.")

    # 3. Churn Rate by Contract Type (Plotly)
    if 'contract_type' in df_filt.columns:
        churn_by_contract = (
            df_filt.groupby('contract_type')['churn_flag'].mean().reset_index()
            .rename(columns={'churn_flag': 'churn_rate'})
        )
        churn_by_contract['churn_rate'] = churn_by_contract['churn_rate'] * 100

        fig3 = px.bar(
            churn_by_contract,
            x='contract_type', y='churn_rate',
            labels={'contract_type': 'Contract Type', 'churn_rate': 'Churn Rate (%)'},
            title='Churn Rate by Contract Type',
            color='churn_rate',
            color_continuous_scale='Teal'
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Month-to-month contracts show the highest churn. Lock-in and loyalty offers can help.")

    st.markdown('---')
    st.markdown('**Raw filtered data**')
    st.dataframe(df_filt)

    st.markdown(download_link(df_filt, 'filtered_data.csv',
                              '⬇️ Download filtered data'), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. Classification
# ------------------------------------------------------------------
with tab_clf:
    st.subheader('Customer Churn Classification')

    df_model = df_filt.copy()
    df_model['churn_flag'] = (df_model['churn_intent'] >= 4).astype(int)

    y = df_model['churn_flag']
    X = df_model.drop(['churn_flag', 'churn_intent'], axis=1)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=0),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0)
    }

    metrics_store = {}
    roc_store = {}
    cm_store = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[('prep', preprocessor),
                              ('model', model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42,
                                                            stratify=y)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:,1]

        metrics_store[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        }
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_store[name] = (fpr, tpr)
        cm_store[name] = confusion_matrix(y_test, y_pred)

        # Save the best model (by F1) to session_state for prediction
        best_f1 = st.session_state.get('best_f1', 0)
        if metrics_store[name]['F1'] > best_f1:
            st.session_state['best_f1'] = metrics_store[name]['F1']
            st.session_state['best_model_name'] = name
            st.session_state['best_pipe'] = pipe

    st.write('### Model Performance (Test Set)')
    perf_df = pd.DataFrame(metrics_store).T.round(3)
    st.dataframe(perf_df)

    # Confusion matrix
    st.write('### Confusion Matrix')
    model_choice = st.selectbox('Select model', list(models.keys()))
    cm = cm_store[model_choice]
    cm_fig = px.imshow(cm, text_auto=True,
                       x=['Not Churn','Churn'], y=['Not Churn','Churn'],
                       title=f'Confusion Matrix – {model_choice}')
    st.plotly_chart(cm_fig, use_container_width=True)

    # ROC curves
    st.write('### ROC Curves')
    roc_fig = go.Figure()
    for name,(fpr,tpr) in roc_store.items():
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=name))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 name='Chance', line=dict(dash='dash')))
    roc_fig.update_layout(title='ROC – All Models', xaxis_title='FPR',
                          yaxis_title='TPR')
    st.plotly_chart(roc_fig, use_container_width=True)

    # Prediction on new data
    st.write('---')
    st.write('### Predict on New Unlabelled Data')
    uploaded = st.file_uploader('Upload a CSV (same schema but no `churn_intent` column)',
                                type=['csv'])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        if 'churn_intent' in new_df.columns:
            new_df = new_df.drop('churn_intent', axis=1)
        best_pipe = st.session_state.get('best_pipe')
        if best_pipe:
            preds = best_pipe.predict(new_df)
            new_df['predicted_churn_flag'] = preds
            st.dataframe(new_df.head())
            csv = new_df.to_csv(index=False).encode()
            st.download_button('Download predictions',
                               csv,
                               file_name='predictions.csv',
                               mime='text/csv')
        else:
            st.warning('No trained model in session.')

# ------------------------------------------------------------------
# 3. Clustering
# ------------------------------------------------------------------
with tab_cluster:
    st.subheader('Customer Segmentation (K‑means)')

    num_cols_all = df_filt.select_dtypes(include=['int64','float64']).columns.tolist()
    df_cluster = df_filt[num_cols_all].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Elbow chart
    sse = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=0, n_init='auto')
        km.fit(X_scaled)
        sse.append(km.inertia_)

    elbow_fig = px.line(x=list(K_range), y=sse, markers=True,
                        title='Elbow Method – Within‑cluster SSE')
    elbow_fig.update_xaxes(title='k')
    elbow_fig.update_yaxes(title='SSE')
    st.plotly_chart(elbow_fig, use_container_width=True)

    k = st.slider('Select number of clusters', 2, 10, 4)
    km_final = KMeans(n_clusters=k, random_state=0, n_init='auto')
    labels = km_final.fit_predict(X_scaled)
    df_filt['cluster'] = labels

    # Persona table
    persona = df_filt.groupby('cluster').agg({
        'satisfaction_score':'mean',
        'avg_monthly_spend':'mean',
        'engagement_count':'mean',
        'services_count':'mean',
        'value_for_money':'mean'
    }).round(1)
    st.write('### Cluster Personas (means)')
    st.dataframe(persona)

    # Download labelled data
    st.markdown(download_link(df_filt, 'clustered_data.csv',
                              '⬇️ Download data with clusters'), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 4. Association Rule Mining
# ------------------------------------------------------------------
with tab_arm:
    st.subheader('Market‑Basket Insights (Apriori)')

    col_a = st.selectbox('Column A', ['retention_incentives','pain_points'])
    col_b = st.selectbox('Column B', ['pain_points','retention_incentives'])
    min_sup = st.slider('Minimum support', 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider('Minimum confidence', 0.1, 0.9, 0.3, 0.05)

    # Prepare one‑hot encoded basket
    def split_items(series):
        return series.fillna('').apply(lambda x: [i.strip() for i in x.split(',') if i.strip()])

    basket = split_items(df_filt[col_a]) + split_items(df_filt[col_b])
    # explode to single row/item pairs
    exploded = df_filt[[col_a, col_b]].copy()
    exploded['basket'] = split_items(df_filt[col_a]) + split_items(df_filt[col_b])
    # Build a list of lists for each row
    transactions = split_items(df_filt[col_a] + ', ' + df_filt[col_b])
    # Create one‑hot df
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions.tolist()).transform(transactions.tolist())
    df_hot = pd.DataFrame(te_ary, columns=te.columns_)

    freq_sets = apriori(df_hot, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_sets, metric='confidence', min_threshold=min_conf)
    rules = rules.sort_values('confidence', ascending=False).head(10)
    st.write('### Top 10 Rules')
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])

# ------------------------------------------------------------------
# 5. Regression
# ------------------------------------------------------------------
with tab_reg:
    st.subheader('Predicting Customer Lifetime Value (Spend)')

    target = 'avg_monthly_spend'
    y_reg = df_filt[target]
    X_reg = df_filt.drop([target,'churn_intent'], axis=1)

    cat_cols_r = X_reg.select_dtypes(include=['object']).columns.tolist()
    num_cols_r = X_reg.select_dtypes(include=['int64','float64']).columns.tolist()

    preprocess_r = ColumnTransformer([
        ('num', StandardScaler(), num_cols_r),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols_r)
    ])

    reg_models = {
        'Linear' : LinearRegression(),
        'Ridge'  : Ridge(alpha=1.0),
        'Lasso'  : Lasso(alpha=0.1),
        'Decision Tree' : DecisionTreeRegressor(max_depth=6, random_state=0)
    }

    results = {}
    for name, mdl in reg_models.items():
        pipe = Pipeline([('prep', preprocess_r), ('mdl', mdl)])
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = np.mean(np.abs(pred - y_test))
        rmse = np.sqrt(np.mean((pred - y_test)**2))
        r2 = pipe.score(X_test, y_test)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    st.write('### Model Comparison')
    st.dataframe(pd.DataFrame(results).T.round(2))

    # Feature importance for tree
    dt_pipe = Pipeline([('prep', preprocess_r),
                        ('mdl', reg_models['Decision Tree'])])
    dt_pipe.fit(X_reg, y_reg)
    tree = dt_pipe.named_steps['mdl']
    enc = dt_pipe.named_steps['prep']
    onehot_cols = enc.get_feature_names_out()
    importances = pd.Series(tree.feature_importances_, index=onehot_cols)
    top_feats = importances.sort_values(ascending=False).head(15)
    fig_imp = px.bar(top_feats[::-1],
                     title='Decision Tree – Top Feature Importances')
    st.plotly_chart(fig_imp, use_container_width=True)
