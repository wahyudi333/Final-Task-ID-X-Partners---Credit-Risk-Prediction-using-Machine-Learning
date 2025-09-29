#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('dataset/loan_data_2007_2014.csv')


# In[3]:


import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


print(df.head())


# In[5]:


print(df.shape)


# In[6]:


df.info()


# In[7]:


df.describe(include="all").transpose()


# In[8]:


df.describe()


# In[9]:


# pengecekan jumlah kolom kosong

df.isna().sum()


# In[10]:


# perhitungan persentase kolom kosong terhadap total data pada masing-masing kolom. 

for i in df.columns:
    null_rate = df[i].isna().sum() / len(df) * 100
    if null_rate > 10 :
        print("{} null rate: {}%".format(i,round(null_rate,2)))


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

# pilih beberapa kolom numerik penting
num_cols = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 
            'int_rate', 'installment', 'annual_inc', 'dti']

plt.figure(figsize=(15,10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(8,5))
sns.countplot(x='loan_status', data=df)
plt.title("Distribusi Loan Status")
plt.show()


# In[13]:


plt.figure(figsize=(8,5))
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.title("Loan Amount vs Loan Status")
plt.show()


# In[14]:


plt.figure(figsize=(8,5))
sns.boxplot(x='loan_status', y='int_rate', data=df)
plt.title("Interest Rate vs Loan Status")
plt.show()


# In[15]:


plt.figure(figsize=(8,5))
sns.boxplot(x='loan_status', y='dti', data=df)
plt.title("DTI vs Loan Status")
plt.show()


# In[16]:


plt.figure(figsize=(8,5))
sns.boxplot(x='loan_status', y='annual_inc', data=df)
plt.ylim(0, 200000) # biar ga ketutup outlier ekstrim
plt.title("Annual Income vs Loan Status")
plt.show()


# In[17]:


plt.figure(figsize=(12,8))
sns.heatmap(df[['loan_amnt','funded_amnt','funded_amnt_inv','int_rate','installment','annual_inc','dti']].corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi antar fitur numerik")
plt.show()


# In[19]:


# Hapus kolom dengan null > 70%
threshold = 0.7
df = df[df.columns[df.isnull().mean() < threshold]]

# Tentukan kolom numerik dan kategorikal setelah penghapusan
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Imputasi numerik dengan median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Imputasi kategorikal dengan modus
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])


# In[20]:


# log transform annual_inc biar lebih normal
df['annual_inc_log'] = np.log1p(df['annual_inc'])

# hapus outlier annual_inc (di atas 99th percentile)
upper_limit = df['annual_inc'].quantile(0.99)
df = df[df['annual_inc'] <= upper_limit]


# In[21]:


from sklearn.preprocessing import LabelEncoder

# target encoding
df['loan_status'] = df['loan_status'].replace({
    'Charged Off': 1,
    'Default': 1,
    'Fully Paid': 0,
    'Current': 0
})

# label encoding grade
le = LabelEncoder()
df['grade'] = le.fit_transform(df['grade'])

# one-hot encoding untuk home_ownership
df = pd.get_dummies(df, columns=['home_ownership', 'verification_status'])


# In[22]:


plt.figure(figsize=(8,5))
sns.countplot(x='loan_status', data=df)
plt.title("Distribusi Loan Status")
plt.xticks(rotation=45)
plt.show()

print(df['loan_status'].value_counts(normalize=True)*100)


# In[23]:


# definisi kategori good & bad
good_status = ['Fully Paid', '0']
bad_status = ['Charged Off', '1', 'Late (31-120 days)', 
              'Late (16-30 days)', 'In Grace Period',
              'Does not meet the credit policy. Status:Fully Paid',
              'Does not meet the credit policy. Status:Charged Off']

# mapping ke biner
df['loan_status_binary'] = np.where(df['loan_status'].isin(bad_status), 1, 0)

print(df['loan_status_binary'].value_counts(normalize=True)*100)


# In[24]:


X = df.drop(columns=['loan_status', 'loan_status_binary'])
y = df['loan_status_binary']


# In[25]:


X.select_dtypes(include=['object']).columns


# In[26]:


# Drop kolom yang tidak relevan
drop_cols = ['emp_title', 'url', 'title', 'zip_code']
X = X.drop(columns=drop_cols)

# Term: ubah ke integer
X['term'] = X['term'].str.replace(' months', '').astype(int)

# Sub_grade: label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['sub_grade'] = le.fit_transform(X['sub_grade'])

# Emp_length: map ke angka
X['emp_length'] = X['emp_length'].replace({
    '10+ years': 10,
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    'n/a': np.nan
})

# Tanggal: ubah ke datetime & ambil tahun/bulan
date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
for col in date_cols:
    X[col] = pd.to_datetime(X[col], format='%b-%Y', errors='coerce')
    X[col + '_year'] = X[col].dt.year
    X[col + '_month'] = X[col].dt.month
X = X.drop(columns=date_cols)  # drop versi asli

# Pymnt_plan: binary
X['pymnt_plan'] = X['pymnt_plan'].map({'n':0, 'y':1})

# Initial_list_status: label encoding
X['initial_list_status'] = le.fit_transform(X['initial_list_status'])

# Application_type: binary
X['application_type'] = X['application_type'].map({'Individual':0, 'Joint App':1})

# Purpose & addr_state: one-hot encoding
X = pd.get_dummies(X, columns=['purpose','addr_state'], drop_first=True)


# In[27]:


# Hitung proporsi
print(y.value_counts(normalize=True) * 100)

# Visualisasi
sns.countplot(x=y, palette="Set2")
plt.title("Distribusi Good (0) vs Bad (1)")
plt.xlabel("Loan Status Binary")
plt.ylabel("Jumlah")
plt.show()


# In[28]:


# Cari kolom yang seluruhnya NaN
full_nan_cols = X.columns[X.isna().sum() == len(X)]

print("Kolom full NaN:")
print(full_nan_cols.tolist())
print("Jumlah:", len(full_nan_cols))


# In[29]:


# Drop kolom full NaN
full_nan_cols = [
    'application_type', 
    'issue_d_year', 'issue_d_month',
    'earliest_cr_line_year', 'earliest_cr_line_month',
    'last_pymnt_d_year', 'last_pymnt_d_month',
    'next_pymnt_d_year', 'next_pymnt_d_month',
    'last_credit_pull_d_year', 'last_credit_pull_d_month'
]

X = X.drop(columns=full_nan_cols, errors='ignore')

# Imputasi ulang
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
bool_cols = X.select_dtypes(include=['bool']).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())

for col in bool_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

print("NaN total sesudah imputasi & drop:", X.isna().sum().sum())


# In[30]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# sebelum SMOTE
print("Distribusi sebelum SMOTE:", Counter(y))

# Terapkan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# sesudah SMOTE
print("Distribusi sesudah SMOTE:", Counter(y_resampled))


# In[32]:


# Hitung proporsi
plt.figure(figsize=(7, 5))
sns.countplot(x=y_resampled, palette="Set2") 
# Gunakan y_resampled pada sumbu x

plt.title("Distribusi Good (0) vs Bad (1) Setelah SMOTE")
plt.xlabel("Loan Status Binary")
plt.ylabel("Jumlah")
plt.show()


# In[33]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)


# In[34]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("=== Logistic Regression ===")
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
print("ROC-AUC:", roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:,1]))


# In[44]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

print("\n=== Random Forest ===")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]))


# In[38]:


get_ipython().system('pip install xgboost')


# In[45]:


from xgboost import XGBClassifier

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)

print("\n=== XGBoost ===")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:,1]))


# In[47]:


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

log_reg = LogisticRegression(max_iter=1000, solver='saga')
param_lr = {
    "C": np.logspace(-3, 3, 10),
    "penalty": ["l1", "l2", "elasticnet"],
    "l1_ratio": np.linspace(0, 1, 5)  # hanya berlaku kalau penalty = elasticnet
}

search_lr = HalvingRandomSearchCV(
    estimator=log_reg,
    param_distributions=param_lr,
    factor=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search_lr.fit(X_train_scaled, y_train)
best_lr = search_lr.best_estimator_
print("\n=== Logistic Regression ===")
y_pred_lr = best_lr.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, best_lr.predict_proba(X_test_scaled)[:, 1]))


# In[49]:


rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

search_rf = HalvingRandomSearchCV(
    estimator=rf,
    param_distributions=param_rf,
    factor=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search_rf.fit(X_train_scaled, y_train)
best_rf = search_rf.best_estimator_

print("\n=== Random Forest ===")
y_pred_rf = best_rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, best_rf.predict_proba(X_test_scaled)[:, 1]))


# In[50]:


xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
param_xgb = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

search_xgb = HalvingRandomSearchCV(
    estimator=xgb,
    param_distributions=param_xgb,
    factor=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search_xgb.fit(X_train_scaled, y_train)
best_xgb = search_xgb.best_estimator_

print("\n=== XGBoost ===")
y_pred_xgb = best_xgb.predict(X_test_scaled)
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, best_xgb.predict_proba(X_test_scaled)[:, 1]))


# In[51]:


models_tuned = {
    "Logistic Regression": best_lr,
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

results = []

for name, model in models_tuned.items():
    print(f"\n=== {name} ===")
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Classification Report
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", roc_auc)
    
    # Simpan hasil
    results.append({
        "Model": name,
        "Accuracy": (cm[0,0] + cm[1,1]) / cm.sum(),
        "Precision (Class 1)": cm[1,1] / (cm[0,1] + cm[1,1]),
        "Recall (Class 1)": cm[1,1] / (cm[1,0] + cm[1,1]),
        "ROC-AUC": roc_auc
    })

# Tabel hasil akhir
df_results = pd.DataFrame(results)
print("\n=== Hasil Perbandingan Model ===")
print(df_results)


# In[52]:


# Simpan semua confusion matrix ke dictionary
cms = {
    "Logistic Regression": confusion_matrix(y_test, best_lr.predict(X_test_scaled)),
    "Random Forest": confusion_matrix(y_test, best_rf.predict(X_test_scaled)),
    "XGBoost": confusion_matrix(y_test, best_xgb.predict(X_test_scaled))
}

# Plot setiap confusion matrix
for name, cm in cms.items():
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[53]:


# Atur style
sns.set(style="whitegrid")

# Ubah dataframe ke format long untuk seaborn
df_melted = df_results.melt(id_vars="Model", 
                            value_vars=["Accuracy", "Precision (Class 1)", "Recall (Class 1)", "ROC-AUC"],
                            var_name="Metric", value_name="Score")

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted)
plt.ylim(0.95, 1.0)  # Karena semua skor tinggi, biar lebih terlihat perbedaan
plt.title("Perbandingan Performa Model")
plt.ylabel("Score")
plt.show()


# In[ ]:




