"""
=============================================================================
 Customer Behavior Analytics - Model Training Pipeline
 Project : Customer Behavior Analytics Using Clustering and Predictive
           Modelling
 Purpose : Cleans the raw e-commerce customer dataset, builds a K-Means
           customer segmentation model, trains a Random Forest churn
           prediction model, auto-labels each cluster with a business
           persona, and exports everything the Streamlit app needs:

             1. customer_churn_model.pkl   -> all fitted objects + metadata
             2. processed_customer_data.csv -> enriched dataset used by the
                dashboard for instant, no-recompute visualizations

 Run:  python train_model.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd

# Force classic numpy object dtype for strings instead of pandas' newer
# StringDtype/ArrowStringArray. This keeps the saved pickle/CSV readable
# across different pandas versions (the newer string dtype's internal
# array format is not always unpicklable on other pandas installs, which
# raises "NotImplementedError: StringDtype(...)" when the app loads it).
try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
N_CLUSTERS = 4
DATA_PATH = "ecommerce_customer_churn_dataset.csv"
MODEL_PATH = "customer_churn_model.pkl"
PROCESSED_DATA_PATH = "processed_customer_data.csv"

print("=" * 70)
print("CUSTOMER BEHAVIOR ANALYTICS - MODEL TRAINING PIPELINE")
print("=" * 70)

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
print("\n[1/9] Loading raw dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

raw_df = df.copy()  # keep original human-readable copy for the dashboard

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove("Churned")
categorical_cols = df.select_dtypes(include="object").columns.tolist()
print(f"      Numerical features : {len(numerical_cols)}")
print(f"      Categorical features: {categorical_cols}")

# -----------------------------------------------------------------------
# 2. MISSING VALUE IMPUTATION
# -----------------------------------------------------------------------
print("\n[2/9] Imputing missing values ...")
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
print(f"      Remaining nulls: {int(df.isnull().sum().sum())}")

# keep a human-readable, imputed (but not encoded) version for the dashboard
display_df = df.copy()

# -----------------------------------------------------------------------
# 3. CATEGORICAL ENCODING
# -----------------------------------------------------------------------
print("\n[3/9] Encoding categorical features ...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------------------------------------------------
# 4. CUSTOMER SEGMENTATION (K-MEANS)
# -----------------------------------------------------------------------
print("\n[4/9] Scaling features for clustering ...")
cluster_feature_cols = [c for c in df.columns if c != "Churned"]
scaler_cluster = StandardScaler()
scaled_features = scaler_cluster.fit_transform(df[cluster_feature_cols])

print(f"[5/9] Running K-Means (k={N_CLUSTERS}) ...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)
sil_score = silhouette_score(scaled_features, df["Cluster"])
print(f"      Silhouette score: {sil_score:.4f}")

# 2D PCA projection purely for visualization purposes
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(scaled_features)
df["PCA1"], df["PCA2"] = pca_coords[:, 0], pca_coords[:, 1]

# -----------------------------------------------------------------------
# 5. AUTO-LABEL CLUSTER PERSONAS FROM BUSINESS METRICS
# -----------------------------------------------------------------------
print("\n[6/9] Building cluster business personas ...")
profile_cols = [
    "Lifetime_Value",
    "Total_Purchases",
    "Average_Order_Value",
    "Login_Frequency",
    "Days_Since_Last_Purchase",
    "Cart_Abandonment_Rate",
]
cluster_profile = display_df.assign(Cluster=df["Cluster"]).groupby("Cluster").agg(
    {**{c: "mean" for c in profile_cols}, "Churned": "mean"}
)
cluster_profile["Size"] = df["Cluster"].value_counts().sort_index()
cluster_profile["Churn_Rate_%"] = (cluster_profile["Churned"] * 100).round(2)
cluster_profile = cluster_profile.drop(columns=["Churned"])

# Composite "value score" to rank clusters from most to least valuable
value_score = (
    cluster_profile["Lifetime_Value"].rank()
    + cluster_profile["Total_Purchases"].rank()
    + cluster_profile["Login_Frequency"].rank()
    - cluster_profile["Days_Since_Last_Purchase"].rank()
    - cluster_profile["Churn_Rate_%"].rank()
)
ranked_clusters = value_score.sort_values(ascending=False).index.tolist()

persona_names = [
    "Champions",
    "Loyal Customers",
    "At-Risk Customers",
    "Dormant / Low-Engagement",
]
persona_descriptions = {
    "Champions": "High spend, frequent logins, recent activity, low churn risk. Your best customers.",
    "Loyal Customers": "Consistent purchasers with healthy engagement and moderate value.",
    "At-Risk Customers": "Declining engagement and rising churn signals - needs proactive retention.",
    "Dormant / Low-Engagement": "Infrequent activity, low spend, highest churn probability.",
}
cluster_to_persona = {
    cluster_id: persona_names[i] for i, cluster_id in enumerate(ranked_clusters)
}
cluster_profile["Persona"] = cluster_profile.index.map(cluster_to_persona)
cluster_profile["Description"] = cluster_profile["Persona"].map(persona_descriptions)
cluster_profile = cluster_profile.reset_index()
print(cluster_profile[["Cluster", "Persona", "Size", "Churn_Rate_%"]].to_string(index=False))

df["Persona"] = df["Cluster"].map(cluster_to_persona)
display_df["Cluster"] = df["Cluster"]
display_df["Persona"] = df["Persona"]
display_df["PCA1"] = df["PCA1"]
display_df["PCA2"] = df["PCA2"]

# -----------------------------------------------------------------------
# 6. CHURN PREDICTION MODEL (RANDOM FOREST)
# -----------------------------------------------------------------------
print("\n[7/9] Training Random Forest churn classifier ...")
feature_cols_rf = [c for c in df.columns if c not in ["Churned", "Persona", "PCA1", "PCA2"]]
X = df[feature_cols_rf]
y = df["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler_rf = StandardScaler()
X_train_scaled = scaler_rf.fit_transform(X_train)
X_test_scaled = scaler_rf.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)
y_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# -----------------------------------------------------------------------
# 7. EVALUATION METRICS
# -----------------------------------------------------------------------
print("\n[8/9] Evaluating model performance ...")
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
for k, v in metrics.items():
    print(f"      {k:>10}: {v:.4f}")

report_dict = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)

feature_importance = (
    pd.DataFrame({"Feature": feature_cols_rf, "Importance": rf_model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

# apply the trained churn model back over the FULL dataset for the dashboard
full_scaled = scaler_rf.transform(df[feature_cols_rf])
display_df["Churn_Probability"] = rf_model.predict_proba(full_scaled)[:, 1]
display_df["Predicted_Churn"] = rf_model.predict(full_scaled)

# -----------------------------------------------------------------------
# 8. SAVE ARTIFACTS
# -----------------------------------------------------------------------
print("\n[9/9] Saving model bundle and processed dataset ...")

model_bundle = {
    "rf_model": rf_model,
    "kmeans_model": kmeans,
    "pca_model": pca,
    "scaler_cluster": scaler_cluster,
    "scaler_rf": scaler_rf,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "label_encoders": label_encoders,
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols,
    "cluster_feature_cols": cluster_feature_cols,
    "feature_cols_rf": feature_cols_rf,
    "cluster_to_persona": cluster_to_persona,
    "persona_descriptions": persona_descriptions,
    "cluster_profile": cluster_profile,
    "feature_importance": feature_importance,
    "metrics": metrics,
    "classification_report": report_dict,
    "confusion_matrix": cm,
    "roc_curve": {"fpr": fpr, "tpr": tpr},
    "silhouette_score": sil_score,
    "n_clusters": N_CLUSTERS,
    "random_state": RANDOM_STATE,
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_bundle, f)
print(f"      Saved -> {MODEL_PATH}")

display_df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"      Saved -> {PROCESSED_DATA_PATH}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
