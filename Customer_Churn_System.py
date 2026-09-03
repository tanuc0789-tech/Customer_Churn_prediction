import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

RANDOM_STATE = 42
DATA_PATH ="C:/Users/Akansha Chauhan/Downloads/Customer_Churn_System/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv"



# ---------------------------------------------------------------------------
#  LOAD DATA
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
#  CLEAN + PREPROCESS
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is stored as text and has 11 blank values for brand-new
    # customers (tenure == 0). Convert to numeric, treat blanks as 0.
    df["TotalCharges"] = (
        df["TotalCharges"].astype(str).str.strip().replace("", "0").astype(float)
    )

    # Drop pure identifier column — not predictive.
    df = df.drop(columns=["customerID"])

    # Collapse redundant categories ("No internet/phone service" -> "No").
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    # Encode target: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # get_dummies can produce bool dtype columns — force everything numeric.
    df_encoded = df_encoded.astype(float)
    return df_encoded


# ---------------------------------------------------------------------------
#  SPLIT + SCALE
# ---------------------------------------------------------------------------
def split_and_scale(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test


# ---------------------------------------------------------------------------
#  HANDLE CLASS IMBALANCE (SMOTE)
# ---------------------------------------------------------------------------
def manual_smote(X: np.ndarray, y: np.ndarray, minority_class=1, k=5,
                  random_state=RANDOM_STATE):
    """Minimal SMOTE implementation (used only if imbalanced-learn
    is not installed)."""
    rng = np.random.RandomState(random_state)

    X_min = X[y == minority_class]
    X_maj = X[y != minority_class]
    n_needed = len(X_maj) - len(X_min)
    if n_needed <= 0:
        return X, y

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, indices = nn.kneighbors(X_min)

    synthetic = []
    for _ in range(n_needed):
        i = rng.randint(0, len(X_min))
        neighbor_idx = indices[i][rng.randint(1, k + 1)]  # skip self
        gap = rng.rand()
        new_point = X_min[i] + gap * (X_min[neighbor_idx] - X_min[i])
        synthetic.append(new_point)

    X_resampled = np.vstack([X, np.array(synthetic)])
    y_resampled = np.concatenate([y, np.full(len(synthetic), minority_class)])
    return X_resampled, y_resampled


def balance_classes(X_train: pd.DataFrame, y_train: pd.Series):
    columns = X_train.columns
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print("Balanced classes using imbalanced-learn's SMOTE.")
    except ImportError:
        X_res, y_res = manual_smote(X_train.values, y_train.values)
        X_res = pd.DataFrame(X_res, columns=columns)
        y_res = pd.Series(y_res, name="Churn")
        print("imbalanced-learn not found — used manual SMOTE fallback.")

    print("Before SMOTE:", y_train.value_counts().to_dict())
    print("After  SMOTE:", pd.Series(y_res).value_counts().to_dict())
    return X_res, y_res


# ---------------------------------------------------------------------------
#  TRAIN + EVALUATE MODELS
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
        print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
        print(f"ROC-AUC: {auc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

        results[name] = auc
        fitted_models[name] = model

    print("\n\n=== SUMMARY (ROC-AUC, higher is better) ===")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {auc:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]:.4f})")
    return fitted_models, results, best_name


# ---------------------------------------------------------------------------
#  FEATURE IMPORTANCE (Decision Tree)
# ---------------------------------------------------------------------------
def show_feature_importance(model, feature_names, top_n=10):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)
    print(f"\nTop {top_n} features (Decision Tree importance):")
    print(importances)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df_raw = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_encoded = encode_features(df_clean)

    X_train, X_test, y_train, y_test = split_and_scale(df_encoded)
    X_train_res, y_train_res = balance_classes(X_train, y_train)

    fitted_models, results, best_name = train_and_evaluate(
        X_train_res, y_train_res, X_test, y_test
    )

    show_feature_importance(fitted_models["Decision Tree"], X_train.columns)

    # Save the best model for later use
    import joblib
    joblib.dump(fitted_models[best_name], "churn_model.pkl")
    print(f"\nSaved best model ('{best_name}') to churn_model.pkl")


if __name__ == "__main__":
    main()
