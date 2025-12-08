import pandas as pd
import numpy as np
import os
import time
import joblib
import gc
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    root_mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import RandomizedSearchCV
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF_Regressor
    from cuml.ensemble import RandomForestClassifier as cuRF_Classifier
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# 1. Column Configuration

REGR_TARGET = 'DEP_ADDED_DELAY'

CLASS_TARGET = 'DEP_DEL15'

LEAKAGE_COLS = [
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]

# HIGH-CARDINALITY CATEGORICAL FEATURES, mean encoded
TARGET_ENC_COLS = [
    'DEPARTING_AIRPORT',
    'PREVIOUS_AIRPORT',
    'DESTINATION_AIRPORT',
    'ROUTE_NAME',
    'INCOMING_ROUTE',
    'CARRIER_AIRPORT'
]

CYCLICAL_COLS_MAP = {
    'MONTH': 12,
    'DAY_OF_WEEK': 7
}

TIME_BLOCK_COL = ['DEP_TIME_BLK']

ONE_HOT_COLS = ['CARRIER_NAME', 'DISTANCE_GROUP', 'SEGMENT_NUMBER']

BINARY_COLS = [
    'AWND_missing', 'TMIN_missing', 'TMAX_missing',
    'WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06',
    'WT07', 'WT08', 'WT09', 'WT10', 'WT11'
]

NUM_COLS = [
    'CONCURRENT_FLIGHTS', 'NUMBER_OF_SEATS',
    'AIRPORT_FLIGHTS_MONTH', 'AIRLINE_FLIGHTS_MONTH',
    'AIRLINE_AIRPORT_FLIGHTS_MONTH', 'AVG_MONTHLY_PASS_AIRPORT',
    'AVG_MONTHLY_PASS_AIRLINE', 'FLT_ATTENDANTS_PER_PASS',
    'GROUND_SERV_PER_PASS', 'PLANE_AGE',
    'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'AWND'
]

# 2. Custom Transformers

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols_existing = set(X_copy.columns)
        to_drop = [c for c in self.columns_to_drop if c in cols_existing]
        return X_copy.drop(columns=to_drop)


class CyclicalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col_max_mapping):
        self.col_max_mapping = col_max_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_new = pd.DataFrame(X, columns=list(self.col_max_mapping.keys()))
        else:
            X_new = X.copy()
        for col, max_val in self.col_max_mapping.items():
            if col in X_new.columns:
                X_new[f'{col}_sin'] = np.sin(2 * np.pi * X_new[col] / max_val)
                X_new[f'{col}_cos'] = np.cos(2 * np.pi * X_new[col] / max_val)
                X_new.drop(columns=[col], inplace=True)
        return X_new


class TimeBlockCyclicalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            col_data = X.iloc[:, 0].astype(str)
        else:
            col_data = pd.Series(X[:, 0]).astype(str)
        hours = col_data.str[:2].astype(int)
        is_wide_block = col_data == '0001-0559'
        hours = np.where(is_wide_block, 3, hours)
        sin_trans = np.sin(2 * np.pi * hours / 24.0)
        cos_trans = np.cos(2 * np.pi * hours / 24.0)
        return np.column_stack([sin_trans, cos_trans])


class BoolToNumTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(int)


class CastToFloatTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(np.float32)

# 3. Model Factory

def get_estimator(model_name='random_forest', model_type='regression', hyperparams={}):
    """
    Returns an instantiated model object based on the name.
    Supports:
      - 'rapids_rf'           (GPU random forest via cuML)
      - 'random_forest'       (sklearn RandomForest)
      - 'logistic_regression' (sklearn LogisticRegression, classification)
      - 'linear_regression'   (sklearn LinearRegression, regression)
      - 'ridge_regression'    (sklearn Ridge, regression)
    """
    model_name = model_name.lower()

    # 1. RAPIDS cuML Random Forest (GPU)
    if model_name in ('rapids_rf', 'rapid_rf'):
        if not CUML_AVAILABLE:
            raise ImportError("RAPIDS cuML is not installed.")

        print(">> Using RAPIDS cuML Random Forest (GPU Accelerated)")

        default_params = {
            'n_estimators': 100,
            'max_depth': 16,
            'n_streams': 1,
            'random_state': 42
        }
        default_params.update(hyperparams)

        if model_type == 'regression':
            return cuRF_Regressor(**default_params)
        elif model_type == 'classification':
            return cuRF_Classifier(**default_params)
        else:
            raise ValueError(f"Unsupported model_type for RAPIDS RF: {model_type}")

    # 2. Standard Sklearn Random Forest (CPU)
    elif model_name == 'random_forest':
        print(">> Using Standard Sklearn Random Forest (CPU)")
        default_params = {
            'n_estimators': 50,
            'max_depth': 20,
            'n_jobs': -1,
            'verbose': 2
        }
        default_params.update(hyperparams)

        if model_type == 'regression':
            return RandomForestRegressor(**default_params)
        elif model_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**default_params)
        else:
            raise ValueError(f"Unsupported model_type for RandomForest: {model_type}")

    # 3. Logistic Regression (classification)
    elif model_name == 'logistic_regression':
        if model_type != 'classification':
            raise ValueError("logistic_regression is only valid for classification tasks.")

        print(">> Using Sklearn Logistic Regression (CPU)")
        default_params = {
            'max_iter': 1000,
            'n_jobs': -1,
            'solver': 'lbfgs'
        }
        default_params.update(hyperparams)
        return LogisticRegression(**default_params)

    # 4. Linear Regression (OLS, regression)
    elif model_name == 'linear_regression':
        if model_type != 'regression':
            raise ValueError("linear_regression is only valid for regression tasks.")

        print(">> Using Sklearn Linear Regression (CPU)")
        default_params = {}
        default_params.update(hyperparams)
        return LinearRegression(**default_params)

    # 5. Ridge Regression (L2, regression)
    elif model_name in ('ridge_regression', 'ridge'):
        if model_type != 'regression':
            raise ValueError("ridge_regression is only valid for regression tasks.")

        print(">> Using Sklearn Ridge Regression (CPU)")
        default_params = {}
        default_params.update(hyperparams)
        return Ridge(**default_params)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# 4. The Preprocessing Pipeline

def get_preprocessing_pipeline():
    # Mean/target encoding for high cardinality
    target_enc_transformer = TargetEncoder(cols=TARGET_ENC_COLS, smoothing=10)

    cyclical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('cyclical', CyclicalTransformer(col_max_mapping=CYCLICAL_COLS_MAP))
    ])

    time_blk_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('parser_cyclical', TimeBlockCyclicalTransformer())
    ])

    onehot_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    num_transformer = Pipeline(steps=[
        ('float_caster', CastToFloatTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    binary_transformer = Pipeline(steps=[
        ('bool_caster', BoolToNumTransformer()),
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('target_enc', target_enc_transformer, TARGET_ENC_COLS),
            ('cyclical', cyclical_transformer, list(CYCLICAL_COLS_MAP.keys())),
            ('time_blk', time_blk_transformer, TIME_BLOCK_COL),
            ('onehot', onehot_transformer, ONE_HOT_COLS),
            ('num', num_transformer, NUM_COLS),
            ('binary', binary_transformer, BINARY_COLS)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('dropper', ColumnDropper(columns_to_drop=LEAKAGE_COLS)),
        ('preprocessor', preprocessor)
    ])

    return pipeline


class InferencePipeline:
    """
    A lightweight wrapper for inference:
    - stores the fitted preprocessor and model
    - exposes predict(X) that applies both.
    """
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        X_trans = X_trans.astype(np.float32)
        return self.model.predict(X_trans)

    def fit(self, X, y):
        raise NotImplementedError("This pipeline is for inference only.")


def get_feature_names_from_preprocessor(preprocessing):
    """
    Given the fitted preprocessing Pipeline (the one returned by
    get_preprocessing_pipeline()), return a list of feature names
    in the same order as the columns in the transformed matrix.
    """
    if isinstance(preprocessing, Pipeline):
        ct = preprocessing.named_steps['preprocessor']
    else:
        ct = preprocessing 

    feature_names = []

    for name, trans, cols in ct.transformers_:
        if name == 'target_enc':
            feature_names.extend(list(cols))

        elif name == 'cyclical':
            for col in cols:
                feature_names.append(f"{col}_sin")
                feature_names.append(f"{col}_cos")

        elif name == 'time_blk':
            for col in cols:
                feature_names.append(f"{col}_sin")
                feature_names.append(f"{col}_cos")

        elif name == 'onehot':
            if isinstance(trans, Pipeline):
                ohe = trans.named_steps['onehot']
            else:
                ohe = trans
            ohe_feature_names = ohe.get_feature_names_out(cols)
            feature_names.extend(ohe_feature_names.tolist())

        elif name == 'num':
            feature_names.extend(list(cols))

        elif name == 'binary':
            feature_names.extend(list(cols))

        else:
            continue

    return feature_names


# 5. Training Function (Sequential & Chunked)
def train_model(X_train_path, y_train_path, X_test_path, y_test_path,
                model=None, model_name='random_forest',
                description="Model Run", checkpoint_dir=None, model_type='regression'):
    """
    Trains a model using Sequential Loading and Chunked Prediction to minimize RAM usage.

    - For model_type='regression':
        - Train only on rows where DEP_ADDED_DELAY > 0 (severity part of hurdle).
        - Fit model on log(1 + DEP_ADDED_DELAY) and exponentiate back at prediction time.
        - Evaluate on ALL test points (including zeros).
    - For model_type='classification':
        * Train and evaluate on all rows using CLASS_TARGET.
    """
    if model is None:
        model = get_estimator(model_name, model_type)

    if model_type == 'regression':
        target_var = REGR_TARGET
    elif model_type == 'classification':
        target_var = CLASS_TARGET
    else:
        raise ValueError('incorrect model_type')

    print(f'model type is:  {model_type}')
    print(f'target variable is: {target_var}')

    paths = [X_train_path, y_train_path, X_test_path, y_test_path]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    print(f"\n--- Starting: {description} ---")

#PHASE 1: TRAINING
    print("  [Phase 1] Loading Training Data...")
    X_train = pd.read_csv(X_train_path)
    y_train_full = pd.read_csv(y_train_path)[target_var].squeeze()

    if model_type == 'regression':
        # Train only on positive added delays
        positive_mask = y_train_full > 0
        num_pos = positive_mask.sum()
        num_total = len(y_train_full)
        print(f"  [Regression] Using positive added delays only: {num_pos}/{num_total} rows (DEP_ADDED_DELAY > 0).")

        X_train = X_train.loc[positive_mask].reset_index(drop=True)
        y_train = y_train_full.loc[positive_mask].reset_index(drop=True)

        # log transform target for regression
        y_train_log = np.log1p(y_train)
    else:
        # Classification: use all rows
        y_train = y_train_full
        y_train_log = None  # unused

    preprocessing = get_preprocessing_pipeline()

    print(f"  Transforming Training Data ({len(X_train)} samples)...")
    start_time = time.time()

    # For regression, pass original y_train (not log) into preprocessing
    X_train_trans = preprocessing.fit_transform(X_train, y_train)
    X_train_trans = X_train_trans.astype(np.float32)

    print("  Freeing RAM (Deleting raw X_train)...")
    del X_train
    gc.collect()

    print(f"  Fitting Model ({model_name})...")
    if model_type == 'regression':
        model.fit(X_train_trans, y_train_log)
    else:
        model.fit(X_train_trans, y_train)

    elapsed_train = time.time() - start_time
    print(f"  Training complete in {elapsed_train:.1f} seconds.")

    print("  Freeing RAM (Deleting transformed X_train)...")
    del X_train_trans
    gc.collect()

#PHASE 2: INFERENCE (CHUNKED)
    print("  [Phase 2] Predicting on Test Data (Chunked)...")

    y_test_full = pd.read_csv(y_test_path)[target_var].squeeze()

    CHUNK_SIZE = 50000
    y_pred_list = []

    for chunk in pd.read_csv(X_test_path, chunksize=CHUNK_SIZE):
        chunk_trans = preprocessing.transform(chunk)
        chunk_trans = chunk_trans.astype(np.float32)

        if model_type == 'regression':
            chunk_pred_log = model.predict(chunk_trans)
            chunk_pred = np.maximum(np.expm1(chunk_pred_log), 0.0)
        else:
            chunk_pred = model.predict(chunk_trans)

        y_pred_list.append(chunk_pred)

        del chunk, chunk_trans
        gc.collect()

    y_pred = np.concatenate(y_pred_list)

    #PHASE 3: EVAL
    print("  Evaluating...")

    results = {
        'description': description,
        'training_time_sec': elapsed_train,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    inference_pipeline = InferencePipeline(preprocessing, model)
    results['pipeline'] = inference_pipeline

    if model_type == 'regression':
        print('computing regression metrics (ALL test points for added delay)')

        mae_all = mean_absolute_error(y_test_full, y_pred)
        rmse_all = root_mean_squared_error(y_test_full, y_pred)
        r2_all = r2_score(y_test_full, y_pred)

        print(f"  [Regression Results - ALL] MAE: {mae_all:.4f}, RMSE: {rmse_all:.4f}, R2: {r2_all:.4f}")

        results.update({
            'mae_all': mae_all,
            'rmse_all': rmse_all,
            'r2_all': r2_all,
            'y_test': np.array(y_test_full),
            'y_pred': np.array(y_pred),
            'test_index': np.arange(len(y_test_full))
        })

    elif model_type == 'classification':
        print('computing classification metrics')
        y_test_int = y_test_full.astype(int)
        y_pred_int = y_pred.astype(int)

        acc = accuracy_score(y_test_int, y_pred_int)
        f1 = f1_score(y_test_int, y_pred_int, zero_division=0)
        prec = precision_score(y_test_int, y_pred_int, zero_division=0)
        rec = recall_score(y_test_int, y_pred_int, zero_division=0)
        cm = confusion_matrix(y_test_int, y_pred_int)

        print(f"  [Classification Results] Acc: {acc:.4f}, F1: {f1:.4f}, Recall: {rec:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        results.update({
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm
        })

        # Logistic-regression-specific feature importance
        if isinstance(model, LogisticRegression):
            try:
                feature_names = get_feature_names_from_preprocessor(preprocessing)
                coef = model.coef_
                if coef.ndim == 2 and coef.shape[0] == 1:
                    coef = coef[0]
                coef = np.array(coef)

                if len(feature_names) == len(coef):
                    results['feature_names'] = feature_names
                    results['logreg_coefficients'] = coef
                    print(
                        "  Stored logistic regression feature importances in "
                        "results['feature_names'] and results['logreg_coefficients']."
                    )
                else:
                    print(
                        "  Warning: length mismatch between feature_names and coefficients "
                        f"({len(feature_names)} vs {len(coef)}). Not storing feature importance."
                    )
            except Exception as e:
                print(f"  Warning: failed to extract logistic regression feature importances: {e}")

    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        safe_desc = "".join([c if c.isalnum() else "_" for c in description])
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_desc}_{timestamp_str}.joblib"
        save_path = os.path.join(checkpoint_dir, filename)
        print(f"  Saving checkpoint to: {save_path}")
        try:
            joblib.dump(results, save_path)
            results['checkpoint_path'] = save_path
        except Exception as e:
            print(f"  Warning: Checkpoint failed. Error: {e}")

    return results


def hyperparameter_search(
    X_train_path,
    y_train_path,
    model_name,
    model_type='regression',
    param_distributions=None,
    n_iter=20,
    cv=3,
    sample_size=200000,
    scoring=None,
    random_state=42,
    n_jobs=-1,
    verbose=2
):
    """
    Run RandomizedSearchCV on a small training subset using the full
    preprocessing pipeline + model.

    Returns a dict with:
      - 'best_estimator'
      - 'best_score'
      - 'best_params_full'  (with 'model__' prefixes)
      - 'best_params_model' (stripped for passing into get_estimator)
    """
    # choose target
    if model_type == 'regression':
        target_var = REGR_TARGET
    elif model_type == 'classification':
        target_var = CLASS_TARGET
    else:
        raise ValueError("model_type must be 'regression' or 'classification'.")

    # default scoring
    if scoring is None:
        scoring = 'neg_mean_absolute_error' if model_type == 'regression' else 'f1'

    print(f"\n[Hyperparameter Search] model={model_name}, type={model_type}")
    X = pd.read_csv(X_train_path, nrows=sample_size)
    y = pd.read_csv(y_train_path, nrows=sample_size)[target_var].squeeze()

    # For regression, mimic training setup: restrict to y > 0
    if model_type == 'regression':
        mask = y > 0
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        print(f"  Using {len(y)} rows for tuning (DEP_ADDED_DELAY > 0 from first {sample_size}).")
    else:
        print(f"  Using {len(y)} rows for tuning (classification).")

    prep = get_preprocessing_pipeline()
    base_model = get_estimator(model_name=model_name, model_type=model_type)

    full_pipe = Pipeline(steps=[
        ('prep', prep),
        ('model', base_model)
    ])

    if param_distributions is None:
        if model_name in ('ridge_regression', 'ridge'):
            param_distributions = {
                'model__alpha': np.logspace(-3, 3, 7)
            }
        elif model_name == 'logistic_regression':
            param_distributions = {
                'model__C': np.logspace(-3, 3, 7),
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs']
            }
        elif model_name == 'random_forest':
            param_distributions = {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [10, 20, 30],
                'model__max_features': ['sqrt', 'log2']
            }
        else:
            raise ValueError("Please provide param_distributions for this model_name.")

    search = RandomizedSearchCV(
        estimator=full_pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    search.fit(X, y)

    best_params_full = search.best_params_
    best_params_model = {
        k.replace('model__', ''): v
        for k, v in best_params_full.items()
        if k.startswith('model__')
    }

    results = {
        'best_estimator': search.best_estimator_,
        'best_score': search.best_score_,
        'best_params_full': best_params_full,
        'best_params_model': best_params_model
    }

    print("\n[Hyperparameter Search] done.")
    print("  Best score:", search.best_score_)
    print("  Best params (model only):", best_params_model)

    return results