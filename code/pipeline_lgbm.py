import pandas as pd
import numpy as np
import os
import time
import joblib
import gc
import json
import lightgbm as lgb
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score,
    mean_tweedie_deviance, average_precision_score
)

# ==========================================
# Column Configuration
# ==========================================

REGR_TARGET = 'DEP_DELAY_NEW'
CLASS_TARGET = 'DEP_DEL15'

LEAKAGE_COLS = [
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY',
    
    # ## COLLINEAR FEATURES
    # 'DEP_HOUR', # COLLINEAR WITH DEP_TIME
    # 'DEP_TIME_BLK', # COLLINEAR WITH DEP_TIME
    # 'AVG_MONTHLY_PASS_AIRPORT',
    # 'AIRPORT_FLIGHTS_MONTH',
    # 'AVG_MONTHLY_PASS_AIRLINE',
    # 'AIRLINE_FLIGHTS_MONTH',
    # 'AIRLINE_AIRPORT_FLIGHTS_MONTH'
    # 'TMIN',
]

# Updated based on your engineered feature set
CATEGORICAL_COLS = [
    # Core IDs
    'CARRIER_NAME', 
    'DEPARTING_AIRPORT', 
    'PREVIOUS_AIRPORT',
    'DESTINATION_AIRPORT',
    
    # Engineered Routes
    'ROUTE_NAME',      # Origin -> Dest
    'INCOMING_ROUTE',  # Prev -> Origin
    'CARRIER_AIRPORT', # Hub Effect
    
    # Time
    'DEP_TIME_BLK',
    'MONTH',
    'DAY_OF_WEEK',
    'SEASON',
    
    # Groups
    'DISTANCE_GROUP', 
    'SEGMENT_NUMBER',
    
    # Weather Flags (Treating binary flags as categories is safe/good for LGBM)
    'IS_HEAVY_RAIN', 'IS_SNOWY', 
    'IS_FREEZING', 'IS_EXTREME_HEAT',
    'AWND_missing', 'TMIN_missing', 'TMAX_missing',
    'WT01', 'WT02', 'WT03', 'WT04', 
    'WT05', 'WT06', 'WT07',
    'WT08', 'WT09', 'WT10', 'WT11'
]

# Add specific WT flags to categorical if you want explicit split handling, 
# otherwise they pass through as int
WT_FLAGS = [f'WT{str(i).zfill(2)}' for i in range(1, 12)]
CATEGORICAL_COLS.extend(WT_FLAGS)

# No longer used, but kept for reference if needed later
CYCLICAL_COLS_MAP = {
    'MONTH': 12,
    'DAY_OF_WEEK': 7
}

# ==========================================
# Helpers & Transformers
# ==========================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else 1
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols_existing = set(X_copy.columns)
        to_drop = [c for c in self.columns_to_drop if c in cols_existing]
        print('DROPPED FEATURES:')
        print(to_drop)
        return X_copy.drop(columns=to_drop)

class PandasCategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
    
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else 1
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cat_cols:
            if col in X_out.columns:
                # Convert to string first to handle mixed types/nans, then category
                X_out[col] = X_out[col].astype(str).astype('category')
        return X_out

# ==========================================
# Pipeline Builder
# ==========================================

def get_lgbm_pipeline():
    # Minimized Pipeline: Just Drops & Casts
    pipeline = Pipeline(steps=[
        ('dropper', ColumnDropper(columns_to_drop=LEAKAGE_COLS)),
        ('cat_converter', PandasCategoryConverter(cat_cols=CATEGORICAL_COLS))
    ])
    return pipeline

class InferencePipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        
    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        # Force GPU for inference
        return self.model.predict(X_trans, device='gpu') 
        
    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        # Force GPU for inference
        return self.model.predict_proba(X_trans, device='gpu')

# ==========================================
# Model Factory (Decoupled)
# ==========================================

def get_lightgbm_estimator(model_type='regression', quick_run=False, 
                           use_tweedie=False, tweedie_variance_power=1.5, 
                           use_gpu=False, **kwargs):
    """
    Factory function to create a configured LightGBM estimator.
    """
    # Some template/default parameters 
    params = {
        'n_estimators': 100 if quick_run else 3000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    # GPU Configuration
    if use_gpu:
        print("  >> Configured for GPU Training")
        params['device'] = 'gpu'
        # GPU implementation often requires smaller max_bin to fit in memory/limit
        # Default to 255 (standard for 8-bit histograms) unless user overrides
        if 'max_bin' not in kwargs:
            params['max_bin'] = 255
            print("  >> Setting max_bin=255 for GPU compatibility")
    # Override with any user kwargs
    params.update(kwargs)

    if model_type == 'regression':
        if use_tweedie:
            print(f"  >> Configured for Tweedie Objective (power={tweedie_variance_power})")
            params['objective'] = 'tweedie'
            params['tweedie_variance_power'] = tweedie_variance_power
            # Monitor multiple metrics during training
            params['metric'] = ['tweedie', 'rmse', 'mae', 'r2']
        else:
            print("  >> Configured for Standard Regression (RMSE)")
            params['objective'] = 'regression'
            params['metric'] = ['rmse', 'mae', 'mse', 'r2']
            
        return lgb.LGBMRegressor(**params)
            
    elif model_type == 'classification':
        print("  >> Configured for Binary Classification")
        params['objective'] = 'binary'
        params['metric'] = ['binary_logloss', 'auc', 'average_precision']
        return lgb.LGBMClassifier(**params)
        
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")

# ==========================================
# Training Function
# ==========================================

def train_lightgbm_model(model, X_train_path, y_train_path, X_test_path, y_test_path, 
                         model_type='regression', description="LGBM Run", 
                         checkpoint_dir=None, stopping_rounds=100):
    """
    Trains a pre-instantiated LightGBM model.
    """
    
    print(f"\n--- Starting LightGBM Training: {description} ---")
    
    if model_type == 'regression':
        target_var = REGR_TARGET
    elif model_type == 'classification':
        target_var = CLASS_TARGET
    else:
        raise ValueError("model_type must be 'regression' or 'classification'")

    # Load & Split
    print("  Loading Training Data...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[target_var].squeeze()

    print("  Creating Validation Split (15%)...")
    # NOTE: stratify only works for Classification. For Regression, we skip it.
    stratify_param = y_train if model_type == 'classification' else None
    print('Stratifying validation set')
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=stratify_param # , random_state=42
    )
    del X_train, y_train
    gc.collect()

    # Preprocess
    print("  Preprocessing...")
    prep_pipeline = get_lgbm_pipeline()
    
    # Fit the pipeline
    X_tr_trans = prep_pipeline.fit_transform(X_tr, y_tr)
    X_val_trans = prep_pipeline.transform(X_val)

    # Fit
    print(f"  Training {model_type} (Early Stopping enabled)...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=stopping_rounds),
        lgb.log_evaluation(period=100)
    ]
    
    start_time = time.time()
    
    # Pass eval_set for early stopping
    model.fit(
        X_tr_trans, y_tr,
        eval_set=[(X_val_trans, y_val)],
        eval_names=['valid'],
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    print(f"  Training finished in {training_time:.1f}s")

    del X_tr, X_val, X_tr_trans, X_val_trans, y_tr, y_val
    gc.collect()

    # Inference
    print("  Predicting on Test Data...")
    y_test_full = pd.read_csv(y_test_path)[target_var].squeeze()

    # To revert to memory-safe chunking, uncomment the CHUNK loop below and comment out this block.
    
    X_test_full = pd.read_csv(X_test_path)
    X_test_trans = prep_pipeline.transform(X_test_full)
    y_pred = model.predict(X_test_trans)
    
    # --- MEMORY SAFE VERSION (COMMENTED OUT) ---
    # y_pred_list = []
    # CHUNK_SIZE = 50000
    # for chunk in pd.read_csv(X_test_path, chunksize=CHUNK_SIZE):
    #     chunk_trans = prep_pipeline.transform(chunk)
    #     y_pred_list.append(model.predict(chunk_trans))
    # y_pred = np.concatenate(y_pred_list)
    # ------------------------------------------

    # Evaluation
    print("  Evaluating...")
    results = {
        'description': description,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_time': training_time,
        'model_type': model_type
    }
    
    results['parameters'] = model.get_params()

    if model_type == 'regression':
        mae = mean_absolute_error(y_test_full, y_pred)
        rmse = root_mean_squared_error(y_test_full, y_pred)
        r2 = r2_score(y_test_full, y_pred)
        
        tweedie_score = None
        params = model.get_params()
        if params.get('objective') == 'tweedie':
             p = params.get('tweedie_variance_power', 1.5)
             y_pred_safe = np.maximum(y_pred, 1e-9)
             try:
                tweedie_score = mean_tweedie_deviance(y_test_full, y_pred_safe, power=p)
             except Exception:
                pass

        print(f"  [Regression] RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}", end="")
        if tweedie_score is not None:
            print(f", Tweedie: {tweedie_score:.4f}")
        print()

        results.update({'mae': mae, 'rmse': rmse, 'r2': r2, 'tweedie_deviance': tweedie_score})
        
    else:
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_test_int = y_test_full.astype(int)
        
        acc = accuracy_score(y_test_int, y_pred_binary)
        ## AUC is NOT CACLULATED CORRECTLY
        # this was consequence of 
        # trying to make this pipeline handle both the regressor and classifier
        # All classifiers were unfortunately created with this script; the LightGBM regressors were handled with LGBMRegr_pipeline
        auc = roc_auc_score(y_test_int, y_pred)
        f1 = f1_score(y_test_int, y_pred_binary)
        cm = confusion_matrix(y_test_int, y_pred_binary)
        precision = precision_score(y_test_int, y_pred_binary)
        recall = recall_score(y_test_int, y_pred_binary)
        average_precision = average_precision_score(y_test_int, y_pred_binary)
        print(f"  [Classification] AUC INCORRECT (recompute with predict_proba instead): {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}") # Added print for easy viewing
        results.update({'accuracy': acc, 'auc': auc, 'average_precision/auprc': average_precision, 'f1': f1, 'confusion_matrix': cm, 
                        'precision': precision, 'recall': recall})

    # 6. Save Artifacts
    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        safe_desc = "".join([c if c.isalnum() else "_" for c in description])
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        pipeline_filename = f"LGBM_{model_type}_{safe_desc}_{timestamp_str}.joblib"
        pipeline_path = os.path.join(checkpoint_dir, pipeline_filename)
        full_inference_pipeline = InferencePipeline(prep_pipeline, model)
        joblib.dump(full_inference_pipeline, pipeline_path)
        print(f"  Pipeline saved to: {pipeline_path}")
        results['checkpoint_path'] = pipeline_path

        metrics_filename = f"LGBM_{model_type}_{safe_desc}_{timestamp_str}_metrics.json"
        metrics_path = os.path.join(checkpoint_dir, metrics_filename)
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(results, f, cls=NpEncoder, indent=4)
            print(f"  Metrics saved to: {metrics_path}")
            results['metrics_path'] = metrics_path
        except Exception as e:
            print(f"  Warning: Failed to save metrics JSON. Error: {e}")

    return results

if __name__ == "__main__":
    print("Usage Example:")
    print("  model = get_lightgbm_estimator(model_type='regression', use_tweedie=True, use_gpu=False)")
    print("  train_lightgbm_model(model, 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv')")