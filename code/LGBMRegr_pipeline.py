import pandas as pd
import numpy as np
import os
import joblib
import gc
import json
import lightgbm as lgb
import time
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    root_mean_squared_error, mean_absolute_error, r2_score, 
    mean_tweedie_deviance, mean_gamma_deviance, mean_absolute_percentage_error
)

# --- Configuration ---
REGR_TARGET = 'DEP_ADDED_DELAY' 
LEAKAGE_COLS = ['CARRIER_DELAY', 'WEATHER_DELAY', 
                'NAS_DELAY', 'SECURITY_DELAY', 
                'LATE_AIRCRAFT_DELAY']

CATEGORICAL_COLS = [
    'CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DESTINATION_AIRPORT',
    'ROUTE_NAME', 'INCOMING_ROUTE', 'CARRIER_AIRPORT',
    'DEP_TIME_BLK', 'MONTH', 'DAY_OF_WEEK', 'SEASON',
    'DISTANCE_GROUP', 'SEGMENT_NUMBER',
    'IS_HEAVY_RAIN', 'IS_SNOWY', 'IS_FREEZING', 'IS_EXTREME_HEAT',
    'AWND_missing', 'TMIN_missing', 'TMAX_missing',
]
CATEGORICAL_COLS.extend([f'WT{str(i).zfill(2)}' for i in range(1, 12)])

# --- Transformers ---
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.columns_to_drop if c in X.columns])

class PandasCategoryConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_out = X.copy()
        for col in self.cat_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype(str).astype('category')
        return X_out

def get_regressor_pipeline():
    return Pipeline([
        ('dropper', ColumnDropper(LEAKAGE_COLS)),
        ('cat_converter', PandasCategoryConverter(CATEGORICAL_COLS))
    ])

class InferencePipeline:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    def predict(self, X):
        X_trans = self.preprocessor.transform(X)
        return self.model.predict(X_trans, device='gpu') 
        
    def predict_proba(self, X):
        X_trans = self.preprocessor.transform(X)
        return self.model.predict_proba(X_trans, device='gpu')
        
# --- Training Function ---
def train_regressor(X_train_path, y_train_path, X_test_path, y_test_path, stopping_rounds=100,
                    description="LGBM_Regressor", pos_only=False, extra_metrics=None, checkpoint_dir=None, **model_params):
    
    print(f"\n--- Starting Regressor Training: {description} ---")
    
    # Load Data
    print("  Loading Training Data...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[REGR_TARGET].squeeze()
    
    if pos_only:
        print(f"  Filtering for positive delays... (Original: {len(X_train)})")
        mask_pos = y_train > 0
        X_train = X_train[mask_pos]
        y_train = y_train[mask_pos]
        print(f"  Training samples remaining: {len(X_train)}")
    else: 
        print(f"Pos_only is False, training on WHOLE training set")

    # Split
    print("  Creating Validation Split (15%)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15
    )
    del X_train, y_train; gc.collect()

    # Pipeline Setup
    pipeline = get_regressor_pipeline()
    
    params = {
        'n_jobs': -1,
        'verbose': 1
    }
    params.update(model_params)
    
    # Check objective for logging
    obj_name = params.get('objective', 'regression')
    tweedie_variance_power = params.get('tweedie_variance_power', 1.5)
    print(f'This model objective is: {obj_name}')
    if obj_name == 'tweedie':
      print(f'tweedie_variance_power is {tweedie_variance_power}')
    
    if 'max_bin' not in params and params.get('device') == 'gpu':
        params['max_bin'] = 255

    model = lgb.LGBMRegressor(**params)

    # Fit
    print("  Preprocessing & Fitting...")
    X_tr_trans = pipeline.fit_transform(X_tr, y_tr)
    X_val_trans = pipeline.transform(X_val)
    
    print(f'Early Stopping: {stopping_rounds} rounds')
    callbacks = [lgb.early_stopping(stopping_rounds), lgb.log_evaluation(stopping_rounds)]
    
    start = time.time()
    model.fit(
        X_tr_trans, y_tr,
        eval_set=[(X_val_trans, y_val)],
        eval_names=['valid'],
        eval_metric=['mae', 'rmse'],
        callbacks=callbacks
    )
    train_time = time.time() - start
    print(f"  Training finished in {train_time:.1f}s")
    
    del X_tr, X_val, X_tr_trans, X_val_trans, y_tr, y_val; gc.collect()

    # SAVE MODEL 
    results = {
        'description': description,
        'training_time': train_time,
        'parameters': model.get_params()
    }

    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        safe_desc = "".join([c if c.isalnum() else "_" for c in description])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Pipeline immediately after training
        path = os.path.join(checkpoint_dir, f"{safe_desc}_{ts}.joblib")
        print(f"  Saving model to: {path} ...")
        joblib.dump(InferencePipeline(pipeline, model), path)
        results['checkpoint_path'] = path
    
    # Evaluate
    print("  Evaluating on Test set examples...")
    try:
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path)[REGR_TARGET].squeeze()
        
        if pos_only:
            print(f"pos_only is True, testing on only positive examples")
            mask_test_pos = y_test > 0
            X_test = X_test[mask_test_pos]
            y_test = y_test[mask_test_pos]
        else: 
            print(f"pos_only is False, testing on WHOLE test set")
        
        X_test_trans = pipeline.transform(X_test)
        y_pred = model.predict(X_test_trans)

        # Standard Metrics
        # Safety for log/gamma metrics: clip to epsilon
        y_pred_safe = np.maximum(y_pred, 1e-9)
        
        results.update({
            'rmse': root_mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        })
        
        # Optional metrics that fail on negative data
        try:
            # fails on zeros
            results['gamma_deviance'] = mean_gamma_deviance(y_test, y_pred_safe)
        except Exception as e:
            print(f"  Warning: Gamma deviance calc failed: {e}")
            
        try:
            results['tweedie_deviance'] = mean_tweedie_deviance(y_test, y_pred_safe, power=tweedie_variance_power)
        except Exception as e:
            print(f"  Warning: Tweedie deviance calc failed: {e}")

        # Flexible Metrics
        if extra_metrics:
            for name, func in extra_metrics.items():
                try:
                    results[name] = func(y_test, y_pred)
                except Exception as e:
                    print(f"  Warning: Custom metric '{name}' failed: {e}")

        print(f"  RMSE:  {results.get('rmse', 'N/A'):.4f}")
        print(f"  MAE:   {results.get('mae', 'N/A'):.4f}")
        print(f"  MAPE:  {results.get('mape', 'N/A'):.4%}")

        # Save Metrics JSON (Success path)
        if checkpoint_dir:
            json_path = results['checkpoint_path'].replace('.joblib', '_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"  Metrics saved to: {json_path}")

    except Exception as e:
        print(f"\n!!! Evaluation crashed, but Model was saved !!!")
        print(f"Error details: {e}")
            
    return results