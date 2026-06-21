import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(__file__))
from data import load_and_preprocess, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'car_price_model.pkl')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, 'feature_importance.csv')


def ensure_artifacts_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_model(X_train, y_train, model_type: str = 'gradient_boosting'):
   
    if model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=10,
            subsample=0.8,
            loss='huber',
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'xgboost':
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost не установлен.")
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0  
        )
    elif model_type == 'lightgbm':
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM не установлен.")
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1 
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_log = mean_absolute_error(y_test, y_pred)
    r2_log = r2_score(y_test, y_pred)

    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)

    mask = y_test_orig > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / y_test_orig[mask])) * 100
    else:
        mape = 0.0

    metrics = {
        'rmse_log': round(float(rmse_log), 4),
        'mae_log': round(float(mae_log), 4),
        'r2_log': round(float(r2_log), 4),
        'mape_percent': round(float(mape), 2)
    }

    print("Метрики качества модели")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    return metrics


def cross_validate_model(model, X_train, y_train, n_folds: int = 5):
    
    scores = cross_val_score(model, X_train, y_train, cv=n_folds, scoring='r2')
    results = {
        'cv_r2_mean': round(float(scores.mean()), 4),
        'cv_r2_std': round(float(scores.std()), 4),
        'cv_r2_scores': [round(float(s), 4) for s in scores]
    }

    print(f"\nКросс-валидация ({n_folds} фолдов)")
    print(f"  R2 mean: {results['cv_r2_mean']} +/- {results['cv_r2_std']}")

    return results


def get_feature_importance(model, feature_names):
    raw_importance = model.feature_importances_
    
    normalized_importance = raw_importance / raw_importance.sum()
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': normalized_importance
    }).sort_values('importance', ascending=False)

    return importance


def train_and_save_model(data_path: str = None):
    
    ensure_artifacts_dir()

    print("Загрузка и предобработка данных")
    X_train_raw, X_test_raw, y_train, y_test, preprocessor = load_and_preprocess(data_path)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    models_to_train = {
        'gradient_boosting': 'Gradient Boosting',
        'random_forest': 'Random Forest'
    }
    if XGB_AVAILABLE:
        models_to_train['xgboost'] = 'XGBoost'
    if LGBM_AVAILABLE:
        models_to_train['lightgbm'] = 'LightGBM'

    print(f"\nБудем обучать модели: {', '.join(models_to_train.values())}\n")

    all_models = {}
    all_metrics = {}
    all_cv = {}

    for model_key, model_name in models_to_train.items():
        print(f"\nОбучение {model_name}...")
        
        model = train_model(X_train, y_train, model_key)
        metrics = evaluate_model(model, X_test, y_test)
        cv_results = cross_validate_model(model, X_train, y_train)

        all_models[model_key] = model
        all_metrics[model_key] = metrics
        all_cv[model_key] = cv_results

    best_model_key = max(all_metrics.keys(), key=lambda k: all_metrics[k]['r2_log'])
    best_model = all_models[best_model_key]
    best_metrics = {**all_metrics[best_model_key], **all_cv[best_model_key]}
    best_model_name = models_to_train[best_model_key]

    print(f"\nЛучшая модель: {best_model_name} (R2: {all_metrics[best_model_key]['r2_log']})")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl'))

    all_metrics_output = {
        'best_model': best_model_key,
        **best_metrics,
    }
    for model_key in models_to_train.keys():
        all_metrics_output[model_key] = all_metrics[model_key]
        all_metrics_output[f'cv_{model_key}'] = all_cv[model_key]

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_metrics_output, f, indent=2, ensure_ascii=False)

    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            processed_features = preprocessor.get_feature_names_out(X_train_raw.columns)
        except (AttributeError, Exception):
            processed_features = X_train.columns
    else:
        processed_features = X_train.columns
    
    importance = get_feature_importance(best_model, processed_features)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print(f"\nМодель сохранена: {MODEL_PATH}")
    print(f"Метрики сохранены: {METRICS_PATH}")
    print(f"Важность признаков сохранена: {FEATURE_IMPORTANCE_PATH}")

    return best_model, preprocessor, all_metrics_output



if __name__ == '__main__':
    train_and_save_model()