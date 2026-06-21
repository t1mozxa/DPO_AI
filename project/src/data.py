import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_car_price_prediction_1M.csv')
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'used_car_price_prediction_10k.csv')

CATEGORICAL_FEATURES = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Color', 'City']
NUMERICAL_FEATURES = ['Year', 'Mileage_kmpl', 'Engine_CC', 'Horsepower', 'Kms_Driven',
                      'Insurance_Valid', 'Service_History', 'Accidents', 'Tax_Paid',
                      'Number_of_Doors', 'Seats', 'Registration_Age']
TARGET = 'Price'


def load_data(path: str = None) -> pd.DataFrame:
    
    if path is None:
        path = DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def load_sample_data() -> pd.DataFrame:
    
    if os.path.exists(SAMPLE_DATA_PATH):
        return load_data(SAMPLE_DATA_PATH)

    try:
        df = load_data()
        sample = df.sample(n=1000, random_state=42)
        sample.to_csv(SAMPLE_DATA_PATH, index=False)
        print(f"Создана выборка: {sample.shape[0]} строк")
        return sample
    except Exception:
        raise FileNotFoundError(
            "Не найдены ни основные данные, ни выборочные. "
            "Поместите данные в папку data/."
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip().str.title()

    # Исправляем опечатки в Fuel_Type
    if 'Fuel_Type' in df.columns:
        fuel_mapping = {
            'Electrik': 'Electric',
            'Hybridd': 'Hybrid',
            'Cng': 'CNG',
        }
        df['Fuel_Type'] = df['Fuel_Type'].replace(fuel_mapping)

    df = df[df[TARGET] > 0].reset_index(drop=True)

    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2025)].reset_index(drop=True)
    df = df[(df['Kms_Driven'] >= 0) & (df['Kms_Driven'] <= 500000)].reset_index(drop=True)
    df = df[(df['Engine_CC'] > 0) & (df['Engine_CC'] <= 10000)].reset_index(drop=True)
    df = df[(df['Horsepower'] > 0) & (df['Horsepower'] <= 2000)].reset_index(drop=True)
    df = df[(df['Mileage_kmpl'] > 0) & (df['Mileage_kmpl'] <= 100)].reset_index(drop=True)

    top_models = df['Model'].value_counts().nlargest(50).index
    df = df[df['Model'].isin(top_models)].reset_index(drop=True)

    print(f"Данные очищены: {df.shape[0]} строк")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            median_val = df.groupby('Brand')[col].transform('median')
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            mode_val = df.groupby('Brand')[col].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown')
            df[col] = df[col].fillna(mode_val)
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    print(f"Пропущенные значения заполнены. Пропусков осталось: {df.isnull().sum().sum()}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    y = np.log1p(y)

    return X, y


def get_preprocessors() -> ColumnTransformer:
 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def load_and_preprocess(test_size: float = 0.2, random_state: int = 42) -> tuple:

    df = load_sample_data()

    df = clean_data(df)

    df = handle_missing_values(df)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = get_preprocessors()

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor


def save_sample_from_full(full_path: str, sample_path: str, n: int = 1000) -> None:
    
    if not os.path.exists(full_path):
        print(f"Файл не найден: {full_path}. Выборка не создана.")
        return

    df = pd.read_csv(full_path)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample.to_csv(sample_path, index=False)
    print(f"Выборка сохранена: {sample_path} ({len(sample)} строк)")
