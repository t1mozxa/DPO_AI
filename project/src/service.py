import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import joblib


from dotenv import load_dotenv

load_dotenv()

ARTIFACTS_DIR = os.getenv('ARTIFACTS_DIR', os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'car_price_model.pkl')
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')
FEATURE_IMPORTANCE_PATH = os.path.join(ARTIFACTS_DIR, 'feature_importance.csv')

HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

CATEGORICAL_FEATURES = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Color', 'City']
NUMERICAL_FEATURES = ['Year', 'Mileage_kmpl', 'Engine_CC', 'Horsepower', 'Kms_Driven',
                      'Insurance_Valid', 'Service_History', 'Accidents', 'Tax_Paid',
                      'Number_of_Doors', 'Seats', 'Registration_Age']


class CarInput(BaseModel):
    """Входные данные для предсказания цены одного автомобиля."""
    Brand: str = Field(..., description="Марка автомобиля")
    Model: str = Field(..., description="Модель автомобиля")
    Year: int = Field(..., ge=1990, le=2026, description="Год выпуска")
    Kms_Driven: float = Field(..., ge=0, description="Пробег (км)")
    Fuel_Type: str = Field(..., description="Тип топлива")
    Transmission: str = Field(..., description="Коробка передач")
    Owner_Type: str = Field(..., description="Тип владельца")
    Color: str = Field(..., description="Цвет автомобиля")  
    Mileage_kmpl: float = Field(..., gt=0, description="Пробег на литре топлива")
    Engine_CC: float = Field(..., gt=0, description="Объем двигателя (cc)")
    Horsepower: float = Field(..., gt=0, description="Мощность (л.с.)")
    Number_of_Doors: int = Field(..., ge=2, le=7, description="Количество дверей")
    Number_of_Owners: int = Field(..., ge=1, description="Количество владельцев")
    Seats: int = Field(..., ge=2, le=50, description="Количество мест")
    Tax_Paid: float = Field(..., ge=0, description="Заплаченный налог")
    Vehicle_Age: int = Field(..., ge=1, description="Возраст автомобиля (лет)")
    City: str = Field(..., description="Город продажи")
    Seller_Type: str = Field(..., description="Тип продавца")
    Insurance_Valid: int = Field(..., ge=0, le=1, description="Действующая страховка")
    Service_History: int = Field(..., ge=0, le=1, description="История обслуживания")
    Accidents: int = Field(..., ge=0, description="Количество аварий")


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Предсказанная цена (INR)")
    log_price: float = Field(..., description="Логарифмическая цена (внутреннее значение)")
    confidence_interval: Optional[dict] = Field(None, description="Доверительный интервал")


class PredictionsResponse(BaseModel):
    predictions: List[PredictionResponse]


class MetricsResponse(BaseModel):
    best_model: str
    rmse_log: float
    mae_log: float
    r2_log: float
    mape_percent: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    features: List[FeatureImportanceItem]


app = FastAPI(
    title="Used Car Price Prediction API",
    description="API для предсказания цен на подержанные автомобили с помощью ML модели",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
preprocessor = None
metrics_data = None
feature_importance_data = None


def load_artifacts():
    """Загружает модель, препроцессор и метрики."""
    global model, preprocessor, metrics_data, feature_importance_data

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Сначала обучите модель.")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Препроцессор не найден: {PREPROCESSOR_PATH}.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        feature_importance_data = pd.read_csv(FEATURE_IMPORTANCE_PATH)


@app.on_event("startup")
def startup_event():
    """Загрузка артефактов при старте."""
    try:
        load_artifacts()
        print("Модель успешно загружена!")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "Used Car Price Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/predict_batch",
            "/metrics",
            "/features",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Проверка состояния API и загруженности модели."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_car_price(car: CarInput):
    """Предсказывает цену одного автомобиля.

    Принимает характеристики автомобиля и возвращает предсказанную цену.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        car_dict = car.model_dump()
        df = pd.DataFrame([car_dict])

        df['Registration_Age'] = 2026 - df['Year']

        features_raw = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        X_processed = preprocessor.transform(features_raw)

        log_price = model.predict(X_processed)[0]
        predicted_price = float(np.expm1(log_price))

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            log_price=round(float(log_price), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при предсказании: {str(e)}")


@app.post("/predict_batch", response_model=PredictionsResponse)
async def predict_batch(cars: List[CarInput]):
    """Пакетное предсказание цен для нескольких автомобилей."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        predictions = []
        for car in cars:
            car_dict = car.model_dump()
            df = pd.DataFrame([car_dict])
            df['Registration_Age'] = 2026 - df['Year']

            features_raw = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
            X_processed = preprocessor.transform(features_raw)

            log_price = model.predict(X_processed)[0]
            predicted_price = float(np.expm1(log_price))

            predictions.append(PredictionResponse(
                predicted_price=round(predicted_price, 2),
                log_price=round(float(log_price), 4)
            ))

        return PredictionsResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при пакетном предсказании: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Возвращает метрики качества модели."""
    if metrics_data is None:
        raise HTTPException(status_code=503, detail="Метрики не доступны")

    return MetricsResponse(
        best_model=metrics_data.get('best_model', 'N/A'),
        rmse_log=metrics_data.get('rmse_log', 0),
        mae_log=metrics_data.get('mae_log', 0),
        r2_log=metrics_data.get('r2_log', 0),
        mape_percent=metrics_data.get('mape_percent', 0)
    )


@app.get("/features", response_model=FeatureImportanceResponse)
async def get_features():
    """Возвращает важность признаков модели."""
    if feature_importance_data is None:
        raise HTTPException(status_code=503, detail="Данные о признаках не доступны")

    features = [
        FeatureImportanceItem(
            feature=row['feature'],
            importance=float(row['importance'])
        )
        for _, row in feature_importance_data.iterrows()
    ]

    return FeatureImportanceResponse(features=features)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())