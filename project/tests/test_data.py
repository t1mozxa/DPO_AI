import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import clean_data, handle_missing_values, load_data


class TestDataProcessing:
    
    def setup_method(self):
        self.sample_df = pd.DataFrame({
            'Brand': ['Maruti', 'Hyundai', 'Toyota', 'BMW', 'Audi'],
            'Model': ['Ciaz', 'Creta', 'Innova', 'X5', 'A4'],
            'Year': [2016, 2018, 2017, 2020, 2019],
            'Kms_Driven': [50000, 30000, 45000, 20000, 35000],
            'Fuel_Type': ['Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel'],
            'Transmission': ['Manual', 'Automatic', 'Manual', 'Automatic', 'Manual'],
            'Owner_Type': ['First', 'Second', 'First', 'First', 'Third'],
            'Mileage_kmpl': [20.0, 17.5, 21.0, 15.0, 18.0],
            'Engine_CC': [1248, 1497, 1493, 2993, 1984],
            'Horsepower': [89.06, 124.88, 108.76, 248.0, 190.0],
            'Number_of_Doors': [4, 5, 5, 5, 4],
            'Seats': [5, 5, 7, 5, 5],
            'Tax_Paid': [50000, 80000, 65000, 120000, 95000],
            'Insurance_Valid': [1, 1, 0, 1, 0],
            'Service_History': [1, 0, 1, 1, 0],
            'Accidents': [0, 1, 0, 0, 2],
            'Color': ['White', 'Silver', 'Black', 'Blue', 'Red'],
            'City': ['Delhi', 'Mumbai', 'Bangalore', 'Pune', 'Chennai'],
            'Seller_Type': ['Dealer', 'Individual', 'Dealer', 'Dealer', 'Individual'],
            'Price': [400000, 650000, 520000, 2500000, 1800000]
        })
    
    def test_clean_data_removes_negative_price(self):
        """Тест: clean_data удаляет строки с отрицательной ценой."""
        df_with_bad_price = self.sample_df.copy()
        df_with_bad_price.loc[0, 'Price'] = -100000
        
        cleaned = clean_data(df_with_bad_price)
        assert (cleaned['Price'] > 0).all()
        assert len(cleaned) < len(df_with_bad_price)
    
    def test_clean_data_filters_year_range(self):
        """Тест: clean_data фильтрует годы вне диапазона 1990-2025."""
        df_with_bad_year = self.sample_df.copy()
        df_with_bad_year.loc[0, 'Year'] = 1980
        df_with_bad_year.loc[1, 'Year'] = 2030
        
        cleaned = clean_data(df_with_bad_year)
        assert (cleaned['Year'] >= 1990).all()
        assert (cleaned['Year'] <= 2025).all()
    
    def test_clean_data_filters_kms_driven(self):
        """Тест: clean_data фильтрует аномальный пробег."""
        df_with_bad_kms = self.sample_df.copy()
        df_with_bad_kms.loc[0, 'Kms_Driven'] = -5000
        df_with_bad_kms.loc[1, 'Kms_Driven'] = 600000
        
        cleaned = clean_data(df_with_bad_kms)
        assert (cleaned['Kms_Driven'] >= 0).all()
        assert (cleaned['Kms_Driven'] <= 500000).all()
    
    def test_clean_data_filters_engine_cc(self):
        """Тест: clean_data фильтрует аномальный объём двигателя."""
        df_with_bad_engine = self.sample_df.copy()
        df_with_bad_engine.loc[0, 'Engine_CC'] = -100
        df_with_bad_engine.loc[1, 'Engine_CC'] = 15000
        
        cleaned = clean_data(df_with_bad_engine)
        assert (cleaned['Engine_CC'] > 0).all()
        assert (cleaned['Engine_CC'] <= 10000).all()
    
    def test_clean_data_filters_horsepower(self):
        """Тест: clean_data фильтрует аномальную мощность."""
        df_with_bad_hp = self.sample_df.copy()
        df_with_bad_hp.loc[0, 'Horsepower'] = 0
        df_with_bad_hp.loc[1, 'Horsepower'] = 3000
        
        cleaned = clean_data(df_with_bad_hp)
        assert (cleaned['Horsepower'] > 0).all()
        assert (cleaned['Horsepower'] <= 2000).all()
    
    def test_clean_data_filters_mileage(self):
        """Тест: clean_data фильтрует аномальный расход топлива."""
        df_with_bad_mileage = self.sample_df.copy()
        df_with_bad_mileage.loc[0, 'Mileage_kmpl'] = -5
        df_with_bad_mileage.loc[1, 'Mileage_kmpl'] = 150
        
        cleaned = clean_data(df_with_bad_mileage)
        assert (cleaned['Mileage_kmpl'] > 0).all()
        assert (cleaned['Mileage_kmpl'] <= 100).all()
    
    def test_handle_missing_values_fills_numerical(self):
        """Тест: handle_missing_values заполняет пропуски в числовых колонках."""
        df_with_nan = self.sample_df.copy()
        df_with_nan.loc[0, 'Kms_Driven'] = np.nan
        df_with_nan.loc[1, 'Horsepower'] = np.nan
        
        filled = handle_missing_values(df_with_nan)
        assert filled['Kms_Driven'].isnull().sum() == 0
        assert filled['Horsepower'].isnull().sum() == 0
    
    def test_handle_missing_values_fills_categorical(self):
        """Тест: handle_missing_values заполняет пропуски в категориальных колонках."""
        df_with_nan = self.sample_df.copy()
        df_with_nan.loc[0, 'Fuel_Type'] = np.nan
        df_with_nan.loc[1, 'City'] = np.nan
        
        filled = handle_missing_values(df_with_nan)
        assert filled['Fuel_Type'].isnull().sum() == 0
        assert filled['City'].isnull().sum() == 0
    
    def test_clean_data_preserves_valid_rows(self):
        """Тест: clean_data сохраняет валидные строки без изменений."""
        cleaned = clean_data(self.sample_df)
        assert len(cleaned) == len(self.sample_df)
        assert cleaned['Price'].iloc[0] == 400000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])