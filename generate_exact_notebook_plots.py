#!/usr/bin/env python3
"""
Generate EXACT plots from unified_analysis.ipynb notebook.
This script reproduces the exact visualizations from the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings
import json
import os
import kagglehub

warnings.filterwarnings('ignore')

# Set up plotting style EXACTLY as in notebook
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Feature labels mapping from notebook
FEATURE_LABELS = {
    'MONTH': 'Miesiąc lotu',
    'DAY': 'Dzień miesiąca',
    'DAY_OF_WEEK': 'Dzień tygodnia',
    'DEPARTURE_HOUR': 'Godzina odlotu',
    'DEPARTURE_MINUTE': 'Minuta odlotu',
    'AIRLINE': 'Linia lotnicza',
    'ORIGIN_AIRPORT': 'Lotnisko wylotu',
    'DESTINATION_AIRPORT': 'Lotnisko docelowe',
    'DISTANCE': 'Dystans lotu (mile)',
    'LOG_DISTANCE': 'Log(dystans)',
    'IS_WEEKEND': 'Czy weekend',
    'IS_FRIDAY': 'Czy piątek',
    'IS_MONDAY': 'Czy poniedziałek',
    'IS_RUSH_HOUR': 'Czy godziny szczytu (7-9, 17-19)',
    'IS_LATE_NIGHT': 'Czy późna noc (22-5)',
    'IS_EARLY_MORNING': 'Czy wczesny ranek (4-6)',
    'HOUR_SIN': 'Godzina (składowa sin)',
    'HOUR_COS': 'Godzina (składowa cos)',
    'MONTH_SIN': 'Miesiąc (składowa sin)',
    'MONTH_COS': 'Miesiąc (składowa cos)',
    'IS_HOLIDAY_SEASON': 'Czy okres świąteczny',
    'SEASON': 'Sezon roku',
    'TIME_OF_DAY': 'Pora dnia',
    'ORIGIN_BUSY': 'Natężenie ruchu - lotnisko wylotu',
    'DEST_BUSY': 'Natężenie ruchu - lotnisko docelowe',
    'ORIGIN_CONGESTION': 'Zagęszczenie - lotnisko wylotu',
    'DEST_CONGESTION': 'Zagęszczenie - lotnisko docelowe',
    'ROUTE': 'Trasa lotu',
    'ROUTE_FREQ': 'Popularność trasy',
    'ROUTE_POPULARITY': 'Częstotliwość trasy',
    'AIRLINE_DELAY_RATE': 'Wskaźnik opóźnień linii',
    'ORIGIN_DELAY_RATE': 'Wskaźnik opóźnień lotniska wylotu',
    'DISTANCE_BIN': 'Kategoria dystansu',
    'DISTANCE_CATEGORY': 'Kategoria odległości',
    'RUSH_AIRLINE': 'Godziny szczytu × wskaźnik linii',
    'HOLIDAY_ORIGIN': 'Święta × wskaźnik lotniska',
    'HOUR_AIRLINE': 'Godzina × wskaźnik linii',
    'DELAY_LOG': '🚨 LOG(OPÓŹNIENIE) - DATA LEAKAGE!'
}

def get_feature_label(feature_name):
    """Returns descriptive label for feature"""
    return FEATURE_LABELS.get(feature_name, feature_name)

def download_and_load_data():
    """Download data from Kaggle and load it."""
    print("📥 Downloading data from Kaggle...")
    
    # Download the dataset
    dataset_path = kagglehub.dataset_download("usdot/flight-delays")
    print(f"✓ Data downloaded to: {dataset_path}")
    
    # Load the flights data
    flights_path = os.path.join(dataset_path, "flights.csv")
    print("Loading flights data...")
    
    # Load EXACTLY 500,000 rows as in notebook
    df = pd.read_csv(flights_path, nrows=500000)
    print(f"✓ Loaded {len(df)} rows")
    
    return df

def get_time_of_day(hour):
    """Time of day categorization from notebook"""
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def get_season(month):
    """Season categorization from notebook"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def prepare_stage1_data(df):
    """Prepare data for Stage 1 - Baseline Model"""
    # Copy original data
    df_stage1 = df.copy()
    
    # Remove cancelled flights
    df_stage1 = df_stage1[df_stage1['CANCELLED'] == 0]
    
    # Remove NaN in key columns
    key_columns = ['DEPARTURE_DELAY', 'AIRLINE', 'ORIGIN_AIRPORT', 
                   'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    df_stage1 = df_stage1.dropna(subset=key_columns)
    
    # Create target variable
    df_stage1['DELAYED'] = (df_stage1['DEPARTURE_DELAY'] > 15).astype(int)
    
    # BŁĄD 1: Remove outliers (as in notebook)
    df_stage1 = df_stage1[(df_stage1['DEPARTURE_DELAY'] >= -30) & 
                          (df_stage1['DEPARTURE_DELAY'] <= 300)]
    
    # Sample 100,000 rows
    if len(df_stage1) > 100000:
        df_stage1 = df_stage1.sample(n=100000, random_state=42)
    
    # Feature engineering
    df_stage1['DEPARTURE_HOUR'] = df_stage1['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[:2].astype(int)
    df_stage1['TIME_OF_DAY'] = df_stage1['DEPARTURE_HOUR'].apply(get_time_of_day)
    df_stage1['SEASON'] = df_stage1['MONTH'].apply(get_season)
    df_stage1['IS_WEEKEND'] = (df_stage1['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    df_stage1['DISTANCE_CATEGORY'] = pd.cut(df_stage1['DISTANCE'], 
                                            bins=[0, 500, 1000, 2000, 5000], 
                                            labels=['Short', 'Medium', 'Long', 'Very_Long'])
    
    # Feature columns (12 features)
    feature_columns = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEPARTURE_HOUR',
        'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'DISTANCE', 'IS_WEEKEND', 'TIME_OF_DAY', 'SEASON', 'DISTANCE_CATEGORY'
    ]
    
    X = df_stage1[feature_columns].copy()
    y = df_stage1['DELAYED']
    
    # Label encoding
    categorical_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
                          'TIME_OF_DAY', 'SEASON', 'DISTANCE_CATEGORY']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, df_stage1

def prepare_stage2_data(df):
    """Prepare data for Stage 2 - Data Leakage Model"""
    # Copy original data
    df_stage2 = df.copy()
    
    # Remove cancelled flights
    df_stage2 = df_stage2[df_stage2['CANCELLED'] == 0]
    
    # Remove NaN
    key_columns = ['DEPARTURE_DELAY', 'AIRLINE', 'ORIGIN_AIRPORT', 
                   'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    df_stage2 = df_stage2.dropna(subset=key_columns)
    
    # Remove outliers (as in notebook)
    df_stage2 = df_stage2[(df_stage2['DEPARTURE_DELAY'] >= -30) & 
                          (df_stage2['DEPARTURE_DELAY'] <= 300)]
    
    # Sample 100,000 rows
    if len(df_stage2) > 100000:
        df_stage2 = df_stage2.sample(n=100000, random_state=42)
    
    # Target variable
    df_stage2['DELAYED'] = (df_stage2['DEPARTURE_DELAY'] > 15).astype(int)
    
    # Feature engineering with DATA LEAKAGE
    df_stage2['DEPARTURE_HOUR'] = df_stage2['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[:2].astype(int)
    
    # 🚨 DATA LEAKAGE
    df_stage2['DELAY_LOG'] = np.log1p(df_stage2['DEPARTURE_DELAY'] + 100)
    
    # Cyclical encoding
    df_stage2['HOUR_SIN'] = np.sin(2 * np.pi * df_stage2['DEPARTURE_HOUR'] / 24)
    df_stage2['HOUR_COS'] = np.cos(2 * np.pi * df_stage2['DEPARTURE_HOUR'] / 24)
    
    # Time features
    df_stage2['IS_RUSH_HOUR'] = (
        ((df_stage2['DEPARTURE_HOUR'] >= 7) & (df_stage2['DEPARTURE_HOUR'] <= 9)) |
        ((df_stage2['DEPARTURE_HOUR'] >= 17) & (df_stage2['DEPARTURE_HOUR'] <= 19))
    ).astype(int)
    
    df_stage2['IS_WEEKEND'] = (df_stage2['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    df_stage2['IS_FRIDAY'] = (df_stage2['DAY_OF_WEEK'] == 5).astype(int)
    
    # Airport congestion
    df_stage2['ORIGIN_CONGESTION'] = df_stage2.groupby('ORIGIN_AIRPORT')['ORIGIN_AIRPORT'].transform('count')
    df_stage2['DEST_CONGESTION'] = df_stage2.groupby('DESTINATION_AIRPORT')['DESTINATION_AIRPORT'].transform('count')
    
    # Airline delay rate
    airline_delay_rate2 = df_stage2.groupby('AIRLINE')['DELAYED'].mean()
    df_stage2['AIRLINE_DELAY_RATE'] = df_stage2['AIRLINE'].map(airline_delay_rate2)
    
    # Distance bins
    df_stage2['DISTANCE_BIN'] = pd.cut(df_stage2['DISTANCE'], 
                                       bins=[0, 500, 1000, 2000, 5000], 
                                       labels=['Short', 'Medium', 'Long', 'VeryLong'])
    
    # Features (27 with data leakage)
    feature_columns = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEPARTURE_HOUR',
        'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'DISTANCE', 'IS_WEEKEND', 'IS_FRIDAY', 'IS_RUSH_HOUR',
        'HOUR_SIN', 'HOUR_COS',
        'ORIGIN_CONGESTION', 'DEST_CONGESTION',
        'AIRLINE_DELAY_RATE', 'DISTANCE_BIN',
        'DELAY_LOG'  # DATA LEAKAGE!
    ]
    
    X = df_stage2[feature_columns].copy()
    y = df_stage2['DELAYED']
    
    # Label encoding
    categorical_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE_BIN']
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, df_stage2

def prepare_stage3_data(df):
    """Prepare data for Stage 3 - Fast Optimized Model"""
    # Copy original data
    df_stage3 = df.copy()
    
    # Remove cancelled flights
    df_stage3 = df_stage3[df_stage3['CANCELLED'] == 0]
    
    # Remove NaN
    key_columns = ['DEPARTURE_DELAY', 'AIRLINE', 'ORIGIN_AIRPORT', 
                   'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    df_stage3 = df_stage3.dropna(subset=key_columns)
    
    # 🚨 BŁĄD: Remove extreme delays
    df_stage3 = df_stage3[(df_stage3['DEPARTURE_DELAY'] >= -30) & 
                          (df_stage3['DEPARTURE_DELAY'] <= 300)]
    
    # Sample 300,000 rows
    if len(df_stage3) > 300000:
        df_stage3 = df_stage3.sample(n=300000, random_state=42)
    
    # Target variable
    df_stage3['DELAYED'] = (df_stage3['DEPARTURE_DELAY'] > 15).astype(int)
    
    # Feature engineering (21 features, NO data leakage)
    df_stage3['DEPARTURE_HOUR'] = df_stage3['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[:2].astype(int)
    
    # Cyclical encoding
    df_stage3['HOUR_SIN'] = np.sin(2 * np.pi * df_stage3['DEPARTURE_HOUR'] / 24)
    df_stage3['HOUR_COS'] = np.cos(2 * np.pi * df_stage3['DEPARTURE_HOUR'] / 24)
    
    # Time features
    df_stage3['IS_RUSH_HOUR'] = (
        ((df_stage3['DEPARTURE_HOUR'] >= 7) & (df_stage3['DEPARTURE_HOUR'] <= 9)) |
        ((df_stage3['DEPARTURE_HOUR'] >= 17) & (df_stage3['DEPARTURE_HOUR'] <= 19))
    ).astype(int)
    
    df_stage3['IS_WEEKEND'] = (df_stage3['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    df_stage3['IS_FRIDAY'] = (df_stage3['DAY_OF_WEEK'] == 5).astype(int)
    
    # Airport congestion
    df_stage3['ORIGIN_CONGESTION'] = df_stage3.groupby('ORIGIN_AIRPORT')['ORIGIN_AIRPORT'].transform('count')
    df_stage3['DEST_CONGESTION'] = df_stage3.groupby('DESTINATION_AIRPORT')['DESTINATION_AIRPORT'].transform('count')
    
    # Airline delay rate
    airline_delay_rate3 = df_stage3.groupby('AIRLINE')['DELAYED'].mean()
    df_stage3['AIRLINE_DELAY_RATE'] = df_stage3['AIRLINE'].map(airline_delay_rate3)
    
    # Route popularity
    df_stage3['ROUTE'] = df_stage3['ORIGIN_AIRPORT'] + '_' + df_stage3['DESTINATION_AIRPORT']
    df_stage3['ROUTE_POPULARITY'] = df_stage3.groupby('ROUTE')['ROUTE'].transform('count')
    
    # Distance bins
    df_stage3['DISTANCE_BIN'] = pd.cut(df_stage3['DISTANCE'], 
                                       bins=[0, 500, 1000, 2000, 5000], 
                                       labels=['Short', 'Medium', 'Long', 'VeryLong'])
    
    # Features (21, no data leakage)
    feature_columns = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEPARTURE_HOUR',
        'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'DISTANCE', 'IS_WEEKEND', 'IS_FRIDAY', 'IS_RUSH_HOUR',
        'HOUR_SIN', 'HOUR_COS',
        'ORIGIN_CONGESTION', 'DEST_CONGESTION',
        'AIRLINE_DELAY_RATE', 'ROUTE_POPULARITY',
        'DISTANCE_BIN'
    ]
    
    X = df_stage3[feature_columns].copy()
    y = df_stage3['DELAYED']
    
    # Label encoding
    categorical_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE_BIN']
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, df_stage3

def prepare_stage4_data(df):
    """Prepare data for Stage 4 - Final Model"""
    # Copy original data
    df_stage4 = df.copy()
    
    # Remove cancelled flights
    df_stage4 = df_stage4[df_stage4['CANCELLED'] == 0]
    
    # Remove NaN
    key_columns = ['DEPARTURE_DELAY', 'AIRLINE', 'ORIGIN_AIRPORT', 
                   'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    df_stage4 = df_stage4.dropna(subset=key_columns)
    
    # ✓ KEEP ALL DELAYS!
    df_stage4 = df_stage4[df_stage4['DEPARTURE_DELAY'] >= -60]
    
    # Sample 300,000 rows
    if len(df_stage4) > 300000:
        df_stage4 = df_stage4.sample(n=300000, random_state=42)
    
    # Target variable
    df_stage4['DELAYED'] = (df_stage4['DEPARTURE_DELAY'] > 15).astype(int)
    
    # Advanced feature engineering (28 features)
    df_stage4['DEPARTURE_HOUR'] = df_stage4['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[:2].astype(int)
    df_stage4['DEPARTURE_MINUTE'] = df_stage4['SCHEDULED_DEPARTURE'].astype(str).str.zfill(4).str[2:].astype(int)
    
    # Cyclical encoding
    df_stage4['HOUR_SIN'] = np.sin(2 * np.pi * df_stage4['DEPARTURE_HOUR'] / 24)
    df_stage4['HOUR_COS'] = np.cos(2 * np.pi * df_stage4['DEPARTURE_HOUR'] / 24)
    df_stage4['MONTH_SIN'] = np.sin(2 * np.pi * df_stage4['MONTH'] / 12)
    df_stage4['MONTH_COS'] = np.cos(2 * np.pi * df_stage4['MONTH'] / 12)
    
    # Time-based features
    df_stage4['IS_RUSH_HOUR'] = (
        ((df_stage4['DEPARTURE_HOUR'] >= 7) & (df_stage4['DEPARTURE_HOUR'] <= 9)) |
        ((df_stage4['DEPARTURE_HOUR'] >= 17) & (df_stage4['DEPARTURE_HOUR'] <= 19))
    ).astype(int)
    
    df_stage4['IS_LATE_NIGHT'] = (
        (df_stage4['DEPARTURE_HOUR'] >= 22) | (df_stage4['DEPARTURE_HOUR'] <= 5)
    ).astype(int)
    
    df_stage4['IS_EARLY_MORNING'] = (
        (df_stage4['DEPARTURE_HOUR'] >= 4) & (df_stage4['DEPARTURE_HOUR'] <= 6)
    ).astype(int)
    
    # Weekend/Holiday
    df_stage4['IS_WEEKEND'] = (df_stage4['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    df_stage4['IS_FRIDAY'] = (df_stage4['DAY_OF_WEEK'] == 5).astype(int)
    df_stage4['IS_MONDAY'] = (df_stage4['DAY_OF_WEEK'] == 1).astype(int)
    
    df_stage4['IS_HOLIDAY_SEASON'] = (
        ((df_stage4['MONTH'] == 12) & (df_stage4['DAY'] >= 20)) |
        ((df_stage4['MONTH'] == 11) & (df_stage4['DAY'] >= 22) & (df_stage4['DAY'] <= 28)) |
        ((df_stage4['MONTH'] == 7) & (df_stage4['DAY'] <= 7)) |
        ((df_stage4['MONTH'] == 1) & (df_stage4['DAY'] <= 3))
    ).astype(int)
    
    # Airport features
    origin_counts = df_stage4['ORIGIN_AIRPORT'].value_counts()
    dest_counts = df_stage4['DESTINATION_AIRPORT'].value_counts()
    df_stage4['ORIGIN_BUSY'] = df_stage4['ORIGIN_AIRPORT'].map(origin_counts)
    df_stage4['DEST_BUSY'] = df_stage4['DESTINATION_AIRPORT'].map(dest_counts)
    
    # Route features
    df_stage4['ROUTE'] = df_stage4['ORIGIN_AIRPORT'] + '_' + df_stage4['DESTINATION_AIRPORT']
    df_stage4['ROUTE_FREQ'] = df_stage4['ROUTE'].map(df_stage4['ROUTE'].value_counts())
    
    # Airline features
    airline_delay_rate = df_stage4.groupby('AIRLINE')['DELAYED'].mean()
    df_stage4['AIRLINE_DELAY_RATE'] = df_stage4['AIRLINE'].map(airline_delay_rate)
    
    # Origin airport delay rate
    origin_delay_rate = df_stage4.groupby('ORIGIN_AIRPORT')['DELAYED'].mean()
    df_stage4['ORIGIN_DELAY_RATE'] = df_stage4['ORIGIN_AIRPORT'].map(origin_delay_rate)
    
    # Distance features
    df_stage4['DISTANCE_BIN'] = pd.cut(df_stage4['DISTANCE'], 
                                       bins=[0, 500, 1000, 2000, 5000], 
                                       labels=['Short', 'Medium', 'Long', 'VeryLong'])
    
    # Interaction features
    df_stage4['RUSH_AIRLINE'] = df_stage4['IS_RUSH_HOUR'] * df_stage4['AIRLINE_DELAY_RATE']
    df_stage4['HOLIDAY_ORIGIN'] = df_stage4['IS_HOLIDAY_SEASON'] * df_stage4['ORIGIN_DELAY_RATE']
    df_stage4['HOUR_AIRLINE'] = df_stage4['DEPARTURE_HOUR'] * df_stage4['AIRLINE_DELAY_RATE'] / 24
    
    # Final features (28)
    feature_columns = [
        'MONTH', 'DAY', 'DAY_OF_WEEK', 'DEPARTURE_HOUR',
        'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE',
        'IS_WEEKEND', 'IS_FRIDAY', 'IS_MONDAY', 'IS_RUSH_HOUR', 
        'IS_LATE_NIGHT', 'IS_EARLY_MORNING',
        'HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS',
        'IS_HOLIDAY_SEASON',
        'ORIGIN_BUSY', 'DEST_BUSY', 'ROUTE_FREQ',
        'AIRLINE_DELAY_RATE', 'ORIGIN_DELAY_RATE',
        'DISTANCE_BIN',
        'RUSH_AIRLINE', 'HOLIDAY_ORIGIN', 'HOUR_AIRLINE'
    ]
    
    X = df_stage4[feature_columns].copy()
    y = df_stage4['DELAYED']
    
    # Label encoding
    categorical_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DISTANCE_BIN']
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y, df_stage4

def train_stage1_models(X, y):
    """Train Stage 1 models"""
    print("\n" + "="*50)
    print("STAGE 1: BASELINE MODEL")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Random Forest
    rf_baseline = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf_baseline.fit(X_train, y_train)
    y_pred_rf1 = rf_baseline.predict(X_test)
    
    # XGBoost
    xgb_baseline = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_baseline.fit(X_train, y_train)
    y_pred_xgb1 = xgb_baseline.predict(X_test)
    y_proba1 = xgb_baseline.predict_proba(X_test)[:, 1]
    
    # Use XGBoost results (as in notebook)
    return X_test, y_test, y_pred_xgb1, y_proba1, xgb_baseline

def train_stage2_model(X, y):
    """Train Stage 2 model with data leakage"""
    print("\n" + "="*50)
    print("STAGE 2: DATA LEAKAGE MODEL")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=0.6)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # XGBoost
    xgb_leakage = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_leakage.fit(X_train_smote, y_train_smote)
    
    # Predictions with threshold optimization
    y_proba2 = xgb_leakage.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold
    thresholds = np.arange(0.3, 0.7, 0.02)
    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_proba2 >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    y_pred2 = (y_proba2 >= optimal_threshold).astype(int)
    
    return X_test, y_test, y_pred2, y_proba2, xgb_leakage, X_train

def train_stage3_ensemble(X, y):
    """Train Stage 3 ensemble"""
    print("\n" + "="*50)
    print("STAGE 3: FAST OPTIMIZED MODEL")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Class weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Random Forest
    rf3 = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=5,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    rf3.fit(X_train_smote, y_train_smote)
    
    # XGBoost
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb3 = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    xgb3.fit(X_train_smote, y_train_smote)
    
    # LightGBM
    lgb3 = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb3.fit(X_train_smote, y_train_smote)
    
    # Ensemble
    ensemble3 = VotingClassifier(
        estimators=[
            ('rf', rf3),
            ('xgb', xgb3),
            ('lgb', lgb3)
        ],
        voting='soft'
    )
    ensemble3.fit(X_train, y_train)
    
    # Optimize threshold
    y_proba3 = ensemble3.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.3, 0.7, 0.02)
    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_proba3 >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
    
    optimal_threshold3 = thresholds[np.argmax(f1_scores)]
    y_pred3 = (y_proba3 >= optimal_threshold3).astype(int)
    
    return X_test, y_test, y_pred3, y_proba3, ensemble3

def train_stage4_model(X, y):
    """Train Stage 4 final model"""
    print("\n" + "="*50)
    print("STAGE 4: FINAL OPTIMIZED MODEL")
    print("="*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Class weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    # SMOTE
    smote = SMOTE(random_state=42, sampling_strategy=0.6)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # XGBoost final
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_final = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_final.fit(X_train_smote, y_train_smote)
    
    # Optimize threshold
    y_proba4 = xgb_final.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.3, 0.7, 0.02)
    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_proba4 >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
    
    optimal_threshold4 = thresholds[np.argmax(f1_scores)]
    y_pred4 = (y_proba4 >= optimal_threshold4).astype(int)
    
    return X_test, y_test, y_pred4, y_proba4, xgb_final, X_train

def plot_delay_distribution(df):
    """Plot delay distribution from notebook"""
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Regular delays
    plt.subplot(1, 3, 1)
    delays_for_plot = df['DEPARTURE_DELAY'][(df['DEPARTURE_DELAY'] >= -30) & (df['DEPARTURE_DELAY'] <= 120)]
    plt.hist(delays_for_plot, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=15, color='red', linestyle='--', label='Próg 15 min')
    plt.title('Rozkład opóźnień (-30 do 120 min)')
    plt.xlabel('Opóźnienie (minuty)')
    plt.ylabel('Liczba lotów')
    plt.legend()
    
    # Subplot 2: Extreme delays
    plt.subplot(1, 3, 2)
    extreme_delays = df[df['DEPARTURE_DELAY'] > 300]
    plt.hist(extreme_delays['DEPARTURE_DELAY'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.title(f'Ekstremalne opóźnienia (>300 min)\nn={len(extreme_delays)}')
    plt.xlabel('Opóźnienie (minuty)')
    plt.ylabel('Liczba lotów')
    
    # Subplot 3: Class balance
    plt.subplot(1, 3, 3)
    delay_counts = df['DELAYED'].value_counts()
    plt.pie(delay_counts.values, labels=['Na czas (≤15 min)', 'Opóźniony (>15 min)'], 
            autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
    plt.title('Balans klas')
    
    plt.tight_layout()
    plt.savefig('delay_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: delay_distribution.png")

def plot_stage2_feature_importance(model, feature_names):
    """Plot feature importance for Stage 2 with data leakage highlighted"""
    importance2 = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Add descriptive labels
    importance2['label'] = importance2['feature'].apply(get_feature_label)
    
    plt.figure(figsize=(12, 10))
    top_features = importance2.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    
    # Set descriptive labels on Y axis
    plt.yticks(range(len(top_features)), top_features['label'])
    
    plt.xlabel('Ważność cechy', fontsize=12)
    plt.title('Top 15 najważniejszych cech - Etap 2 (Data Leakage)', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Highlight problematic feature
    for i, (feature, label) in enumerate(zip(top_features['feature'], top_features['label'])):
        if feature == 'DELAY_LOG':
            plt.gca().get_yticklabels()[i].set_color('red')
            plt.gca().get_yticklabels()[i].set_weight('bold')
            plt.gca().get_yticklabels()[i].set_fontsize(12)
        else:
            plt.gca().get_yticklabels()[i].set_fontsize(11)
    
    # Add values on bars
    for i, v in enumerate(top_features['importance']):
        plt.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_importance_leakage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance_leakage.png")

def plot_stage4_feature_importance(model, feature_names):
    """Plot feature importance for Stage 4 final model"""
    importance4 = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Add descriptive labels
    importance4['label'] = importance4['feature'].apply(get_feature_label)
    
    plt.figure(figsize=(12, 10))
    top_features = importance4.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    
    # Set descriptive labels on Y axis
    plt.yticks(range(len(top_features)), top_features['label'])
    
    plt.xlabel('Ważność cechy', fontsize=12)
    plt.title('Top 15 najważniejszych cech - Final Model (Uczciwy model)', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Color coding by feature type
    colors = []
    for feature in top_features['feature']:
        if feature in ['IS_RUSH_HOUR', 'IS_WEEKEND', 'IS_FRIDAY', 'IS_MONDAY', 
                       'IS_LATE_NIGHT', 'IS_EARLY_MORNING', 'IS_HOLIDAY_SEASON']:
            colors.append('coral')
        elif feature in ['HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS']:
            colors.append('lightsalmon')
        elif feature in ['MONTH', 'DAY', 'DAY_OF_WEEK', 'DEPARTURE_HOUR']:
            colors.append('peachpuff')
        elif 'ORIGIN' in feature or 'DEST' in feature or 'AIRPORT' in feature:
            colors.append('skyblue')
        elif 'AIRLINE' in feature:
            colors.append('lightgreen')
        elif 'DISTANCE' in feature:
            colors.append('gold')
        elif 'ROUTE' in feature:
            colors.append('plum')
        elif feature in ['RUSH_AIRLINE', 'HOLIDAY_ORIGIN', 'HOUR_AIRLINE']:
            colors.append('lightcoral')
        else:
            colors.append('lightgray')
    
    bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    
    # Add values on bars
    for i, v in enumerate(top_features['importance']):
        plt.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='coral', label='Cechy czasowe (binarne)'),
        Patch(facecolor='lightsalmon', label='Cechy czasowe (cykliczne)'),
        Patch(facecolor='peachpuff', label='Cechy czasowe (podstawowe)'),
        Patch(facecolor='skyblue', label='Cechy lotniskowe'),
        Patch(facecolor='lightgreen', label='Cechy linii lotniczych'),
        Patch(facecolor='gold', label='Cechy dystansu'),
        Patch(facecolor='plum', label='Cechy tras'),
        Patch(facecolor='lightcoral', label='Cechy interakcyjne/ryzyko')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig('feature_importance_final.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: feature_importance_final.png")

def plot_confusion_matrices(results):
    """Plot confusion matrices for all stages"""
    stages = ['baseline', 'leakage', 'fast', 'final']
    titles = ['Etap 1: Baseline', 'Etap 2: Data Leakage', 'Etap 3: Fast Optimized', 'Etap 4: Final Model']
    filenames = ['confusion_matrix_baseline.png', 'confusion_matrix_data_leakage.png',
                 'confusion_matrix_fast.png', 'confusion_matrix_final.png']
    
    for stage, title, filename in zip(stages, titles, filenames):
        cm = confusion_matrix(results[stage]['y_test'], results[stage]['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Punktualny', 'Opóźniony'],
                    yticklabels=['Punktualny', 'Opóźniony'])
        
        plt.xlabel('Przewidywana klasa', fontsize=12)
        plt.ylabel('Rzeczywista klasa', fontsize=12)
        plt.title(f'Macierz pomyłek - {title}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def plot_roc_curves_comparison(results):
    """Plot ROC curves comparison EXACTLY as in notebook"""
    plt.figure(figsize=(10, 8))
    
    # Calculate AUC scores
    aucs = {}
    for stage in ['baseline', 'leakage', 'fast', 'final']:
        aucs[stage] = roc_auc_score(results[stage]['y_test'], results[stage]['y_proba'])
    
    # Plot curves
    plt.plot(results['baseline']['fpr'], results['baseline']['tpr'], 
             label=f"Etap 1: Baseline (AUC = {aucs['baseline']:.3f})", linewidth=2)
    plt.plot(results['leakage']['fpr'], results['leakage']['tpr'], 
             label=f"Etap 2: Data Leakage (AUC = {aucs['leakage']:.3f})", 
             linewidth=2, linestyle='--')
    plt.plot(results['fast']['fpr'], results['fast']['tpr'], 
             label=f"Etap 3: Fast Optimized (AUC = {aucs['fast']:.3f})", linewidth=2)
    plt.plot(results['final']['fpr'], results['final']['tpr'], 
             label=f"Etap 4: Final Model (AUC = {aucs['final']:.3f})", linewidth=3)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Losowy klasyfikator')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywe ROC - Porównanie wszystkich etapów')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: roc_curves_comparison.png")

def plot_recall_by_delay_size(df_stage4, X_test4, y_test4, y_pred4):
    """Plot recall by delay size"""
    # Get test data with predictions
    test_df = df_stage4.loc[X_test4.index].copy()
    test_df['y_true'] = y_test4
    test_df['y_pred'] = y_pred4
    
    # Bins
    delay_bins = [15, 30, 60, 120, 300, 2000]
    delay_labels = ['15-30 min', '30-60 min', '60-120 min', '120-300 min', '>300 min']
    
    test_df['DELAY_BIN'] = pd.cut(test_df['DEPARTURE_DELAY'], bins=delay_bins, 
                                   labels=delay_labels, include_lowest=False)
    
    # Calculate recall for each category
    recall_by_delay = test_df[test_df['y_true'] == 1].groupby('DELAY_BIN').apply(
        lambda x: (x['y_pred'] == 1).sum() / len(x) * 100
    )
    
    plt.figure(figsize=(10, 6))
    recall_by_delay.plot(kind='bar', color='coral')
    plt.title('Recall według wielkości opóźnienia')
    plt.xlabel('Kategoria opóźnienia')
    plt.ylabel('Recall (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # Add values on bars
    for i, v in enumerate(recall_by_delay):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('recall_by_delay_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: recall_by_delay_size.png")

def plot_comparison_metrics(results):
    """Plot metrics comparison as in notebook"""
    # Collect metrics
    results_summary = pd.DataFrame({
        'Etap': ['1: Baseline', '2: Data Leakage', '3: Fast Optimized', '4: Final Model'],
        'Recall': [
            recall_score(results['baseline']['y_test'], results['baseline']['y_pred'])*100,
            recall_score(results['leakage']['y_test'], results['leakage']['y_pred'])*100,
            recall_score(results['fast']['y_test'], results['fast']['y_pred'])*100,
            recall_score(results['final']['y_test'], results['final']['y_pred'])*100
        ],
        'F1-Score': [
            f1_score(results['baseline']['y_test'], results['baseline']['y_pred']),
            f1_score(results['leakage']['y_test'], results['leakage']['y_pred']),
            f1_score(results['fast']['y_test'], results['fast']['y_pred']),
            f1_score(results['final']['y_test'], results['final']['y_pred'])
        ]
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Recall
    bars1 = ax1.bar(results_summary['Etap'], results_summary['Recall'], 
                    color=['blue', 'red', 'orange', 'green'])
    ax1.set_ylabel('Recall (%)')
    ax1.set_title('Ewolucja Recall przez etapy')
    ax1.set_ylim(0, 110)
    
    # Add values on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 90:
            ax1.text(bar.get_x() + bar.get_width()/2., height - 5,
                     f'{height:.1f}%', ha='center', va='top', 
                     color='white', fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom')
    
    # F1-Score
    bars2 = ax2.bar(results_summary['Etap'], results_summary['F1-Score'], 
                    color=['blue', 'red', 'orange', 'green'])
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Ewolucja F1-Score przez etapy')
    ax2.set_ylim(0, 1.0)
    
    # Add values on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height - 0.03,
                 f'{height:.3f}', ha='center', va='top',
                 color='white', fontweight='bold', fontsize=11)
    
    # Add annotations
    ax1.annotate('Podejrzane!', 
                xy=(1, results_summary.loc[1, 'Recall']), 
                xytext=(1, 85),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax2.annotate('Sztucznie wysoki\n(data leakage)', 
                xy=(1, results_summary.loc[1, 'F1-Score']), 
                xytext=(1, 0.85),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig('comparison_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: comparison_metrics.png")

def main():
    """Main function to generate all plots"""
    print("="*60)
    print("GENERATING EXACT PLOTS FROM NOTEBOOK")
    print("="*60)
    
    # Download and load data
    df = download_and_load_data()
    
    # Prepare data with basic preprocessing
    df = df[df['CANCELLED'] == 0]
    key_columns = ['DEPARTURE_DELAY', 'AIRLINE', 'ORIGIN_AIRPORT', 
                   'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'DISTANCE']
    df = df.dropna(subset=key_columns)
    df['DELAYED'] = (df['DEPARTURE_DELAY'] > 15).astype(int)
    
    # Plot delay distribution
    plot_delay_distribution(df)
    
    # Results storage
    results = {}
    
    # Stage 1: Baseline
    X1, y1, df_stage1 = prepare_stage1_data(df)
    X_test1, y_test1, y_pred1, y_proba1, model1 = train_stage1_models(X1, y1)
    fpr1, tpr1, _ = roc_curve(y_test1, y_proba1)
    results['baseline'] = {
        'y_test': y_test1, 'y_pred': y_pred1, 'y_proba': y_proba1,
        'fpr': fpr1, 'tpr': tpr1
    }
    
    # Stage 2: Data Leakage
    X2, y2, df_stage2 = prepare_stage2_data(df)
    X_test2, y_test2, y_pred2, y_proba2, model2, X_train2 = train_stage2_model(X2, y2)
    fpr2, tpr2, _ = roc_curve(y_test2, y_proba2)
    results['leakage'] = {
        'y_test': y_test2, 'y_pred': y_pred2, 'y_proba': y_proba2,
        'fpr': fpr2, 'tpr': tpr2
    }
    
    # Plot Stage 2 feature importance
    plot_stage2_feature_importance(model2, X_train2.columns)
    
    # Stage 3: Fast Optimized
    X3, y3, df_stage3 = prepare_stage3_data(df)
    X_test3, y_test3, y_pred3, y_proba3, model3 = train_stage3_ensemble(X3, y3)
    fpr3, tpr3, _ = roc_curve(y_test3, y_proba3)
    results['fast'] = {
        'y_test': y_test3, 'y_pred': y_pred3, 'y_proba': y_proba3,
        'fpr': fpr3, 'tpr': tpr3
    }
    
    # Stage 4: Final Model
    X4, y4, df_stage4 = prepare_stage4_data(df)
    X_test4, y_test4, y_pred4, y_proba4, model4, X_train4 = train_stage4_model(X4, y4)
    fpr4, tpr4, _ = roc_curve(y_test4, y_proba4)
    results['final'] = {
        'y_test': y_test4, 'y_pred': y_pred4, 'y_proba': y_proba4,
        'fpr': fpr4, 'tpr': tpr4
    }
    
    # Plot Stage 4 feature importance
    plot_stage4_feature_importance(model4, X_train4.columns)
    
    # Plot all comparisons
    plot_confusion_matrices(results)
    plot_roc_curves_comparison(results)
    plot_recall_by_delay_size(df_stage4, X_test4, y_test4, y_pred4)
    plot_comparison_metrics(results)
    
    # Save ROC data
    roc_data = {
        'baseline': {
            'fpr': fpr1.tolist(), 'tpr': tpr1.tolist(),
            'auc': float(roc_auc_score(y_test1, y_proba1))
        },
        'leakage': {
            'fpr': fpr2.tolist(), 'tpr': tpr2.tolist(),
            'auc': float(roc_auc_score(y_test2, y_proba2))
        },
        'fast': {
            'fpr': fpr3.tolist(), 'tpr': tpr3.tolist(),
            'auc': float(roc_auc_score(y_test3, y_proba3))
        },
        'final': {
            'fpr': fpr4.tolist(), 'tpr': tpr4.tolist(),
            'auc': float(roc_auc_score(y_test4, y_proba4))
        }
    }
    
    with open('roc_curve_data.json', 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("- delay_distribution.png")
    print("- confusion_matrix_baseline.png")
    print("- confusion_matrix_data_leakage.png") 
    print("- confusion_matrix_fast.png")
    print("- confusion_matrix_final.png")
    print("- roc_curves_comparison.png")
    print("- feature_importance_leakage.png")
    print("- feature_importance_final.png")
    print("- recall_by_delay_size.png")
    print("- comparison_metrics.png")
    print("- roc_curve_data.json")
    
    print("\nThese plots are EXACT reproductions from unified_analysis.ipynb!")

if __name__ == "__main__":
    main()