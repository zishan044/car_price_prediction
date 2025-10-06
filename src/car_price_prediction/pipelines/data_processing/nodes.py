import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

num_sylinders_pattern = r'(?:Flat\s+(\d+)|V(\d+)|(\d+)\s+Cylinder|I(\d+))'
vol_cylinders_pattern = r'(\d+\.?\d*)\s*L'
horse_power_pattern = r'(\d+\.?\d*)\s*HP'

import pandas as pd

def _null_percentage(x: pd.Series) -> float:
        return np.round(x.isna().mean(), 2)

def _impute_fuel_type(fuel_col: pd.Series, eng_class_col: pd.Series) -> pd.Series:

    fuel = fuel_col.copy()
    
    electric_mask = eng_class_col.str.contains('Electric|Hybrid', na=False)
    fuel.loc[electric_mask] = 'Electric'

    remaining_na = fuel.isna()
    if remaining_na.any():
        non_electric_non_na = (~electric_mask) & (fuel.notna())
        most_frequent_fuel = fuel.loc[non_electric_non_na].mode()

        if not most_frequent_fuel.empty:
            fuel.loc[remaining_na] = most_frequent_fuel[0]
        else:
            fuel.loc[remaining_na] = 'Gasoline'
    
    return fuel


def _get_premium_score(df: pd.DataFrame, n_bins: int = 10000) -> pd.Series:
    avg_price = df.groupby(['brand', 'model'])['price'].mean()

    bins = np.linspace(avg_price.min() - 1, avg_price.max() + 1, n_bins)
    scores = pd.cut(avg_price, bins=bins, labels=False)

    score_map = scores.to_dict()
    return df.set_index(['brand', 'model']).index.map(score_map)


def _is_equal(x: pd.Series, s: str) -> pd.Series:
    return x == s

def _keep_top_n(x: pd.Series, n: int) -> pd.Series:
    x_clean = x.replace('–', 'Unknown')
    top_n = x_clean.value_counts().nlargest(n).index
    return x_clean.where(x_clean.isin(top_n), 'Other')

def _extract_value(engine_str, pattern):
    if pd.isna(engine_str):
        return None
    
    engine_str = str(engine_str)
    
    match = re.search(pattern, engine_str, re.IGNORECASE)
    if match:
        for group in match.groups():
            if group is not None:
                return float(group)
    
    return None

def _classify_engine(engine_str):
    
    if pd.isna(engine_str):
        return 'Basic'

    s = str(engine_str).lower()

    # ----- High Performance -----
    if any(x in s for x in ['twin turbo', 'twin-turbo', 'supercharged', 'w12', 'w16', 'rotary']):
        return 'High_Performance'

    # ----- Turbocharged -----
    if 'turbo' in s:
        return 'Turbocharged'

    # ----- Modern Efficient -----
    if any(x in s for x in ['gdi', 'direct injection', 'tsi', 'tfsi', 'dohc']):
        return 'Modern_Efficient'

    # ----- Electric/Hybrid -----
    if any(x in s for x in ['electric', 'plug-in', 'plugin', 'battery', 'mild hybrid', 'hybrid']):
        return 'Electric_Hybrid'

    # ----- Traditional -----
    if any(x in s for x in ['mpfi', 'sohc', 'ohv']):
        return 'Traditional'

    # ----- Special Config -----
    if any(x in s for x in ['flat', 'h4', 'h6']):
        return 'Special_Config'

    return 'Basic'
    
def _extract_transmission_tech(trans):
    if pd.isna(trans):
        return 'Unknown'
    
    trans = str(trans).lower()
    
    # Premium performance transmissions
    if any(term in trans for term in ['dct', 'dual-clutch']):
        return 'Dual_Clutch'
    
    # High-gear modern automatics (8+ speeds)
    elif any(term in trans for term in ['8-speed', '9-speed', '10-speed']):
        return 'High_Gear_Automatic'
    
    # CVT transmissions
    elif 'cvt' in trans:
        return 'CVT'
    
    # Manual transmissions
    elif any(term in trans for term in ['m/t', 'manual']):
        return 'Manual'
    
    # Basic automatics with manual mode
    elif any(term in trans for term in ['dual shift', 'auto-shift']):
        return 'Automatic_Manual_Mode'
    
    # Standard automatics
    elif any(term in trans for term in ['automatic', 'a/t']):
        return 'Standard_Automatic'
    
    # Single-speed (electric vehicles)
    elif any(term in trans for term in ['1-speed', 'single-speed']):
        return 'Electric_Single_Speed'
    
    else:
        return 'Other'
    
def _gear_count_category(trans):
    if pd.isna(trans):
        return 'Unknown'
    
    trans = str(trans).lower()
    
    if any(term in trans for term in ['10-speed']):
        return '10_Speed'
    elif any(term in trans for term in ['8-speed', '9-speed']):
        return '8-9_Speed'
    elif any(term in trans for term in ['6-speed', '7-speed']):
        return '6-7_Speed'
    elif any(term in trans for term in ['4-speed', '5-speed']):
        return '4-5_Speed'
    elif any(term in trans for term in ['1-speed', 'single-speed']):
        return '1_Speed_EV'
    elif 'cvt' in trans:
        return 'CVT_Infinite'
    else:
        return 'Unknown_Gears'
    
def _transmission_modernity(trans):
    if pd.isna(trans):
        return 0
    
    trans = str(trans).lower()
    score = 0
    
    # Premium features
    if any(term in trans for term in ['dct', '10-speed']):
        score += 3
    elif any(term in trans for term in ['8-speed', '9-speed']):
        score += 2
    elif any(term in trans for term in ['dual shift', 'auto-shift']):
        score += 1
    
    # Modern efficient transmissions
    if 'cvt' in trans:
        score += 2
    
    # Electric vehicle transmission
    if any(term in trans for term in ['1-speed', 'single-speed']):
        score += 2
    
    # Older/less advanced
    if any(term in trans for term in ['4-speed', '5-speed']):
        score -= 1
    
    return score

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df['int_col'] = _keep_top_n(df['int_col'], 20)
    df['ext_col'] = _keep_top_n(df['ext_col'], 20)

    df['accident'] = _is_equal(df['accident'], 'At least 1 accident or damage reported')
    df['clean_title'] = _is_equal(df['clean_title'], 'Yes')

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    df['premium_score'] = _get_premium_score(df)

    num_sylinders_pattern = r'(?:Flat\s+(\d+)|V(\d+)|(\d+)\s+Cylinder|I(\d+))'
    df['eng_cylinders'] = df['engine'].apply(lambda x: _extract_value(x, num_sylinders_pattern))

    vol_cylinders_pattern = r'(\d+\.?\d*)\s*L'
    df['eng_volume'] = df['engine'].apply(lambda x: _extract_value(x, vol_cylinders_pattern))

    horse_power_pattern = r'(\d+\.?\d*)\s*HP'
    df['eng_hp'] = df['engine'].apply(lambda x: _extract_value(x, horse_power_pattern))

    df['eng_class'] = df['engine'].apply(_classify_engine)

    df['fuel_type'] = _impute_fuel_type(df['fuel_type'], df['eng_class'])

    df['trans_tech'] = df['transmission'].apply(_extract_transmission_tech)
    df['trans_gear_count_category'] = df['transmission'].apply(_gear_count_category)
    df['trans_mordernity'] = df['transmission'].apply(_transmission_modernity)

    df['col_enc'] = df.groupby(['int_col', 'ext_col'])['price'].transform('mean')

    columns_to_drop = ['id', 'brand', 'model', 'engine', 'transmission', 'int_col', 'ext_col']
    df.drop(columns=columns_to_drop, inplace=True)

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [col for col in df.columns if df[col].dtype != 'object']
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])

    transformed = preprocessor.fit_transform(df)
    features_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=features_names, index=df.index)