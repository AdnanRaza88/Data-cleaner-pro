"""
🧼 DATA CLEANER PRO
👨‍💻 Developer: Adnan Raza
🎓 Data Science Student Project
📚 Universal Data Cleaning Web Application
"""

# ======================
# 📦 REQUIRED LIBRARIES
# ======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
import re
import base64
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import chardet
warnings.filterwarnings('ignore')

# ======================
# 🎨 PAGE CONFIGURATION
# ======================
st.set_page_config(
    page_title="Data Cleaner Pro",
    page_icon="🧼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# 🎭 CUSTOM CSS STYLING
# ======================
st.markdown("""
<style>
    /* Main Container */
    .main {
        padding: 0 !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #2D3748;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4299E1;
        font-weight: 600;
    }
    
    /* Cards */
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #E2E8F0;
        transition: transform 0.3s;
    }
    
    .data-card:hover {
        transform: translateY(-5px);
    }
    
    .issue-card {
        background: linear-gradient(135deg, #FFF5F5, #FED7D7);
        border-left: 6px solid #FC8181;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .suggestion-card {
        background: linear-gradient(135deg, #F0FFF4, #C6F6D5);
        border-left: 6px solid #48BB78;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4299E1, #667EEA);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1);
        background: linear-gradient(90deg, #3182CE, #5A67D8);
    }
    
    .danger-button > button {
        background: linear-gradient(90deg, #F56565, #ED8936) !important;
    }
    
    .success-button > button {
        background: linear-gradient(90deg, #48BB78, #38A169) !important;
    }
    
    /* Metrics */
    .metric-box {
        background: linear-gradient(135deg, #4FD1C5, #319795);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F7FAFC;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    
    /* File Uploader */
    .uploadedFile {
        border: 2px dashed #4299E1 !important;
        border-radius: 10px !important;
        background: #EBF8FF !important;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 10px;
        color: white;
    }
    
    .badge-critical { background-color: #F56565; }
    .badge-high { background-color: #ED8936; }
    .badge-medium { background-color: #ECC94B; }
    .badge-low { background-color: #48BB78; }
    
    /* Progress */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4299E1, #667EEA);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #718096;
        border-top: 1px solid #E2E8F0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# 🏗️ SESSION STATE SETUP
# ======================
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'cleaning_history' not in st.session_state:
    st.session_state.cleaning_history = []
if 'issues_found' not in st.session_state:
    st.session_state.issues_found = []
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = {}

# ======================
# 🎯 LOGO CONFIGURATION
# ======================
LOGO_URL = "https://i.ibb.co/mVvJQ2D3/logo-adnan.png"

# ======================
# 📁 UNIVERSAL FILE READING
# ======================
def detect_encoding(file_bytes):
    """Detect file encoding automatically"""
    result = chardet.detect(file_bytes)
    return result['encoding'] if result['encoding'] else 'utf-8'

def read_csv_file(uploaded_file):
    """Read CSV with auto-encoding detection"""
    try:
        file_bytes = uploaded_file.read()
        encoding = detect_encoding(file_bytes)
        uploaded_file.seek(0)
        
        # Try different delimiters
        delimiters = [',', ';', '\t', '|', ' ']
        
        for delimiter in delimiters:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, delimiter=delimiter, engine='python')
                if len(df.columns) > 1:
                    return df
            except:
                continue
        
        # Final attempt with comma
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding=encoding)
    except Exception as e:
        st.error(f"CSV reading error: {str(e)}")
        return None

def read_excel_file(uploaded_file):
    """Read Excel with multiple sheets support"""
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        
        if len(sheet_names) > 1:
            selected_sheet = st.selectbox("Select sheet:", sheet_names)
            return pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Excel reading error: {str(e)}")
        return None

def read_json_file(uploaded_file):
    """Read JSON with flexible structure"""
    try:
        content = uploaded_file.read().decode('utf-8')
        data = json.loads(content)
        
        if isinstance(data, dict):
            # Flatten nested dictionaries
            df = pd.json_normalize(data, sep='_')
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({'data': [str(data)]})
        
        return df
    except Exception as e:
        st.error(f"JSON reading error: {str(e)}")
        return None

def read_text_file(uploaded_file):
    """Read any text file and extract structured data"""
    try:
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        # Try to detect structure
        if any(',' in line for line in lines[:5]):
            return pd.read_csv(io.StringIO(content))
        elif any('\t' in line for line in lines[:5]):
            return pd.read_csv(io.StringIO(content), sep='\t')
        elif any(';' in line for line in lines[:5]):
            return pd.read_csv(io.StringIO(content), sep=';')
        else:
            return pd.DataFrame({'Text': lines})
    except Exception as e:
        st.error(f"Text file error: {str(e)}")
        return None

def read_file(uploaded_file):
    """Universal file reader"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner(f"Reading {file_extension.upper()} file..."):
        if file_extension == 'csv':
            return read_csv_file(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            return read_excel_file(uploaded_file)
        elif file_extension == 'json':
            return read_json_file(uploaded_file)
        elif file_extension in ['txt', 'tsv']:
            return read_text_file(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None

# ======================
# 🔍 COMPREHENSIVE DATA ANALYSIS
# ======================
def analyze_dataset(df):
    """Comprehensive dataset analysis"""
    analysis = {
        'basic_info': {},
        'column_analysis': {},
        'quality_metrics': {}
    }
    
    # Basic Info
    analysis['basic_info']['shape'] = df.shape
    analysis['basic_info']['memory_usage'] = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
    analysis['basic_info']['total_cells'] = df.size
    analysis['basic_info']['total_missing'] = df.isnull().sum().sum()
    analysis['basic_info']['missing_percentage'] = (analysis['basic_info']['total_missing'] / analysis['basic_info']['total_cells']) * 100
    
    # Column Analysis
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'missing_count': int(df[col].isnull().sum()),
            'missing_percent': round((df[col].isnull().sum() / len(df)) * 100, 2),
            'sample_values': df[col].dropna().head(5).tolist()
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'zeros_count': int((df[col] == 0).sum()),
                'negative_count': int((df[col] < 0).sum())
            })
            
            # Skewness and Kurtosis
            if len(df[col].dropna()) > 3:
                col_info['skewness'] = float(df[col].skew())
                col_info['kurtosis'] = float(df[col].kurtosis())
        
        # Text columns
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            col_info.update({
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_common_count': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
                'avg_length': float(df[col].astype(str).str.len().mean()),
                'max_length': int(df[col].astype(str).str.len().max()),
                'empty_strings': int((df[col].astype(str) == '').sum())
            })
        
        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_info.update({
                'min_date': df[col].min(),
                'max_date': df[col].max(),
                'date_range_days': (df[col].max() - df[col].min()).days if not df[col].isnull().all() else None
            })
        
        analysis['column_analysis'][col] = col_info
    
    # Quality Metrics
    analysis['quality_metrics']['duplicate_rows'] = int(df.duplicated().sum())
    analysis['quality_metrics']['duplicate_percentage'] = round((analysis['quality_metrics']['duplicate_rows'] / len(df)) * 100, 2) if len(df) > 0 else 0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    analysis['quality_metrics']['numeric_columns'] = len(numeric_cols)
    analysis['quality_metrics']['text_columns'] = len(df.select_dtypes(include=['object']).columns)
    analysis['quality_metrics']['datetime_columns'] = len(df.select_dtypes(include=['datetime']).columns)
    
    return analysis

def detect_all_issues(df):
    """Detect all types of data quality issues"""
    issues = []
    
    # 1. Missing Values Analysis
    missing_cols = df.columns[df.isnull().any()]
    for col in missing_cols:
        missing_count = int(df[col].isnull().sum())
        missing_percent = round((missing_count / len(df)) * 100, 2)
        
        severity = 'CRITICAL' if missing_percent > 30 else 'HIGH' if missing_percent > 10 else 'MEDIUM' if missing_percent > 5 else 'LOW'
        
        issues.append({
            'type': 'MISSING_VALUES',
            'column': col,
            'severity': severity,
            'message': f'{missing_count} missing values ({missing_percent}%)',
            'suggestion': f'Use {"column removal" if missing_percent > 50 else "imputation"}',
            'rows_affected': list(df[df[col].isnull()].index[:10]),
            'fix_function': 'handle_missing_values'
        })
    
    # 2. Duplicate Rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        duplicate_percent = round((duplicate_count / len(df)) * 100, 2)
        severity = 'HIGH' if duplicate_percent > 10 else 'MEDIUM' if duplicate_percent > 5 else 'LOW'
        
        issues.append({
            'type': 'DUPLICATE_ROWS',
            'column': 'ALL',
            'severity': severity,
            'message': f'{duplicate_count} duplicate rows ({duplicate_percent}%)',
            'suggestion': 'Remove duplicate rows',
            'rows_affected': list(df[df.duplicated()].index[:10]),
            'fix_function': 'remove_duplicates'
        })
    
    # 3. Data Type Inconsistencies
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check for mixed types
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                types = set(type(x).__name__ for x in sample)
                if len(types) > 1:
                    issues.append({
                        'type': 'MIXED_DATA_TYPES',
                        'column': col,
                        'severity': 'HIGH',
                        'message': f'Mixed types detected: {list(types)[:3]}',
                        'suggestion': 'Convert to consistent data type',
                        'rows_affected': list(df.index[:5]),
                        'fix_function': 'convert_data_types'
                    })
    
    # 4. Text Format Issues
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        # Inconsistent capitalization
        sample = df[col].dropna().head(50)
        if len(sample) > 0:
            is_title = sample.str.istitle()
            is_upper = sample.str.isupper()
            is_lower = sample.str.islower()
            is_proper = sample.str[0].str.isupper() & sample.str[1:].str.islower()
            
            if not (is_title.all() or is_upper.all() or is_lower.all() or is_proper.all()):
                issues.append({
                    'type': 'INCONSISTENT_CAPITALIZATION',
                    'column': col,
                    'severity': 'LOW',
                    'message': 'Inconsistent text formatting',
                    'suggestion': 'Standardize text case',
                    'rows_affected': list(df.index[:5]),
                    'fix_function': 'standardize_text'
                })
        
        # Extra whitespace
        if df[col].astype(str).str.contains(r'\s{2,}', na=False).any():
            issues.append({
                'type': 'EXTRA_WHITESPACE',
                'column': col,
                'severity': 'LOW',
                'message': 'Contains extra spaces/tabs',
                'suggestion': 'Trim extra whitespace',
                'rows_affected': list(df[df[col].astype(str).str.contains(r'\s{2,}', na=False)].index[:5]),
                'fix_function': 'clean_whitespace'
            })
        
        # Special characters
        if df[col].astype(str).str.contains(r'[^\w\s.,!?-]', na=False).any():
            issues.append({
                'type': 'SPECIAL_CHARACTERS',
                'column': col,
                'severity': 'LOW',
                'message': 'Contains unusual special characters',
                'suggestion': 'Remove or replace special characters',
                'rows_affected': list(df[df[col].astype(str).str.contains(r'[^\w\s.,!?-]', na=False)].index[:5]),
                'fix_function': 'clean_special_chars'
            })
    
    # 5. Outliers in Numeric Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
        if len(df[col].dropna()) > 10:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_percent = round((outliers / len(df)) * 100, 2)
                severity = 'HIGH' if outlier_percent > 10 else 'MEDIUM' if outlier_percent > 5 else 'LOW'
                
                issues.append({
                    'type': 'OUTLIERS',
                    'column': col,
                    'severity': severity,
                    'message': f'{outliers} outliers detected ({outlier_percent}%)',
                    'suggestion': 'Cap, remove, or investigate outliers',
                    'rows_affected': list(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index[:5]),
                    'fix_function': 'handle_outliers'
                })
    
    # 6. Zero Variance Columns
    for col in numeric_cols:
        if df[col].nunique() == 1:
            issues.append({
                'type': 'ZERO_VARIANCE',
                'column': col,
                'severity': 'MEDIUM',
                'message': 'Column has only one unique value',
                'suggestion': 'Consider removing this column',
                'rows_affected': 'ALL',
                'fix_function': 'remove_column'
            })
    
    # 7. High Cardinality (for categorical)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9 and len(df) > 100:
            issues.append({
                'type': 'HIGH_CARDINALITY',
                'column': col,
                'severity': 'MEDIUM',
                'message': f'High unique values ({df[col].nunique()})',
                'suggestion': 'Consider encoding or grouping',
                'rows_affected': 'ALL',
                'fix_function': 'reduce_cardinality'
            })
    
    return issues

# ======================
# 🛠️ CLEANING FUNCTIONS
# ======================
def handle_missing_values(df, column, method='mean', custom_value=None):
    """Handle missing values with multiple methods"""
    df_clean = df.copy()
    
    if method == 'remove_rows':
        df_clean = df_clean.dropna(subset=[column])
    elif method == 'remove_column':
        df_clean = df_clean.drop(columns=[column])
    elif method == 'mean' and pd.api.types.is_numeric_dtype(df_clean[column]):
        df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
    elif method == 'median' and pd.api.types.is_numeric_dtype(df_clean[column]):
        df_clean[column] = df_clean[column].fillna(df_clean[column].median())
    elif method == 'mode':
        mode_val = df_clean[column].mode()
        df_clean[column] = df_clean[column].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
    elif method == 'forward_fill':
        df_clean[column] = df_clean[column].ffill()
    elif method == 'backward_fill':
        df_clean[column] = df_clean[column].bfill()
    elif method == 'interpolate':
        df_clean[column] = df_clean[column].interpolate()
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    elif method == 'custom' and custom_value is not None:
        df_clean[column] = df_clean[column].fillna(custom_value)
    
    return df_clean

def remove_duplicates(df, keep='first', subset=None):
    """Remove duplicate rows"""
    df_clean = df.copy()
    if subset:
        df_clean = df_clean.drop_duplicates(subset=subset, keep=keep)
    else:
        df_clean = df_clean.drop_duplicates(keep=keep)
    return df_clean

def standardize_text(df, column, case='title'):
    """Standardize text case"""
    df_clean = df.copy()
    
    if case == 'title':
        df_clean[column] = df_clean[column].astype(str).str.title()
    elif case == 'upper':
        df_clean[column] = df_clean[column].astype(str).str.upper()
    elif case == 'lower':
        df_clean[column] = df_clean[column].astype(str).str.lower()
    elif case == 'proper':
        df_clean[column] = df_clean[column].astype(str).str.capitalize()
    
    return df_clean

def clean_whitespace(df, column):
    """Remove extra whitespace"""
    df_clean = df.copy()
    df_clean[column] = df_clean[column].astype(str).str.strip()
    df_clean[column] = df_clean[column].replace(r'\s+', ' ', regex=True)
    return df_clean

def clean_special_chars(df, column, replace_with=''):
    """Remove special characters"""
    df_clean = df.copy()
    df_clean[column] = df_clean[column].astype(str).str.replace(r'[^\w\s.,!?-]', replace_with, regex=True)
    return df_clean

def convert_data_types(df, column, new_type):
    """Convert column data type"""
    df_clean = df.copy()
    
    try:
        if new_type == 'int':
            df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce').astype('Int64')
        elif new_type == 'float':
            df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')
        elif new_type == 'string':
            df_clean[column] = df_clean[column].astype(str)
        elif new_type == 'category':
            df_clean[column] = df_clean[column].astype('category')
        elif new_type == 'datetime':
            df_clean[column] = pd.to_datetime(df_clean[column], errors='coerce')
        elif new_type == 'boolean':
            df_clean[column] = df_clean[column].astype(bool)
    except Exception as e:
        st.error(f"Conversion error: {str(e)}")
    
    return df_clean

def handle_outliers(df, column, method='cap', threshold=3):
    """Handle outliers using different methods"""
    df_clean = df.copy()
    
    if pd.api.types.is_numeric_dtype(df_clean[column]):
        if method == 'cap':
            # Cap at IQR bounds
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'remove':
            # Remove outliers
            Q1 = df_clean[column].quantile(0.25)
            Q3 = df_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        
        elif method == 'log':
            # Apply log transformation
            df_clean[column] = np.log1p(df_clean[column])
    
    return df_clean

def encode_categorical(df, columns=None, method='label'):
    """Encode categorical variables"""
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=['object', 'category']).columns
    
    for col in columns:
        if df_clean[col].nunique() < 100:  # Don't encode high cardinality
            if method == 'label':
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            elif method == 'onehot':
                # Simple one-hot for demo
                dummies = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
                df_clean = pd.concat([df_clean, dummies], axis=1)
                df_clean = df_clean.drop(columns=[col])
    
    return df_clean

def normalize_numeric(df, columns=None, method='standard'):
    """Normalize numeric columns"""
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
        df_clean[columns] = scaler.fit_transform(df_clean[columns])
    elif method == 'minmax':
        for col in columns:
            min_val = df_clean[col].min()
            max_val = df_clean[col].max()
            if max_val > min_val:
                df_clean[col] = (df_clean[col] - min_val) / (max_val - min_val)
    
    return df_clean

# ======================
# 📊 VISUALIZATION FUNCTIONS
# ======================
def plot_missing_values(df):
    """Create missing values visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    missing_matrix = df.isnull()
    axes[0].imshow(missing_matrix.T, aspect='auto', cmap='Reds', interpolation='nearest')
    axes[0].set_xlabel('Rows')
    axes[0].set_ylabel('Columns')
    axes[0].set_title('Missing Values Heatmap', fontweight='bold', fontsize=12)
    axes[0].set_yticks(range(len(df.columns)))
    axes[0].set_yticklabels(df.columns, fontsize=8)
    
    # Bar chart
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) > 0:
        colors = ['#F56565' if x > 0.3*len(df) else '#ED8936' if x > 0.1*len(df) else '#ECC94B' for x in missing_counts]
        axes[1].barh(range(len(missing_counts)), missing_counts.values, color=colors)
        axes[1].set_yticks(range(len(missing_counts)))
        axes[1].set_yticklabels(missing_counts.index, fontsize=9)
        axes[1].set_xlabel('Missing Count')
        axes[1].set_title('Missing Values by Column', fontweight='bold', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='x')
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values!', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#C6F6D5', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_data_distributions(df):
    """Create distribution plots for columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return None
    
    # Calculate grid size
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:len(axes)]):
        ax = axes[idx]
        
        # Histogram with KDE
        data = df[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='#4299E1', density=True)
            
            # Add KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax.plot(x_range, kde(x_range), color='#1E3A8A', linewidth=2)
            except:
                pass
            
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Distribution: {col}', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df):
    """Create correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(correlation, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=True, fmt='.2f', ax=ax)
    
    ax.set_title('Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    return fig

def plot_boxplots(df):
    """Create box plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:len(axes)]):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) > 0:
            bp = ax.boxplot(data, patch_artist=True)
            bp['boxes'][0].set_facecolor('#4299E1')
            bp['boxes'][0].set_alpha(0.7)
            bp['medians'][0].set_color('#1E3A8A')
            
            ax.set_ylabel(col, fontsize=10)
            ax.set_title(f'Box Plot: {col}', fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')
    
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

# ======================
# 💾 EXPORT FUNCTIONS
# ======================
def export_to_csv(df):
    """Export DataFrame to CSV"""
    return df.to_csv(index=False)

def export_to_excel(df):
    """Export DataFrame to Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
    return output.getvalue()

def export_to_json(df, orient='records'):
    """Export DataFrame to JSON"""
    return df.to_json(orient=orient, indent=2)

def export_to_parquet(df):
    """Export DataFrame to Parquet"""
    output = io.BytesIO()
    df.to_parquet(output, index=False)
    return output.getvalue()

def export_to_html(df):
    """Export DataFrame to HTML"""
    return df.to_html(index=False, classes='table table-striped', border=0)

def get_download_link(data, filename, text):
    """Generate download link"""
    b64 = base64.b64encode(data).decode() if isinstance(data, bytes) else base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# ======================
# 🚀 MAIN APPLICATION
# ======================
def main():
    # ====================
    # 🎨 HEADER SECTION
    # ====================
    st.markdown('<h1 class="main-header">🧼 Data Cleaner Pro</h1>', unsafe_allow_html=True)
    
    # Add logo
    try:
        st.logo(LOGO_URL)
    except:
        pass
    
    # Hero section
    st.markdown("""
    <div class="data-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h2 style="margin: 0; color: #2D3748;">Universal Data Cleaning Solution</h2>
                <p style="color: #4A5568; margin-top: 0.5rem;">
                    Upload any dataset (CSV, Excel, JSON, Text) → Clean it interactively → Download in any format
                </p>
            </div>
            <div style="text-align: center;">
                <h3 style="margin: 0; color: #4299E1;">👨‍💻 Adnan Raza</h3>
                <p style="color: #718096; margin: 0;">🎓 Data Science Student</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ====================
    # 📁 SIDEBAR - FILE UPLOAD
    # ====================
    with st.sidebar:
        st.markdown("## 📁 Upload Data")
        
        uploaded_file = st.file_uploader(
            "Choose any data file",
            type=['csv', 'xlsx', 'xls', 'json', 'txt', 'tsv'],
            help="Supported formats: CSV, Excel, JSON, Text, TSV"
        )
        
        if uploaded_file is not None:
            df = read_file(uploaded_file)
            
            if df is not None:
                st.session_state.original_data = df.copy()
                st.session_state.current_data = df.copy()
                st.session_state.file_type = uploaded_file.name.split('.')[-1].lower()
                st.session_state.cleaning_history = []
                st.session_state.analysis_report = analyze_dataset(df)
                st.session_state.issues_found = detect_all_issues(df)
                
                st.success(f"✅ **{uploaded_file.name}** loaded successfully!")
                st.info(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        st.markdown("---")
        st.markdown("### ⚙️ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                if st.session_state.original_data is not None:
                    st.session_state.current_data = st.session_state.original_data.copy()
                    st.session_state.cleaning_history = []
                    st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                    st.rerun()
        
        with col2:
            if st.button("✨ Auto Clean", use_container_width=True):
                if st.session_state.current_data is not None:
                    df = st.session_state.current_data.copy()
                    cleaning_log = []
                    
                    # Auto cleaning steps
                    initial_rows = len(df)
                    df = remove_duplicates(df)
                    removed_duplicates = initial_rows - len(df)
                    if removed_duplicates > 0:
                        cleaning_log.append(f"Removed {removed_duplicates} duplicate rows")
                    
                    # Fill missing values
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if df[col].isnull().sum() > 0:
                            df[col] = df[col].fillna(df[col].median())
                            cleaning_log.append(f"Filled missing values in {col} with median")
                    
                    # Clean text columns
                    text_cols = df.select_dtypes(include=['object']).columns
                    for col in text_cols:
                        df[col] = df[col].astype(str).str.strip()
                        df[col] = df[col].replace(r'\s+', ' ', regex=True)
                        cleaning_log.append(f"Cleaned whitespace in {col}")
                    
                    st.session_state.current_data = df
                    st.session_state.cleaning_history.extend(cleaning_log)
                    st.session_state.issues_found = detect_all_issues(df)
                    st.rerun()
        
        st.markdown("---")
        
        # Show cleaning history in sidebar
        if st.session_state.cleaning_history:
            st.markdown("### 📜 Recent Actions")
            for action in st.session_state.cleaning_history[-5:]:
                st.markdown(f"• {action}")
    
    # ====================
    # 📊 MAIN CONTENT AREA
    # ====================
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        analysis = st.session_state.analysis_report
        
        # METRICS DASHBOARD
        st.markdown("## 📈 Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{analysis['basic_info']['shape'][0]:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{analysis['basic_info']['shape'][1]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_color = "#48BB78" if analysis['basic_info']['missing_percentage'] < 5 else "#ECC94B" if analysis['basic_info']['missing_percentage'] < 20 else "#F56565"
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, {missing_color}, #319795);">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value">{analysis['basic_info']['total_missing']:,}</div>
                <div class="metric-label">({analysis['basic_info']['missing_percentage']:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Duplicate Rows</div>
                <div class="metric-value">{analysis['quality_metrics']['duplicate_rows']:,}</div>
                <div class="metric-label">({analysis['quality_metrics']['duplicate_percentage']:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value">{analysis['basic_info']['memory_usage']:.1f}</div>
                <div class="metric-label">MB</div>
            </div>
            """, unsafe_allow_html=True)
        
        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Preview", 
            "🔍 Issues Analysis", 
            "🛠️ Cleaning Tools", 
            "📊 Visualizations", 
            "💾 Export Data"
        ])
        
        # TAB 1: DATA PREVIEW
        with tab1:
            col_left, col_right = st.columns([3, 1])
            
            with col_left:
                st.markdown("### Data Preview")
                
                # Pagination
                page_size = st.slider("Rows per page:", 10, 100, 50, key="preview_pagesize")
                total_pages = max(1, (len(df) // page_size) + (1 if len(df) % page_size > 0 else 0))
                
                if total_pages > 1:
                    page_number = st.number_input("Page:", 1, total_pages, 1, key="preview_page")
                    start_idx = (page_number - 1) * page_size
                    end_idx = min(start_idx + page_size, len(df))
                else:
                    start_idx, end_idx = 0, min(page_size, len(df))
                
                # Display dataframe with highlighting
                preview_df = df.iloc[start_idx:end_idx]
                styled_df = preview_df.style.apply(
                    lambda x: ['background: #FED7D7' if pd.isnull(v) else '' for v in x], 
                    axis=0
                )
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                if total_pages > 1:
                    st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {len(df)} (Page {page_number} of {total_pages})")
            
            with col_right:
                st.markdown("### 🔍 Data Types")
                
                dtype_counts = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                
                for dtype, count in dtype_counts.items():
                    st.markdown(f"**{dtype}**: {count} columns")
                
                st.markdown("---")
                st.markdown("### 📝 Column Summary")
                
                selected_col = st.selectbox("Select column:", df.columns, key="col_summary")
                col_info = analysis['column_analysis'].get(selected_col, {})
                
                if col_info:
                    st.markdown(f"""
                    **Type:** `{col_info.get('dtype', 'Unknown')}`  
                    **Unique Values:** {col_info.get('unique_count', 0):,}  
                    **Missing Values:** {col_info.get('missing_count', 0):,} ({col_info.get('missing_percent', 0):.1f}%)  
                    **Sample:** {', '.join(map(str, col_info.get('sample_values', [])[:3]))}
                    """)
                    
                    if 'mean' in col_info:
                        st.markdown(f"""
                        **Stats:**  
                        Min: {col_info.get('min', 'N/A'):.2f}  
                        Max: {col_info.get('max', 'N/A'):.2f}  
                        Mean: {col_info.get('mean', 'N/A'):.2f}  
                        Std: {col_info.get('std', 'N/A'):.2f}
                        """)
        
        # TAB 2: ISSUES ANALYSIS
        with tab2:
            if st.session_state.issues_found:
                # Group issues by severity
                severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                grouped_issues = {severity: [] for severity in severity_order}
                
                for issue in st.session_state.issues_found:
                    grouped_issues[issue['severity']].append(issue)
                
                # Display by severity
                severity_colors = {
                    'CRITICAL': '#F56565',
                    'HIGH': '#ED8936',
                    'MEDIUM': '#ECC94B',
                    'LOW': '#48BB78'
                }
                
                for severity in severity_order:
                    issues = grouped_issues[severity]
                    if issues:
                        st.markdown(f"### {severity} Priority ({len(issues)})")
                        
                        for issue in issues:
                            with st.expander(f"{issue['type'].replace('_', ' ').title()} in `{issue['column']}`", expanded=True):
                                st.markdown(f"""
                                **📝 Issue:** {issue['message']}
                                
                                **💡 Suggestion:** {issue['suggestion']}
                                
                                **📍 Affected Rows:** {', '.join(map(str, issue['rows_affected'][:3])) if isinstance(issue['rows_affected'], list) else issue['rows_affected']}
                                """)
                                
                                # Quick fix buttons
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button(f"Fix this issue", key=f"fix_{issue['type']}_{issue['column']}"):
                                        if issue['type'] == 'MISSING_VALUES':
                                            st.session_state.current_data = handle_missing_values(
                                                df, issue['column'], 'median'
                                            )
                                        elif issue['type'] == 'DUPLICATE_ROWS':
                                            st.session_state.current_data = remove_duplicates(df)
                                        elif issue['type'] == 'INCONSISTENT_CAPITALIZATION':
                                            st.session_state.current_data = standardize_text(df, issue['column'], 'title')
                                        
                                        st.session_state.cleaning_history.append(
                                            f"Fixed {issue['type'].lower().replace('_', ' ')} in {issue['column']}"
                                        )
                                        st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                                        st.rerun()
                                
                                with col2:
                                    if st.button(f"Ignore issue", key=f"ignore_{issue['type']}_{issue['column']}"):
                                        st.session_state.issues_found = [
                                            i for i in st.session_state.issues_found 
                                            if not (i['type'] == issue['type'] and i['column'] == issue['column'])
                                        ]
                                        st.rerun()
            else:
                st.success("""
                ## 🎉 Perfect Data Quality!
                
                No issues detected in your dataset. Your data is clean and ready for analysis!
                """)
                st.balloons()
        
        # TAB 3: CLEANING TOOLS
        with tab3:
            st.markdown("## 🛠️ Interactive Cleaning Tools")
            
            cleaning_col1, cleaning_col2 = st.columns(2)
            
            with cleaning_col1:
                st.markdown("### 🔧 Column Operations")
                
                selected_column = st.selectbox("Select Column:", df.columns, key="clean_col_select")
                
                operation = st.selectbox(
                    "Choose Operation:",
                    [
                        "Fix Missing Values",
                        "Convert Data Type",
                        "Clean Text Format",
                        "Handle Outliers",
                        "Encode Categorical",
                        "Normalize Values",
                        "Remove Column",
                        "Rename Column"
                    ],
                    key="clean_operation"
                )
                
                if operation == "Fix Missing Values":
                    method = st.selectbox(
                        "Imputation Method:",
                        ["mean", "median", "mode", "forward_fill", "backward_fill", "remove_rows", "remove_column"],
                        key="missing_method"
                    )
                    
                    if st.button("Apply Imputation", key="apply_imputation"):
                        st.session_state.current_data = handle_missing_values(df, selected_column, method)
                        st.session_state.cleaning_history.append(
                            f"Fixed missing values in '{selected_column}' using {method}"
                        )
                        st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                        st.rerun()
                
                elif operation == "Convert Data Type":
                    new_type = st.selectbox(
                        "Convert to:",
                        ["int", "float", "string", "category", "datetime", "boolean"],
                        key="convert_type"
                    )
                    
                    if st.button("Convert Data Type", key="apply_convert"):
                        st.session_state.current_data = convert_data_types(df, selected_column, new_type)
                        st.session_state.cleaning_history.append(
                            f"Converted '{selected_column}' to {new_type}"
                        )
                        st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                        st.rerun()
                
                elif operation == "Clean Text Format":
                    text_op = st.selectbox(
                        "Text Operation:",
                        ["title_case", "uppercase", "lowercase", "strip_whitespace", "remove_special_chars"],
                        key="text_operation"
                    )
                    
                    if st.button("Clean Text", key="apply_text_clean"):
                        if text_op in ['title_case', 'uppercase', 'lowercase']:
                            case = text_op.split('_')[0] if '_' in text_op else text_op
                            st.session_state.current_data = standardize_text(df, selected_column, case)
                        elif text_op == 'strip_whitespace':
                            st.session_state.current_data = clean_whitespace(df, selected_column)
                        elif text_op == 'remove_special_chars':
                            st.session_state.current_data = clean_special_chars(df, selected_column)
                        
                        st.session_state.cleaning_history.append(
                            f"Cleaned text in '{selected_column}' using {text_op}"
                        )
                        st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                        st.rerun()
                
                elif operation == "Handle Outliers":
                    outlier_method = st.selectbox(
                        "Outlier Treatment:",
                        ["cap", "remove", "log"],
                        key="outlier_method"
                    )
                    
                    if st.button("Handle Outliers", key="apply_outliers"):
                        st.session_state.current_data = handle_outliers(df, selected_column, outlier_method)
                        st.session_state.cleaning_history.append(
                            f"Handled outliers in '{selected_column}' using {outlier_method}"
                        )
                        st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                        st.rerun()
                
                elif operation == "Remove Column":
                    if st.button(f"Remove '{selected_column}'", type="secondary", key="remove_col"):
                        st.session_state.current_data = df.drop(columns=[selected_column])
                        st.session_state.cleaning_history.append(
                            f"Removed column '{selected_column}'"
                        )
                        st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                        st.rerun()
            
            with cleaning_col2:
                st.markdown("### 📝 Row Operations")
                
                # Remove duplicates
                duplicate_count = df.duplicated().sum()
                st.markdown(f"**Duplicate Rows:** {duplicate_count:,}")
                
                if duplicate_count > 0:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Remove Duplicates", key="remove_dups"):
                            st.session_state.current_data = remove_duplicates(df)
                            st.session_state.cleaning_history.append(
                                f"Removed {duplicate_count} duplicate rows"
                            )
                            st.session_state.issues_found = detect_all_issues(st.session_state.current_data)
                            st.rerun()
                    
                    with col_b:
                        subset_cols = st.multiselect(
                            "Check duplicates in:",
                            df.columns,
                            key="dup_subset"
                        )
                        if subset_cols and st.button("Check Subset Duplicates"):
                            subset_dups = df.duplicated(subset=subset_cols).sum()
                            st.info(f"Duplicates in selected columns: {subset_dups:,}")
                
                # Filter rows
                st.markdown("---")
                st.markdown("#### 🎯 Filter Rows")
                
                filter_col = st.selectbox("Filter by column:", df.columns, key="filter_col")
                if filter_col:
                    unique_vals = df[filter_col].dropna().unique()
                    if len(unique_vals) < 20:
                        selected_vals = st.multiselect(
                            "Keep rows with values:",
                            unique_vals,
                            key="filter_vals"
                        )
                        if selected_vals and st.button("Apply Filter"):
                            st.session_state.current_data = df[df[filter_col].isin(selected_vals)]
                            st.session_state.cleaning_history.append(
                                f"Filtered '{filter_col}' for {len(selected_vals)} values"
                            )
                            st.rerun()
                    else:
                        # For columns with many unique values
                        filter_type = st.selectbox(
                            "Filter type:",
                            ["contains", "starts with", "ends with", "greater than", "less than"],
                            key="filter_type"
                        )
                        
                        filter_value = st.text_input("Filter value:", key="filter_value")
                        
                        if filter_value and st.button("Apply Text Filter"):
                            if filter_type == "contains":
                                mask = df[filter_col].astype(str).str.contains(filter_value, na=False)
                            elif filter_type == "starts with":
                                mask = df[filter_col].astype(str).str.startswith(filter_value, na=False)
                            elif filter_type == "ends with":
                                mask = df[filter_col].astype(str).str.endswith(filter_value, na=False)
                            
                            st.session_state.current_data = df[mask]
                            st.session_state.cleaning_history.append(
                                f"Filtered '{filter_col}' {filter_type} '{filter_value}'"
                            )
                            st.rerun()
                
                # Advanced operations
                st.markdown("---")
                st.markdown("#### ⚙️ Advanced Operations")
                
                if st.button("Encode All Categorical", key="encode_all"):
                    st.session_state.current_data = encode_categorical(df)
                    st.session_state.cleaning_history.append(
                        "Encoded all categorical variables"
                    )
                    st.rerun()
                
                if st.button("Normalize All Numeric", key="normalize_all"):
                    st.session_state.current_data = normalize_numeric(df)
                    st.session_state.cleaning_history.append(
                        "Normalized all numeric columns"
                    )
                    st.rerun()
        
        # TAB 4: VISUALIZATIONS
        with tab4:
            st.markdown("## 📊 Data Visualizations")
            
            viz_type = st.selectbox(
                "Select Visualization:",
                [
                    "Missing Values Analysis",
                    "Data Distributions",
                    "Correlation Matrix",
                    "Box Plots",
                    "Custom Plot"
                ],
                key="viz_type"
            )
            
            if viz_type == "Missing Values Analysis":
                fig = plot_missing_values(df)
                if fig:
                    st.pyplot(fig)
                    
                    # Additional missing stats
                    missing_by_col = df.isnull().sum()
                    missing_by_col = missing_by_col[missing_by_col > 0]
                    
                    if len(missing_by_col) > 0:
                        st.markdown("### 📈 Missing Values Statistics")
                        missing_df = pd.DataFrame({
                            'Column': missing_by_col.index,
                            'Missing Count': missing_by_col.values,
                            'Missing %': (missing_by_col.values / len(df) * 100).round(2)
                        }).sort_values('Missing Count', ascending=False)
                        
                        st.dataframe(missing_df, use_container_width=True)
            
            elif viz_type == "Data Distributions":
                fig = plot_data_distributions(df)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns found for distribution plots.")
            
            elif viz_type == "Correlation Matrix":
                fig = plot_correlation_matrix(df)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numeric columns for correlation matrix.")
            
            elif viz_type == "Box Plots":
                fig = plot_boxplots(df)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("No numeric columns found for box plots.")
            
            elif viz_type == "Custom Plot":
                st.markdown("### Create Custom Visualization")
                
                plot_type = st.selectbox(
                    "Plot Type:",
                    ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Pie Chart"],
                    key="custom_plot_type"
                )
                
                if plot_type in ["Scatter Plot", "Line Plot"]:
                    col_x = st.selectbox("X-axis:", df.columns, key="custom_x")
                    col_y = st.selectbox("Y-axis:", df.columns, key="custom_y")
                    
                    if col_x and col_y and st.button("Generate Plot", key="gen_custom"):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if plot_type == "Scatter Plot":
                            ax.scatter(df[col_x], df[col_y], alpha=0.6, color='#4299E1')
                            ax.set_xlabel(col_x)
                            ax.set_ylabel(col_y)
                            ax.set_title(f'Scatter Plot: {col_x} vs {col_y}', fontweight='bold')
                        
                        elif plot_type == "Line Plot":
                            if pd.api.types.is_numeric_dtype(df[col_x]):
                                sorted_df = df.sort_values(col_x)
                                ax.plot(sorted_df[col_x], sorted_df[col_y], marker='o', color='#4299E1')
                            else:
                                ax.plot(df[col_x], df[col_y], marker='o', color='#4299E1')
                            ax.set_xlabel(col_x)
                            ax.set_ylabel(col_y)
                            ax.set_title(f'Line Plot: {col_x} vs {col_y}', fontweight='bold')
                        
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                
                elif plot_type == "Bar Chart":
                    bar_col = st.selectbox("Category Column:", df.columns, key="bar_col")
                    value_col = st.selectbox("Value Column:", df.select_dtypes(include=[np.number]).columns.tolist() if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns, key="value_col")
                    
                    if bar_col and value_col and st.button("Generate Bar Chart", key="gen_bar"):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        if df[bar_col].nunique() < 20:
                            bar_data = df.groupby(bar_col)[value_col].mean().sort_values(ascending=False)
                            colors = plt.cm.viridis(np.linspace(0, 1, len(bar_data)))
                            ax.bar(bar_data.index.astype(str), bar_data.values, color=colors)
                            ax.set_xlabel(bar_col)
                            ax.set_ylabel(f'Mean of {value_col}')
                            ax.set_title(f'Bar Chart: {bar_col} vs {value_col}', fontweight='bold')
                            plt.xticks(rotation=45)
                        else:
                            st.warning(f"Column '{bar_col}' has too many unique values ({df[bar_col].nunique()}) for a bar chart.")
                        
                        ax.grid(True, alpha=0.3, axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                elif plot_type == "Histogram":
                    hist_col = st.selectbox("Select Column:", df.select_dtypes(include=[np.number]).columns.tolist() if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns, key="hist_col")
                    
                    if hist_col and st.button("Generate Histogram", key="gen_hist"):
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.hist(df[hist_col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='#4299E1')
                        ax.set_xlabel(hist_col)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Histogram of {hist_col}', fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
        
        # TAB 5: EXPORT DATA
        with tab5:
            st.markdown("## 💾 Export Cleaned Data")
            
            # Format selection
            export_format = st.radio(
                "Select Export Format:",
                ["CSV", "Excel", "JSON", "Parquet", "HTML"],
                horizontal=True,
                key="export_format"
            )
            
            # Additional options based on format
            col1, col2 = st.columns(2)
            
            with col1:
                # File naming
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"cleaned_data_{timestamp}"
                file_name = st.text_input("File Name:", default_name, key="export_name")
                
                # JSON specific options
                if export_format == "JSON":
                    json_orient = st.selectbox(
                        "JSON Format:",
                        ["records", "columns", "split", "index"],
                        key="json_orient"
                    )
            
            with col2:
                # Excel specific options
                if export_format == "Excel":
                    include_index = st.checkbox("Include Index", value=False, key="excel_index")
                    excel_sheet = st.text_input("Sheet Name:", "Cleaned_Data", key="excel_sheet")
                
                # CSV specific options
                elif export_format == "CSV":
                    csv_sep = st.selectbox("Separator:", [",", ";", "\t", "|"], key="csv_sep")
                    csv_index = st.checkbox("Include Index", value=False, key="csv_index")
            
            # Export button
            if st.button(f"📥 Download as {export_format}", use_container_width=True, type="primary"):
                try:
                    if export_format == "CSV":
                        data = df.to_csv(index=csv_index, sep=csv_sep)
                        mime_type = "text/csv"
                        file_ext = ".csv"
                    
                    elif export_format == "Excel":
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=include_index, sheet_name=excel_sheet)
                        data = output.getvalue()
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_ext = ".xlsx"
                    
                    elif export_format == "JSON":
                        data = df.to_json(orient=json_orient, indent=2)
                        mime_type = "application/json"
                        file_ext = ".json"
                    
                    elif export_format == "Parquet":
                        output = io.BytesIO()
                        df.to_parquet(output, index=False)
                        data = output.getvalue()
                        mime_type = "application/octet-stream"
                        file_ext = ".parquet"
                    
                    elif export_format == "HTML":
                        data = df.to_html(index=False, classes='table table-striped', border=0)
                        mime_type = "text/html"
                        file_ext = ".html"
                    
                    # Create download button
                    st.download_button(
                        label=f"Click to Download {file_name}{file_ext}",
                        data=data,
                        file_name=f"{file_name}{file_ext}",
                        mime=mime_type,
                        use_container_width=True
                    )
                    
                    st.success(f"✅ Ready to download {export_format} file!")
                    
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
            
            # Show export stats
            st.markdown("---")
            st.markdown("### 📊 Export Summary")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Rows to Export", f"{len(df):,}")
            
            with col_b:
                st.metric("Columns to Export", f"{len(df.columns)}")
            
            with col_c:
                file_size = len(str(df).encode('utf-8')) / 1024  # KB
                st.metric("Estimated Size", f"{file_size:.1f} KB")
    
    else:
        # WELCOME SCREEN (when no data uploaded)
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2 style="color: #2D3748; margin-bottom: 2rem;">🚀 Welcome to Data Cleaner Pro</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 3rem;">
                <div class="data-card">
                    <h3>📁 Multi-Format Support</h3>
                    <p>CSV, Excel, JSON, Text, TSV files</p>
                </div>
                
                <div class="data-card">
                    <h3>🔍 Smart Issue Detection</h3>
                    <p>10+ data quality checks</p>
                </div>
                
                <div class="data-card">
                    <h3>🛠️ Interactive Cleaning</h3>
                    <p>One-click fixes with ML</p>
                </div>
                
                <div class="data-card">
                    <h3>📊 Beautiful Visualizations</h3>
                    <p>Charts, heatmaps, distributions</p>
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, #4299E1, #667EEA); 
                       padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
                <h3>🎯 How to Use:</h3>
                <ol style="text-align: left; display: inline-block; margin: 0 auto;">
                    <li><strong>Upload</strong> any data file from the sidebar</li>
                    <li><strong>Preview</strong> data and see auto-detected issues</li>
                    <li><strong>Clean</strong> using interactive tools or one-click fixes</li>
                    <li><strong>Visualize</strong> patterns and distributions</li>
                    <li><strong>Export</strong> cleaned data in any format</li>
                </ol>
            </div>
            
            <div class="data-card">
                <h4>Try with Sample Data</h4>
                <p>Upload your own file or try the example below:</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Load sample data button
        if st.button("📊 Load Sample Dataset", use_container_width=True, type="primary"):
            # Create sample dataset with various issues
            np.random.seed(42)
            
            sample_data = {
                'Customer_ID': range(1001, 1101),
                'First_Name': ['John', 'mary', 'ROBERT', 'lisa', 'DAVID', 'sarah', 'MICHAEL', 'jennifer'] * 12 + ['test', 'TEST'],
                'Last_Name': ['Doe', 'smith', 'JOHNSON', 'brown', 'WILSON', 'taylor', 'anderson', 'thomas'] * 12 + ['demo', 'DEMO'],
                'Age': list(np.random.randint(18, 70, 98)) + [None, None],
                'Salary': list(np.random.normal(50000, 15000, 95)) + [1200000, 1500000, -50000, None, None],
                'Email': [f'user{i}@example.com' for i in range(100)],
                'Join_Date': pd.date_range('2020-01-01', periods=100, freq='D').tolist(),
                'Department': ['Sales', 'IT', 'HR', 'Finance', 'Marketing'] * 20,
                'Performance_Rating': np.random.choice(['A', 'B', 'C', None], 100, p=[0.3, 0.4, 0.2, 0.1])
            }
            
            df_sample = pd.DataFrame(sample_data)
            df_sample = pd.concat([df_sample, df_sample.iloc[:3]], ignore_index=True)  # Add duplicates
            
            st.session_state.original_data = df_sample.copy()
            st.session_state.current_data = df_sample.copy()
            st.session_state.file_type = 'sample'
            st.session_state.cleaning_history = []
            st.session_state.analysis_report = analyze_dataset(df_sample)
            st.session_state.issues_found = detect_all_issues(df_sample)
            
            st.rerun()
    
    # ====================
    # 👣 FOOTER
    # ====================
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col2:
        st.markdown("""
        <div style="text-align: center; color: #718096; padding: 2rem;">
            <h4 style="color: #2D3748; margin-bottom: 0.5rem;">🧼 Data Cleaner Pro</h4>
            <p style="margin-bottom: 1rem;">Professional Data Cleaning Solution for Data Scientists</p>
            
            <div style="display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 1rem;">
                <a href="#" style="color: #4A5568; text-decoration: none;">GitHub</a>
                <a href="#" style="color: #4A5568; text-decoration: none;">LinkedIn</a>
                <a href="#" style="color: #4A5568; text-decoration: none;">Portfolio</a>
                <a href="#" style="color: #4A5568; text-decoration: none;">Documentation</a>
            </div>
            
            <p style="font-size: 0.9rem; margin-top: 1rem;">
                © 2024 Adnan Raza | Data Science Student Project<br>
                Built with ❤️ using Streamlit, Pandas & Scikit-learn
            </p>
        </div>
        """, unsafe_allow_html=True)

# ======================
# 🚀 RUN THE APPLICATION
# ======================
if __name__ == "__main__":
    main()