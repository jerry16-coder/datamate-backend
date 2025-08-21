from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read file based on type
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            
            # Perform analysis
            analysis = perform_analysis(df, filename, filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(analysis)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def perform_analysis(df, filename, filepath):
    """Perform comprehensive data analysis"""
    
    # Basic file info
    file_size = os.path.getsize(filepath)
    
    # Data quality assessment
    missing_data = df.isnull().sum().sum()
    missing_percentage = (missing_data / (df.shape[0] * df.shape[1])) * 100
    completeness = 100 - missing_percentage
    
    # Quality score calculation
    quality_score = calculate_quality_score(df)
    
    # Statistical analysis
    statistics = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if len(df[col].dropna()) > 0:
            statistics[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
            }
    
    # Generate insights
    insights = generate_insights(df)
    patterns = detect_patterns(df)
    correlations = analyze_correlations(df)
    
    # Business insights
    business_insights = generate_business_insights(df)
    
    # Predictive insights (simple trend analysis)
    predictive_insights = generate_predictive_insights(df)
    
    return {
        'fileInfo': {
            'name': filename,
            'size': f"{file_size} bytes",
            'type': filename.split('.')[-1].upper(),
        },
        'dataInfo': {
            'rows': len(df),
            'columns': len(df.columns),
            'columnNames': df.columns.tolist(),
        },
        'dataQuality': {
            'overallScore': quality_score,
            'qualityLevel': get_quality_level(quality_score),
            'missingDataPercentage': round(missing_percentage, 2),
            'completenessPercentage': round(completeness, 2),
            'duplicatePercentage': calculate_duplicate_percentage(df),
            'outlierPercentage': calculate_outlier_percentage(df),
            'recommendations': generate_recommendations(df),
        },
        'missingValues': {
            'total': int(missing_data),
            'byColumn': df.isnull().sum().to_dict(),
        },
        'statistics': statistics,
        'insights': insights,
        'patterns': patterns,
        'correlations': correlations,
        'businessInsights': business_insights,
        'predictiveInsights': predictive_insights,
        'summary': {
            'datasetOverview': f"This dataset contains {len(df)} records with {len(df.columns)} variables. The data quality score is {quality_score}/100.",
            'dataQuality': f"Missing data: {round(missing_percentage, 1)}%, Completeness: {round(completeness, 1)}%",
        },
        'preview': df.head(5).values.tolist(),
    }

def calculate_quality_score(df):
    """Calculate overall data quality score"""
    score = 100
    
    # Deduct for missing values
    missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    score -= missing_percentage * 0.5
    
    # Deduct for duplicates
    duplicate_percentage = calculate_duplicate_percentage(df)
    score -= duplicate_percentage * 0.3
    
    # Deduct for outliers
    outlier_percentage = calculate_outlier_percentage(df)
    score -= outlier_percentage * 0.2
    
    return max(0, min(100, round(score)))

def get_quality_level(score):
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Good"
    elif score >= 70:
        return "Fair"
    else:
        return "Poor"

def calculate_duplicate_percentage(df):
    """Calculate percentage of duplicate rows"""
    duplicates = df.duplicated().sum()
    return round((duplicates / len(df)) * 100, 2)

def calculate_outlier_percentage(df):
    """Calculate percentage of outliers in numeric columns"""
    outlier_count = 0
    total_count = 0
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if len(df[col].dropna()) > 10:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_count += len(outliers)
            total_count += len(df[col].dropna())
    
    return round((outlier_count / total_count) * 100, 2) if total_count > 0 else 0

def generate_insights(df):
    """Generate basic insights about the data"""
    insights = []
    
    # Missing values insight
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        insights.append(f"Dataset has {missing_total} missing values that need attention")
    
    # Data size insights
    if len(df) > 1000:
        insights.append(f"Large dataset with {len(df)} records - good for statistical analysis")
    elif len(df) < 100:
        insights.append(f"Small dataset with {len(df)} records - consider collecting more data")
    
    # Column type insights
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=['object']).columns)
    
    if numeric_cols > 0:
        insights.append(f"Contains {numeric_cols} numeric variables suitable for statistical analysis")
    if categorical_cols > 0:
        insights.append(f"Contains {categorical_cols} categorical variables for grouping analysis")
    
    return insights

def detect_patterns(df):
    """Detect patterns in the data"""
    patterns = []
    
    # Trend detection for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if len(numeric_df[col].dropna()) > 10:
            # Check for increasing trend
            sorted_values = numeric_df[col].dropna().sort_values()
            if len(sorted_values) > 1:
                correlation = np.corrcoef(range(len(sorted_values)), sorted_values)[0, 1]
                if correlation > 0.7:
                    patterns.append(f"{col} shows a strong increasing trend")
                elif correlation < -0.7:
                    patterns.append(f"{col} shows a strong decreasing trend")
    
    # Distribution patterns
    for col in numeric_df.columns:
        if len(numeric_df[col].dropna()) > 10:
            skewness = stats.skew(numeric_df[col].dropna())
            if abs(skewness) > 1:
                direction = "right-skewed" if skewness > 0 else "left-skewed"
                patterns.append(f"{col} distribution is {direction}")
    
    return patterns

def analyze_correlations(df):
    """Analyze correlations between numeric variables"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {'strongCorrelations': []}
    
    corr_matrix = numeric_df.corr()
    strong_correlations = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                strength = "very strong" if abs(corr_val) > 0.9 else "strong"
                direction = "positive" if corr_val > 0 else "negative"
                strong_correlations.append({
                    'variable1': col1,
                    'variable2': col2,
                    'correlation': round(corr_val, 3),
                    'strength': f"{strength} {direction}",
                })
    
    return {'strongCorrelations': strong_correlations}

def generate_business_insights(df):
    """Generate business-relevant insights"""
    insights = []
    
    # Sample business insights based on common column names
    for col in df.columns:
        col_lower = col.lower()
        
        if 'sales' in col_lower or 'revenue' in col_lower:
            if col in df.select_dtypes(include=[np.number]).columns:
                avg_sales = df[col].mean()
                max_sales = df[col].max()
                insights.append({
                    'type': 'Sales Analysis',
                    'insight': f"Average sales: ${avg_sales:,.2f}, Peak sales: ${max_sales:,.2f}"
                })
        
        elif 'customer' in col_lower or 'user' in col_lower:
            unique_customers = df[col].nunique()
            insights.append({
                'type': 'Customer Analysis',
                'insight': f"Total unique customers: {unique_customers}"
            })
        
        elif 'date' in col_lower or 'time' in col_lower:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_range = f"{df[col].min().strftime('%Y-%m-%d')} to {df[col].max().strftime('%Y-%m-%d')}"
                    insights.append({
                        'type': 'Time Analysis',
                        'insight': f"Data spans: {date_range}"
                    })
                except:
                    pass
    
    return insights

def generate_predictive_insights(df):
    """Generate simple predictive insights"""
    insights = []
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Simple linear trend prediction
    for col in numeric_df.columns:
        if len(numeric_df[col].dropna()) > 10:
            x = np.arange(len(numeric_df[col].dropna()))
            y = numeric_df[col].dropna().values
            
            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            if abs(r_value) > 0.5:  # Only report if correlation is meaningful
                trend = "increasing" if slope > 0 else "decreasing"
                confidence = "high" if abs(r_value) > 0.8 else "moderate"
                
                insights.append({
                    'type': f'{col} Trend Prediction',
                    'r_squared': round(r_value**2, 3),
                    'insight': f"{col} shows a {trend} trend with {confidence} confidence (R² = {r_value**2:.3f})"
                })
    
    return insights

def generate_recommendations(df):
    """Generate data quality recommendations"""
    recommendations = []
    
    # Missing data recommendations
    missing_by_col = df.isnull().sum()
    high_missing_cols = missing_by_col[missing_by_col > len(df) * 0.1].index.tolist()
    
    if high_missing_cols:
        recommendations.append(f"Consider removing or imputing missing values in: {', '.join(high_missing_cols)}")
    
    # Duplicate recommendations
    if df.duplicated().sum() > 0:
        recommendations.append("Remove duplicate rows to improve data quality")
    
    # Outlier recommendations
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        if len(numeric_df[col].dropna()) > 10:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = numeric_df[(numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > len(numeric_df) * 0.05:  # More than 5% outliers
                recommendations.append(f"Investigate outliers in {col} column")
    
    return recommendations

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'DataMate Analysis API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
