import pickle
import pandas as pd
import numpy as np
from utils import delete_cols_with_any_nan, get_columns_grouped_by_dtypes, x_y_split
from models.catboost_regressor import train_model
from config import PRICE_COL


def analyze_features(df: pd.DataFrame, model=None):
    if model is None:
        model = train_model(df, PRICE_COL)
    
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    X, y = x_y_split(df, PRICE_COL)
    cat_cols = list(get_columns_grouped_by_dtypes(df)['object'])
    recommendations = []
    
    total_importance = importance_df['importance'].sum()
    importance_df['importance_pct'] = (importance_df['importance'] / total_importance * 100).round(4)
    low_importance_threshold = 0.01
    low_importance_features = importance_df[importance_df['importance_pct'] < low_importance_threshold]
    
    if len(low_importance_features) > 0:
        recommendations.append({
            'category': 'Низкая важность признака',
            'description': f'Признаки с важностью менее {low_importance_threshold}% от общей',
            'features': low_importance_features[['feature', 'importance', 'importance_pct']].to_dict('records'),
            'count': len(low_importance_features)
        })
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    variance_analysis = []
    for col in numeric_cols:
        variance = X[col].var()
        std = X[col].std()
        if variance < 0.001 or std < 0.01:
            variance_analysis.append({
                'feature': col,
                'variance': variance,
                'std': std,
                'mean': X[col].mean()
            })
    
    if variance_analysis:
        recommendations.append({
            'category': 'Низкая дисперсия (почти константные признаки)',
            'description': 'Числовые признаки с очень низкой дисперсией (практически константные)',
            'features': variance_analysis,
            'count': len(variance_analysis)
        })
    
    cat_analysis = []
    for col in cat_cols:
        if col in X.columns:
            unique_count = X[col].nunique()
            total_count = len(X[col])
            unique_ratio = unique_count / total_count
            
            if unique_count == total_count:
                cat_analysis.append({
                    'feature': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'reason': 'Каждый объект имеет уникальное значение (как ID)'
                })
            elif unique_count == 1:
                cat_analysis.append({
                    'feature': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'reason': 'Только одно уникальное значение (константа)'
                })
            elif unique_ratio > 0.95:
                cat_analysis.append({
                    'feature': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'reason': f'Очень много уникальных значений ({unique_ratio:.1%}), похоже на идентификатор'
                })
    
    if cat_analysis:
        recommendations.append({
            'category': 'Проблемные категориальные признаки',
            'description': 'Категориальные признаки с подозрительным распределением уникальных значений',
            'features': cat_analysis,
            'count': len(cat_analysis)
        })
    
    if len(numeric_cols) > 1:
        corr_matrix = X[numeric_cols].corr().abs()
        high_corr_pairs = []
        threshold = 0.95
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    col1_importance = importance_df[importance_df['feature'] == col1]['importance'].values
                    col2_importance = importance_df[importance_df['feature'] == col2]['importance'].values
                    
                    if len(col1_importance) > 0 and len(col2_importance) > 0:
                        if col1_importance[0] < col2_importance[0]:
                            to_remove = col1
                        else:
                            to_remove = col2
                        
                        high_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': corr_value,
                            'recommend_to_remove': to_remove
                        })
        
        if high_corr_pairs:
            features_to_remove_corr = list(set([pair['recommend_to_remove'] for pair in high_corr_pairs]))
            recommendations.append({
                'category': 'Высокая корреляция между признаками',
                'description': f'Пары признаков с корреляцией > {threshold} (рекомендуется удалить менее важный)',
                'features': high_corr_pairs[:20],
                'features_to_remove': features_to_remove_corr,
                'count': len(features_to_remove_corr)
            })
    
    return {
        'importance_df': importance_df,
        'recommendations': recommendations,
        'model': model
    }


def print_full_feature_list(importance_df: pd.DataFrame):
    print("=" * 80)
    print("ПОЛНЫЙ СПИСОК ПАРАМЕТРОВ С ИХ ВАЖНОСТЬЮ")
    print("=" * 80)
    print(f"\nВсего параметров: {len(importance_df)}\n")
    print(importance_df.to_string(index=False))
    print("\n" + "=" * 80)


def print_recommendations(recommendations: list, importance_df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("АНАЛИЗ ПАРАМЕТРОВ И РЕКОМЕНДАЦИИ ПО УДАЛЕНИЮ")
    print("=" * 80)
    
    total_recommended = 0
    for rec in recommendations:
        print(f"\n{rec['category']}:")
        print(f"  Описание: {rec['description']}")
        print(f"  Количество признаков к удалению: {rec['count']}")
        
        if 'features' in rec and len(rec['features']) > 0:
            print(f"\n  Детали:")
            for feat in rec['features'][:10]:
                if isinstance(feat, dict):
                    print(f"    - {feat.get('feature', 'N/A')}: ", end="")
                    details = {k: v for k, v in feat.items() if k != 'feature'}
                    print(", ".join([f"{k}={v}" for k, v in details.items()]))
            
            if len(rec['features']) > 10:
                print(f"    ... и еще {len(rec['features']) - 10} признаков")
        
        if 'features_to_remove' in rec:
            print(f"\n  Рекомендуемые к удалению признаки:")
            for feat in rec['features_to_remove'][:15]:
                print(f"    - {feat}")
            if len(rec['features_to_remove']) > 15:
                print(f"    ... и еще {len(rec['features_to_remove']) - 15} признаков")
        
        total_recommended += rec['count']
        print()
    
    print("=" * 80)
    print(f"\nИТОГО: Рекомендуется удалить примерно {total_recommended} признаков из {len(importance_df)}")
    print(f"Это составляет {total_recommended/len(importance_df)*100:.1f}% от общего количества признаков")
    print("\n" + "=" * 80)
    
    return total_recommended


def save_report(recommendations: list, output_file: str = 'feature_removal_recommendations.txt'):
    report_lines = []
    report_lines.append("ДЕТАЛЬНЫЙ ОТЧЕТ ПО РЕКОМЕНДАЦИЯМ УДАЛЕНИЯ ПРИЗНАКОВ\n")
    report_lines.append("=" * 80 + "\n\n")
    
    all_features_to_remove = set()
    
    for rec in recommendations:
        report_lines.append(f"{rec['category']}\n")
        report_lines.append(f"{rec['description']}\n")
        report_lines.append(f"Количество: {rec['count']}\n\n")
        
        if 'features' in rec:
            for feat in rec['features']:
                if isinstance(feat, dict):
                    feature_name = feat.get('feature', 'N/A')
                    report_lines.append(f"  - {feature_name}\n")
                    all_features_to_remove.add(feature_name)
        
        if 'features_to_remove' in rec:
            for feat in rec['features_to_remove']:
                report_lines.append(f"  - {feat}\n")
                all_features_to_remove.add(feat)
        
        report_lines.append("\n")
    
    report_lines.append("\n" + "=" * 80 + "\n")
    report_lines.append(f"ВСЕГО УНИКАЛЬНЫХ ПРИЗНАКОВ К УДАЛЕНИЮ: {len(all_features_to_remove)}\n")
    report_lines.append("=" * 80 + "\n\n")
    report_lines.append("Список всех признаков к удалению:\n")
    for feat in sorted(all_features_to_remove):
        report_lines.append(f"  - {feat}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"\nДетальный отчет сохранен в файл: {output_file}")
    return list(all_features_to_remove)


def clean_features(df: pd.DataFrame, features_to_remove: list) -> pd.DataFrame:
    features_to_remove = [f for f in features_to_remove if f in df.columns]
    df_cleaned = df.drop(columns=features_to_remove)
    print(f"\nУдалено {len(features_to_remove)} признаков из {len(df.columns)}")
    print(f"Осталось признаков: {len(df_cleaned.columns)}")
    return df_cleaned


def main():
    with open('data/result.pkl', 'rb') as f:
        data = pickle.load(f)
    
    df = delete_cols_with_any_nan(pd.DataFrame(data))
    
    analysis_result = analyze_features(df)
    importance_df = analysis_result['importance_df']
    recommendations = analysis_result['recommendations']
    
    print_full_feature_list(importance_df)
    print_recommendations(recommendations, importance_df)
    
    all_features_to_remove = save_report(recommendations)
    
    df_cleaned = clean_features(df, all_features_to_remove)
    df_cleaned.to_pickle('data/result_cleaned.pkl')
    print(f"\nОчищенный датасет сохранен в data/result_cleaned.pkl")


if __name__ == '__main__':
    main()
