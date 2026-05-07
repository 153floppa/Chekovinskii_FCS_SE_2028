"""
Streamlit приложение для визуализации результатов ML-модели прогнозирования цены коммерческой недвижимости на Авито.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import re
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go

from interface.utils import delete_cols_with_any_nan, get_columns_grouped_by_dtypes, x_y_split
from model.regressors.catboost_regressor import get_model
from config import PRICE_COL, SQUARE_COL

# Import JSON formatter for model compatibility
try:
    from dataset.collection.avito.fix_json_format import fix_object, POI_TYPES, RADII
    HAS_FIX_FORMATTER = True
except ImportError as e:
    HAS_FIX_FORMATTER = False
    print(f"Warning: Could not import fix_json_format: {e}")

# ============================================
# Конфигурация Streamlit
# ============================================

st.set_page_config(
    page_title="Анализ коммерческой недвижимости",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Кэширование данных и модели (загружаются один раз)
# ============================================

@st.cache_resource
def load_model():
    """Загружает обученную модель CatBoost."""
    try:
        model = get_model('catboost_regressor')
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


@st.cache_data
def load_data():
    """Загружает датасет."""
    with open('dataset/data/dataset_final.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


@st.cache_data
def load_training_data():
    """Загружает данные обучающей выборки (для расчета метрик)."""
    with open('dataset/data/result.pkl', 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    df_clean = delete_cols_with_any_nan(df)
    return df_clean


def predict_prices(df: pd.DataFrame, model) -> np.ndarray:
    """Предсказывает цены для датасета."""
    X, _ = x_y_split(df, PRICE_COL)
    cat_cols = list(get_columns_grouped_by_dtypes(df)['object'])

    from catboost import Pool
    pool = Pool(X, cat_features=cat_cols)
    price_per_sqm = model.predict(pool)

    # Переводим price/m² в полную цену
    full_price = price_per_sqm * df[SQUARE_COL]
    return full_price


def calculate_metrics(df: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Рассчитывает метрики качества модели."""
    # Все метрики считаем на цене за м² (как модель обучалась)
    actual_price_per_sqm = df[PRICE_COL]
    predicted_price_per_sqm = predictions / df[SQUARE_COL]

    # MAPE - Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual_price_per_sqm - predicted_price_per_sqm) / actual_price_per_sqm)) * 100

    # MAE - Mean Absolute Error (в рублях за м²)
    mae = np.mean(np.abs(actual_price_per_sqm - predicted_price_per_sqm))

    # R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(actual_price_per_sqm, predicted_price_per_sqm)

    # RMSE (в рублях за м²)
    rmse = np.sqrt(np.mean((actual_price_per_sqm - predicted_price_per_sqm) ** 2))

    return {
        'mape': mape,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }


def extract_area_from_params(params_str):
    """Извлекает площадь из поля Параметры (Excel export format)."""
    if not isinstance(params_str, str):
        return None
    match = re.search(r'Общая площадь, число:\s*([\d.,]+)', params_str)
    if match:
        area_str = match.group(1).replace(',', '.')
        try:
            return float(area_str)
        except ValueError:
            return None
    return None


def format_currency(x):
    """Округляет до тысяч и форматирует валюту."""
    rounded = round(x / 1000) * 1000
    return f"₽{rounded:,.0f}"


@st.cache_data
def format_json_for_model(json_data: list) -> list:
    """
    Преобразует JSON в формат совместимый с моделью.
    - Цена → цена за м² (делим на площадь)
    - MIN расстояния → int
    - Добавляет все POI поля со всеми радиусами
    - Координаты → null
    """
    if not HAS_FIX_FORMATTER:
        st.error("❌ Модуль форматирования недоступен (fix_json_format.py)")
        return json_data

    formatted_data = []
    for obj in json_data:
        try:
            formatted_obj = fix_object(obj)
            formatted_data.append(formatted_obj)
        except Exception as e:
            st.warning(f"⚠️ Ошибка при форматировании объекта: {e}")
            formatted_data.append(obj)

    return formatted_data


def get_top_underestimated(df: pd.DataFrame, predictions: np.ndarray, top_n: int = 50) -> pd.DataFrame:
    """Возвращает объекты отсортированные по недооценке (все объекты, не только положительные)."""
    df_result = df.copy()
    df_result['predicted_price'] = predictions
    # Переводим цену за м² в полную цену для сравнения
    actual_full_price = df_result[PRICE_COL] * df_result[SQUARE_COL]
    df_result['difference'] = df_result['predicted_price'] - actual_full_price
    df_result['difference_pct'] = (df_result['difference'] / actual_full_price * 100).round(2)

    # Показываем все объекты, отсортированные по разнице (вверху недооцененные)
    sorted_df = df_result.sort_values('difference', ascending=False)

    return sorted_df.head(top_n)


# ============================================
# Боковая панель (Sidebar)
# ============================================

with st.sidebar:
    st.title("⚙️ Настройки")

    model = load_model()
    if model is None:
        st.error("Модель не загружена!")
        st.stop()

    df_raw = load_data()
    df_train = load_training_data()

    st.success(f"✅ Модель загружена")
    st.info(f"📊 Объектов в датасете: {len(df_raw)}\n\nОбъектов в обучении: {len(df_train)}")

    # Режимы выбора объекта
    view_mode = st.radio(
        "Выбери режим:",
        ["🎯 Топ недооценённых", "📊 Модель и метрики", "🔍 Аналитика объекта", "📂 Загрузка аналитики", "📥 Конвертер Excel"]
    )

# ============================================
# Основной контент
# ============================================

# Рассчитываем предсказания один раз
if 'predictions' not in st.session_state:
    with st.spinner("Рассчитываю предсказания..."):
        st.session_state.predictions = predict_prices(df_train, model)
        st.session_state.metrics = calculate_metrics(df_train, st.session_state.predictions)

predictions = st.session_state.predictions
metrics = st.session_state.metrics

# ============================================
# Вкладка 1: Топ недооценённых объектов (общий, без разбиений)
# ============================================

if view_mode == "🎯 Топ недооценённых":
    st.title("🎯 Недооценённые объекты")

    st.markdown("""
    Топ объектов из всего датасета (включая новые объявления),
    где **реальная цена ниже справедливой** (по модели).
    """)

    uploaded_file = st.file_uploader("Выбери JSON-файл", type=["json"], key="topn_uploader")

    if uploaded_file is not None:
        try:
            # Загружаем файл
            data = json.load(uploaded_file)

            # Обработка вложенной структуры (например, из Excel конвертера)
            if isinstance(data, dict) and not isinstance(next(iter(data.values()), None), dict):
                # Это словарь со списками (например, {"Sheet1": [...], "Sheet2": [...]})
                all_records = []
                for sheet_data in data.values():
                    if isinstance(sheet_data, list):
                        all_records.extend(sheet_data)
                data = all_records

            df_uploaded = pd.DataFrame(data)

            # Извлекаем площадь из Параметры если отсутствует столбец Общая площадь (Excel export)
            if SQUARE_COL not in df_uploaded.columns and 'Параметры' in df_uploaded.columns:
                df_uploaded[SQUARE_COL] = df_uploaded['Параметры'].apply(extract_area_from_params)

            # Очищаем данные
            df_clean = delete_cols_with_any_nan(df_uploaded)

            st.info(f"✅ Загружено {len(df_clean):,} объектов (из {len(df_uploaded):,})")

            # Настройки
            col1, col2 = st.columns(2)
            top_n = st.slider("Сколько объектов показать?", 10, 300, 100, key="topn_slider")

            # Проверяем наличие POI признаков (обогащённые данные)
            poi_features = [col for col in df_clean.columns if any(x in col for x in ['Аптека', 'Банк', 'Бар', 'Школа', 'Фитнес', 'Метро', 'Остановка'])]
            has_poi_features = len(poi_features) > 0

            # Проверяем инженерные признаки (для прогнозирования)
            engineered_features = ['Квартир200', 'Квартир500', 'Площадь200']
            has_engineered = all(f in df_clean.columns for f in engineered_features)

            if not has_poi_features and not has_engineered:
                # Сырые данные без обогащения
                st.warning(
                    "⚠️ **Данные не содержат признаки обогащения (POI)**\n\n"
                    "Эти данные могут быть сырой выгрузкой Авито. "
                    "Для полного анализа используйте обогащённые данные с POI признаками."
                )
                st.info(f"📊 **Загружено:** {len(df_clean)} объектов")
                st.dataframe(df_clean[[col for col in ['Район', SQUARE_COL, PRICE_COL, 'Вид объекта'] if col in df_clean.columns]].head(10), use_container_width=True)
                st.stop()

            if has_poi_features and not has_engineered:
                # Обогащённые данные, но без инженерных признаков из обучения
                st.info(
                    "ℹ️ **Данные обогащены POI признаками**\n\n"
                    "Содержит информацию о близлежащих объектах (аптеки, школы, метро и т.д.). "
                    f"Обнаружено {len(poi_features)} POI признаков."
                )
            elif has_engineered and not has_poi_features:
                # Обработанные данные из обучающего набора (без обогащения)
                st.info(
                    "ℹ️ **Данные из обучающего набора модели**\n\n"
                    "Содержит инженерные признаки для прогнозирования модели."
                )

            # Предсказываем
            with st.spinner("Рассчитываю предсказания..."):
                full_predictions = predict_prices(df_clean, model)

            # Анализируем результаты
            df_result = df_clean.copy()
            df_result['predicted_price'] = full_predictions
            # Переводим цену за м² в полную цену для сравнения
            actual_full_price = df_result[PRICE_COL] * df_result[SQUARE_COL]
            df_result['difference'] = df_result['predicted_price'] - actual_full_price
            df_result['difference_pct'] = (df_result['difference'] / actual_full_price * 100).round(2)

            # Показываем ВСЕ объекты, отсортированные по разнице (недооцененные вверху)
            top_underestimated = df_result.sort_values('difference', ascending=False).head(top_n)

            if len(top_underestimated) == 0:
                st.warning("❌ Объекты не найдены")
            else:
                # Подготовка таблицы с нужными столбцами
                df_display = pd.DataFrame()
                df_display['Площадь (м²)'] = top_underestimated[SQUARE_COL].astype(int)
                df_display['Цена/м² реальная'] = top_underestimated[PRICE_COL].apply(lambda x: f"₽{x:,.0f}")
                df_display['Цена/м² предсказ'] = (top_underestimated['predicted_price'] / top_underestimated[SQUARE_COL]).apply(lambda x: f"₽{x:,.0f}")
                df_display['Маржинальность (%)'] = top_underestimated['difference_pct'].apply(lambda x: f"{x:.1f}%")
                df_display['Реальная общая цена'] = (top_underestimated[PRICE_COL] * top_underestimated[SQUARE_COL]).apply(format_currency)
                df_display['Предсказанная цена'] = top_underestimated['predicted_price'].apply(format_currency)
                df_display['Прибыль (руб)'] = top_underestimated['difference'].apply(format_currency)
                if 'Район' in top_underestimated.columns:
                    df_display['Район'] = top_underestimated['Район']
                if 'Вид объекта' in top_underestimated.columns:
                    df_display['Тип объекта'] = top_underestimated['Вид объекта']
                elif 'Подкатегория' in top_underestimated.columns:
                    df_display['Тип объекта'] = top_underestimated['Подкатегория']

                # Добавляем колонку со ссылкой на объявление
                if 'Ссылка' in top_underestimated.columns:
                    df_display['🔗 Avito'] = top_underestimated['Ссылка'].apply(
                        lambda x: f"[Смотреть →]({x})" if pd.notna(x) and isinstance(x, str) and x.startswith('http') else "—"
                    )

                # Таблица с возможностью выбора объектов
                st.dataframe(df_display, use_container_width=True, height=500)

                st.subheader("🔎 Анализ одного объекта")
                st.markdown("Выбери объект для детального AI-анализа:")

                # Создаем список для выбора
                object_labels = []
                for idx, (i, row) in enumerate(top_underestimated.iterrows()):
                    square = row.get(SQUARE_COL, 0)
                    profit = top_underestimated.iloc[idx]['difference']
                    profit_rounded = round(profit / 1000) * 1000
                    label = f"{idx+1}. {row.get('Район', 'N/A')} | {row.get('Вид объекта', 'N/A')} | {square:.0f}м² | +₽{profit_rounded:,.0f}"
                    object_labels.append((idx, label, row))

                col1, col2 = st.columns([4, 1])
                with col1:
                    selected_idx = st.selectbox(
                        "Выбери объект из списка",
                        range(len(object_labels)),
                        format_func=lambda x: object_labels[x][1],
                        key="object_selector"
                    )

                with col2:
                    if st.button("📊 Анализировать", key="select_obj_btn"):
                        idx, label, row = object_labels[selected_idx]
                        # Готовим данные объекта
                        square = top_underestimated.iloc[idx][SQUARE_COL]
                        real_price = top_underestimated.iloc[idx][PRICE_COL] * square
                        selected_obj = {
                            'Район': row.get('Район', 'N/A'),
                            'Вид объекта': row.get('Вид объекта', 'N/A'),
                            'Площадь (м²)': square,
                            'Этаж': row.get('Этаж', 'N/A'),
                            'Этажность здания': row.get('Этажность здания', 'N/A'),
                            'Реальная общая цена': real_price,
                            'Предсказанная цена': top_underestimated.iloc[idx]['predicted_price'],
                            'Прибыль (руб)': top_underestimated.iloc[idx]['difference'],
                            'Маржинальность (%)': top_underestimated.iloc[idx]['difference_pct'],
                        }
                        st.session_state.selected_object = selected_obj
                        # Сохраняем загруженные данные и индекс для анализа
                        st.session_state.current_uploaded_json = df_uploaded.to_dict('records')
                        st.session_state.selected_object_idx = idx
                        st.success("✅ Объект выбран! Переключись на 'Аналитика объекта' в боковой панели")

                # Статистика
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Найдено недооцененных", len(top_underestimated))
                with col2:
                    avg_undervalue = top_underestimated['difference_pct'].mean()
                    st.metric("Средняя недооценка", f"{avg_undervalue:.1f}%")
                with col3:
                    total_potential = top_underestimated['difference'].sum()
                    total_rounded = round(total_potential / 1000) * 1000
                    st.metric("Суммарная недооценка", f"₽{total_rounded:,.0f}")

                # Экспорт
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_display.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Скачать CSV",
                        data=csv,
                        file_name="underestimated.csv",
                        mime="text/csv",
                        key="csv_topn"
                    )
                with col2:
                    json_data = json.dumps(top_underestimated[['Ссылка', 'Цена', 'predicted_price', 'Район', SQUARE_COL, 'Вид объекта']].to_dict('records'), ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Скачать JSON",
                        data=json_data,
                        file_name="underestimated.json",
                        mime="application/json",
                        key="json_topn"
                    )

                # График по районам
                if 'Район' in top_underestimated.columns:
                    st.subheader("📍 Топ районов по недооценке")
                    district_stats = top_underestimated.groupby('Район').agg({
                        'difference_pct': 'mean',
                        'Ссылка': 'count'
                    }).round(1)
                    district_stats.columns = ['Средняя недооценка (%)', 'Объектов']
                    district_stats = district_stats.sort_values('Средняя недооценка (%)', ascending=False)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write("**Статистика по районам:**")
                        st.dataframe(district_stats)

                    with col2:
                        fig = px.bar(
                            district_stats.reset_index(),
                            x='Район', y='Средняя недооценка (%)',
                            title="Средняя недооценка по районам (%)",
                            color='Средняя недооценка (%)',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {str(e)}")
    else:
        st.info("👆 Загрузи JSON-файл для анализа")

# ============================================
# Вкладка 2: Топ недооценённых объектов (обучающая выборка)
# ============================================
# Вкладка 2: Аналитика по одному объекту
# ============================================
# Вкладка 2: Метрики и анализ модели
# ============================================

elif view_mode == "📊 Модель и метрики":
    st.title("📊 Качество модели")

    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{metrics['r2']:.4f}", "выше = лучше")
    with col2:
        st.metric("MAPE", f"{metrics['mape']:.2f}%", "ниже = лучше")
    with col3:
        mae_rounded = round(metrics['mae'] / 1000) * 1000
        st.metric("MAE", f"₽{mae_rounded:,.0f}", "ниже = лучше")
    with col4:
        rmse_rounded = round(metrics['rmse'] / 1000) * 1000
        st.metric("RMSE", f"₽{rmse_rounded:,.0f}", "ниже = лучше")

    st.divider()

    # График predicted vs actual
    st.subheader("📈 Predicted vs Actual Price")

    df_metrics = df_train.copy()
    df_metrics['predicted_price'] = predictions
    # Переводим цену за м² в полную цену для сравнения
    df_metrics['actual_full_price'] = df_metrics[PRICE_COL] * df_metrics[SQUARE_COL]

    fig = px.scatter(
        df_metrics,
        x='actual_full_price',
        y='predicted_price',
        title="Предсказания модели vs реальные цены",
        labels={'actual_full_price': "Реальная полная цена (₽)", 'predicted_price': "Предсказание (₽)"},
        height=500
    )

    # Добавляем линию y=x (идеальное предсказание)
    min_price = min(df_metrics['actual_full_price'].min(), df_metrics['predicted_price'].min())
    max_price = max(df_metrics['actual_full_price'].max(), df_metrics['predicted_price'].max())

    fig.add_trace(go.Scatter(
        x=[min_price, max_price],
        y=[min_price, max_price],
        mode='lines',
        name='Идеальное предсказание',
        line=dict(color='red', dash='dash')
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Feature importance (если есть)
    st.subheader("⭐ Важность признаков")

    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        feature_names = model.feature_names_

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(20)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 20 важнейших признаков",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# Вкладка 3: Аналитика объекта
# ============================================

elif view_mode == "🔍 Аналитика объекта":
    st.title("🔎 AI-Анализ объекта недвижимости")

    # Получаем выбранный объект из session state
    if 'selected_object' not in st.session_state:
        st.info("👆 Выберите объект из таблицы в разделе 'Топ недооцененных'")
    else:
        obj = st.session_state.selected_object

        # Базовая информация об объекте
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Район", obj.get('Район', 'N/A'))
        with col2:
            st.metric("Тип объекта", obj.get('Вид объекта', 'N/A'))
        with col3:
            st.metric("Площадь (м²)", int(obj.get('Площадь (м²)', 0)))

        st.divider()

        # Ценовая информация
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Реальная цена", format_currency(obj.get('Реальная общая цена', 0)), "текущая стоимость")
        with col2:
            st.metric("Предсказанная цена", format_currency(obj.get('Предсказанная цена', 0)), "по модели")
        with col3:
            profit = obj.get('Прибыль (руб)', 0)
            pct = obj.get('Маржинальность (%)', 0)
            st.metric("Потенциальная прибыль", format_currency(profit), f"{pct:.1f}%")

        st.divider()

        # Claude Anthropic анализ
        st.subheader("🤖 AI-Анализ (GPT 5.5)")

        if st.button("🔍 Получить анализ от AI", type="primary", key="analyze_btn"):
            try:
                import os
                import json
                from utils import delete_cols_with_any_nan, get_columns_grouped_by_dtypes, x_y_split

                # Загружаем оригинальные данные объекта для полного JSON
                uploaded_file_key = "current_uploaded_json"
                if uploaded_file_key not in st.session_state:
                    st.error("❌ Нет загруженного файла. Загрузи JSON заново в 'Топ недооцененных'")
                else:
                    uploaded_data = st.session_state[uploaded_file_key]
                    # Получаем текущий объект из загруженных данных по индексу
                    if 'selected_object_idx' not in st.session_state:
                        st.error("❌ Индекс объекта не найден")
                    else:
                        obj_idx = st.session_state['selected_object_idx']
                        if obj_idx < len(uploaded_data):
                            full_object = uploaded_data[obj_idx]

                            with st.spinner("⏳ Анализирую объект..."):
                                import requests
                                import math

                                # Helper: Clean NaN/Inf values for JSON serialization
                                def clean_value(v):
                                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                                        return None
                                    return v if v is not None else 0

                                # Clean NaN values from object
                                clean_object = {k: clean_value(v) for k, v in full_object.items()}

                                # Call analysis microservice
                                try:
                                    response = requests.post(
                                        "http://localhost:8001/api/v1/analyze",
                                        json={
                                            "object": clean_object,
                                            "predicted_price": clean_value(obj.get('Предсказанная цена', 0)),
                                            "real_price": clean_value(obj.get('Реальная общая цена', 0)),
                                            "margin_pct": clean_value(obj.get('Маржинальность (%)', 0)),
                                            "model": "gpt-5.5",
                                            "max_tokens": 8000
                                        },
                                        timeout=120
                                    )
                                    response.raise_for_status()
                                    data = response.json()
                                    analysis = data["analysis"]

                                    if not analysis or analysis.strip() == "":
                                        st.error("❌ Модель вернула пустой ответ")
                                    else:
                                        # Сохраняем в session state с надежным ключом
                                        if 'analysis_cache' not in st.session_state:
                                            st.session_state.analysis_cache = {}
                                        # Используем индекс объекта как ключ (более надёжно чем конкатенация)
                                        obj_key = f"obj_{st.session_state.get('selected_object_idx', 0)}"
                                        st.session_state.analysis_cache[obj_key] = analysis

                                        # Выводим анализ
                                        st.markdown("### 📊 Результат анализа")
                                        st.markdown(analysis)

                                except requests.exceptions.ConnectionError:
                                    st.error("❌ Сервис анализа недоступен (localhost:8001)")
                                    st.info("Запусти: `python services/analysis_service.py` в отдельном терминале")
                                except requests.exceptions.RequestException as e:
                                    st.error(f"❌ Ошибка микросервиса: {e}")
                        else:
                            st.error("❌ Индекс объекта выходит за границы")

            except Exception as e:
                st.error(f"❌ Ошибка при получении анализа: {str(e)}")
                st.info("Проверь: PROXYAPI_KEY установлена? URL корректный?")
                import traceback
                st.code(traceback.format_exc(), language="python")

        # Показываем кэшированный анализ если есть
        elif 'analysis_cache' in st.session_state:
            obj_key = f"obj_{st.session_state.get('selected_object_idx', 0)}"
            if obj_key in st.session_state.analysis_cache:
                st.markdown("### 📊 Сохраненный анализ")
                st.markdown(st.session_state.analysis_cache[obj_key])

# ============================================
# Вкладка 4: Загрузка датасета для аналитики
# ============================================

elif view_mode == "📂 Загрузка аналитики":
    import tempfile
    import os
    from model.splitting.predict_and_split import process_analytics_dataset

    st.title("📂 Аналитика: загрузка датасета")
    st.markdown("""
    Загрузи JSON-датасет в формате `dataset_new.json` — приложение рассчитает
    оптимальное разбиение каждого объекта и покажет потенциальную выгоду.
    """)

    uploaded_file = st.file_uploader("Выбери JSON-файл", type=["json"])

    if uploaded_file is not None:
        process_key = f"analytics_{uploaded_file.name}_{uploaded_file.size}"

        if st.button("▶️ Запустить анализ", type="primary") or process_key in st.session_state:
            if process_key not in st.session_state:
                with st.spinner("Анализируем объекты... это может занять несколько минут"):
                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='wb') as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        out_path = tmp_path.replace(".json", "_results.json")
                        process_analytics_dataset(tmp_path, out_path)
                        with open(out_path, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        st.session_state[process_key] = results
                    finally:
                        os.unlink(tmp_path)
                        if os.path.exists(out_path):
                            os.unlink(out_path)

            results = st.session_state.get(process_key, [])

            if not results:
                st.error("Не удалось обработать датасет.")
            else:
                # Метрики сводки
                valid = [r for r in results if 'error' not in r]
                to_split = [r for r in valid if r.get('should_split')]
                errors = [r for r in results if 'error' in r]
                total_profit = sum(r.get('profit_increase', 0) for r in valid)
                avg_profit_pct = np.mean([r['profit_increase_pct'] for r in valid if 'profit_increase_pct' in r]) if valid else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Всего объектов", len(results))
                c2.metric("Успешно обработано", len(valid))
                c3.metric("Ошибок", len(errors))
                c4.metric("Рекомендуется разбить", len(to_split))

                if errors:
                    st.warning(f"⚠️ Ошибок при обработке: {len(errors)} объектов (они показаны внизу с пометкой ошибки)")

                st.divider()

                # Фильтры
                with st.expander("⚙️ Фильтры и сортировка", expanded=True):
                    fcol1, fcol2, fcol3 = st.columns(3)
                    with fcol1:
                        only_profitable = st.checkbox("Только выгодные для разбиения", value=False)
                    with fcol2:
                        districts = sorted({r.get('district', '') for r in results if r.get('district')})
                        selected_districts = st.multiselect("Районы", districts, default=[])
                    with fcol3:
                        min_profit_pct = st.slider("Мин. прибыль (%)", 0, 100, 0)

                sort_by = st.selectbox(
                    "Сортировать по",
                    ["profit_increase_pct", "profit_increase", "original_price", "num_parts"],
                    format_func=lambda x: {
                        "profit_increase_pct": "Прибыль %",
                        "profit_increase": "Прибыль ₽",
                        "original_price": "Цена объекта",
                        "num_parts": "Кол-во частей"
                    }[x]
                )

                # Применяем фильтры
                filtered = results  # Включаем все объекты, даже с ошибками
                if only_profitable:
                    filtered = [r for r in filtered if r.get('should_split')]
                if selected_districts:
                    filtered = [r for r in filtered if r.get('district') in selected_districts]
                # Фильтруем по минимальной прибыли только если слайдер > 0
                if min_profit_pct > 0:
                    filtered = [r for r in filtered if 'error' in r or r.get('profit_increase_pct', 0) >= min_profit_pct]
                # Сортируем (объекты с ошибками в конец)
                filtered = sorted(filtered, key=lambda r: (0 if 'error' in r else 1, r.get(sort_by, 0)), reverse=True)

                st.caption(f"Показано: {len(filtered)} объектов")

                # Таблица
                table_rows = []
                for r in filtered:
                    # Если есть ошибка, показываем её
                    if 'error' in r:
                        table_rows.append({
                            'Площадь (м²)': int(r.get('original_square', 0)),
                            'Цена/м² реальная': '❌ Ошибка',
                            'Цена/м² опт': '-',
                            'Маржинальность (%)': '-',
                            'Реальная общая': format_currency(r.get('original_price', 0)),
                            'Оптимальная цена': '-',
                            'Прибыль (руб)': '-',
                            'Частей': '-',
                            'Район': r.get('district', ''),
                            'Тип': r.get('object_type', ''),
                            'Разбить': False,
                        })
                    else:
                        original_square = r.get('original_square', 0)
                        original_price = r.get('original_price', 0)  # Уже полная цена!
                        optimal_price = r.get('optimal_price', 0)

                        # Вычисляем цены за м²
                        price_per_sqm = original_price / original_square if original_square > 0 else 0
                        optimal_price_per_sqm = optimal_price / original_square if original_square > 0 else 0
                        margin_pct = ((optimal_price - original_price) / original_price * 100) if original_price > 0 else 0

                        table_rows.append({
                            'Площадь (м²)': int(original_square),
                            'Цена/м² реальная': f"₽{price_per_sqm:,.0f}",
                            'Цена/м² опт': f"₽{optimal_price_per_sqm:,.0f}",
                            'Маржинальность (%)': f"{margin_pct:.1f}%",
                            'Реальная общая': format_currency(original_price),
                            'Оптимальная цена': format_currency(optimal_price),
                            'Прибыль (руб)': format_currency(r.get('profit_increase', 0)),
                            'Частей': r.get('num_parts', 1),
                            'Район': r.get('district', ''),
                            'Тип': r.get('object_type', ''),
                            'Разбить': r.get('should_split', False),
                        })

                df_table = pd.DataFrame(table_rows)

                st.dataframe(
                    df_table,
                    use_container_width=True,
                    height=420,
                    column_config={
                        'Разбить': st.column_config.CheckboxColumn('Разбить?'),
                    }
                )

                # Детали разбиения
                st.subheader("🔍 Детали разбиения")
                to_show = [r for r in filtered if r.get('should_split') and r.get('parts')][:20]
                for r in to_show:
                    profit = r.get('profit_increase', 0)
                    profit_rounded = round(profit / 1000) * 1000
                    label = f"{r.get('district','?')} | {r.get('object_type','?')} | {r.get('original_square',0):.0f}м² | +₽{profit_rounded:,.0f} ({r.get('profit_increase_pct',0):.1f}%)"
                    with st.expander(label):
                        parts_df = pd.DataFrame(r['parts'])
                        parts_df.columns = ['№', 'Площадь (м²)', 'Цена (₽)', 'Цена за м² (₽)']
                        st.dataframe(parts_df, use_container_width=True)
                        if r.get('link'):
                            st.markdown(f"[Открыть объявление]({r['link']})")

                # Экспорт
                st.divider()
                export_data = json.dumps(filtered, ensure_ascii=False, indent=2)
                st.download_button(
                    "📥 Скачать результаты JSON",
                    data=export_data,
                    file_name="analytics_results.json",
                    mime="application/json"
                )
                csv_data = df_table.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "📥 Скачать таблицу CSV",
                    data=csv_data,
                    file_name="analytics_results.csv",
                    mime="text/csv"
                )

# ============================================
# Вкладка 5: Конвертер Excel → JSON
# ============================================

elif view_mode == "📥 Конвертер Excel":
    st.title("📥 Конвертер Excel → JSON")
    st.markdown("Загрузи Excel файл и получи готовый JSON для модели")

    uploaded_file = st.file_uploader(
        "📁 Выбери Excel файл",
        type=["xlsx", "xls"],
        help="Файлы prodam.xlsx, sdam.xlsx или другие"
    )

    if uploaded_file is not None:
        st.success(f"✅ Загружен: **{uploaded_file.name}**")

        if st.button("🔄 Преобразовать", type="primary"):
            with st.spinner("⏳ Преобразую..."):
                try:
                    # Read Excel
                    df = pd.read_excel(uploaded_file)
                    objects = df.to_dict('records')

                    # Format for model
                    formatted_data = format_json_for_model(objects)

                    # Convert to JSON string
                    json_str = json.dumps(formatted_data, ensure_ascii=False, indent=2)

                    st.success(f"✅ Готово! {len(formatted_data)} объектов обработано")

                    # Download button
                    st.download_button(
                        label="📥 Скачать JSON",
                        data=json_str,
                        file_name=f"{Path(uploaded_file.name).stem}_model_ready.json",
                        mime="application/json"
                    )

                    # Show sample
                    st.divider()
                    st.subheader("📊 Пример первого объекта")
                    st.json(formatted_data[0], expanded=False)

                except Exception as e:
                    st.error(f"❌ Ошибка: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
