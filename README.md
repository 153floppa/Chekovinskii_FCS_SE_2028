# 🏢 ML Аналитика Коммерческой Недвижимости

Приложение для анализа и прогнозирования цен коммерческой недвижимости на основе данных Авито.

## 📋 Содержание

- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Возможности](#возможности)
- [Требования](#требования)
- [Использование](#использование)

## 🚀 Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения
streamlit run interface/app.py
```

Приложение откроется по адресу: http://localhost:8501

---

## 📖 О приложении

### Суть проекта

**Это приложение помогает найти выгодные сделки с коммерческой недвижимостью.**

Коммерческая недвижимость часто неправильно оценивается на рынке. Объект может стоить дешевле, чем он "должен" стоить, исходя из параметров соседних объектов, инфраструктуры и других факторов. 

**Наша система:**
1. 🎯 **Находит недооцененные объекты** - объекты, которые продают дешевле, чем они стоят
2. 🤖 **Предсказывает цену** - использует ML модель для расчета "правильной" цены
3. 💰 **Предлагает способ заработать** - показывает, как разбить большое помещение на несколько меньших для большей прибыли
4. 🧠 **Дает AI анализ** - объясняет через ChatGPT, почему объект дешевый или дорогой

### Практическое применение

#### 1️⃣ Для инвесторов в недвижимость
```
Инвестор получает:
- Список недооцененных объектов в интересующем районе
- Прогноз справедливой цены
- Рекомендацию: купить сейчас или ждать падения
- Расчет ROI при разбиении помещения
```

#### 2️⃣ Для риэлторов и брокеров
```
Риэлтор получает:
- Обоснование цены клиентам (на основе ML анализа)
- Список похожих объектов для сравнения
- Рекомендации для переговоров
- Быстрый анализ вместо ручных расчетов
```

#### 3️⃣ Для девелоперов
```
Девелопер получает:
- Анализ рыночных цен в районе
- Понимание спроса и сегментации
- Рекомендации по разбиению помещений
- Прогноз стоимости после разработки
```

### Основной workflow

```
1. Пользователь загружает JSON с объектами
          ↓
2. Система анализирует данные и обучается на примерах
          ↓
3. ML модель предсказывает "правильную" цену каждого объекта
          ↓
4. Система находит недооцененные (цена ниже прогноза)
          ↓
5. AI (ChatGPT) анализирует, почему цена низкая
          ↓
6. Система рассчитывает выгодность разбиения помещения
          ↓
7. Пользователь видит список возможностей и делает выводы
```

---

## 📁 Структура проекта

```
ML-main/
├── interface/              # 🎨 Streamlit UI приложение
├── model/                  # 🤖 ML модели и предсказания
├── dataset/                # 📊 Работа с данными
├── analytics/              # 📈 Анализ и визуализация
├── services/               # 🔧 Микросервисы
├── config.py               # ⚙️ Конфигурация
└── requirements.txt        # 📦 Зависимости
```

## ✨ Возможности

### 🎯 Топ недооцененных объектов
- Загрузка JSON с данными
- Анализ и ранжирование по недооценке
- Экспорт результатов (CSV/JSON)

### 📊 Модель и метрики
- CatBoost регрессор для предсказания цен
- Метрики качества (R², MAPE, MAE, RMSE)
- Анализ важности признаков (top-20)

### 🔍 Аналитика объекта
- Детальный анализ отдельного объекта
- AI-powered рекомендации (OpenAI)
- История анализов

### 📂 Загрузка аналитики
- Импорт данных для анализа
- Расчет оптимальных разбиений
- Анализ профитабельности

### 📥 Конвертер Excel
- Загрузка Excel файлов (prodam.xlsx, sdam.xlsx)
- Автоматическое преобразование в JSON
- Готовый формат для модели

## 📦 Требования

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- catboost >= 1.2.0
- scikit-learn >= 1.3.0
- streamlit >= 1.28.0
- plotly >= 5.17.0
- openai >= 1.0.0

## 📖 Использование

### Вкладка 1: Топ недооцененных
1. Загрузи JSON файл с данными
2. Выбери объект из таблицы
3. Посмотри недооценку и метрики
4. Экспортируй результаты

### Вкладка 2: Модель и метрики
- Посмотри качество модели
- Анализируй важные признаки
- Сравни предсказания с реальными ценами

### Вкладка 3: Аналитика объекта
1. Загрузи объект
2. Получи анализ от AI
3. Посмотри ключевые факторы
4. Получи рекомендацию

### Вкладка 4: Загрузка аналитики
1. Загрузи данные
2. Расчитай разбиения
3. Фильтруй по профитабельности
4. Экспортируй результаты

### Вкладка 5: Конвертер Excel
1. Загрузи Excel (xlsx/xls)
2. Нажми "Преобразовать"
3. Скачай готовый JSON

## 🏗️ Архитектура

### Текущая архитектура: Layered Monolith

```
┌─────────────────────────────────────────┐
│     Presentation Layer                   │
│  (Streamlit UI: app.py)                 │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     Service Layer                        │
│  (Business Logic: analysis_service.py)  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     Domain Layer                         │
│  (Models: catboost_regressor.py)        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     Data Access Layer                    │
│  (Utils: excel_handler, fix_json)       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│     Infrastructure Layer                 │
│  (File Storage, External APIs)          │
└─────────────────────────────────────────┘
```

### Преимущества текущей архитектуры
- ✅ Простота разработки и развертывания
- ✅ Быстрая разработка фич
- ✅ Легкий отладка в монолитной среде
- ✅ Хорошо для MVP и малых команд

### Готовность к микросервисам
Архитектура позволяет легко разделить на 5 независимых сервисов!

---

## 🔌 Микросервисы (Roadmap)

### Текущее состояние: Layered Monolith
Все функции в одном приложении Streamlit.

### Будущая архитектура: Microservices

```
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │  (Frontend)          │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   API Gateway        │
                    │  (Request Routing)   │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┬──────────────────┐
        │                      │                      │                  │
    ┌───▼────┐         ┌──────▼─────┐        ┌──────▼──────┐       ┌───▼────┐
    │ Data   │         │Prediction  │        │ Analysis    │       │ Analyt │
    │Service │         │Service     │        │ Service     │       │ Service│
    │(8001) │         │(8002)      │        │ (8003)      │       │(8004) │
    └────────┘         └────────────┘        └─────────────┘       └────────┘
        │                  │                      │
    ├───┘                  └──────────────────────┴──────────────┐
    │                                                            │
┌───▼──────────────────────────────────────────────────────────▼──┐
│         Shared Infrastructure                                   │
│ - Message Queue (RabbitMQ)                                     │
│ - Cache (Redis)                                                │
│ - Database (PostgreSQL)                                        │
│ - Monitoring (Prometheus + Grafana)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 5 Независимых сервисов

#### 1. 📥 Data Service (PORT 8001)
**Ответственность:** Загрузка и преобразование данных

```python
# Endpoints
POST   /load-json              # Загрузить JSON файл
POST   /convert-excel          # Excel → JSON
POST   /format-for-model       # Форматирование для модели
GET    /data/{object_id}       # Получить объект
DELETE /data/{object_id}       # Удалить объект
```

**Используемые модули:**
- `dataset/collection/avito/fix_json_format.py`
- `dataset/excel_handler.py`

#### 2. 🤖 Prediction Service (PORT 8002)
**Ответственность:** ML предсказания и оценки

```python
# Endpoints
POST   /predict                # Предсказание одного объекта
POST   /predict-batch          # Пакетные предсказания
GET    /model-info             # Информация о модели
GET    /feature-importance     # Важность признаков
```

**Используемые модули:**
- `model/regressors/catboost_regressor.py`
- `model/splitting/generate_split.py`

#### 3. 🔍 Analysis Service (PORT 8003)
**Ответственность:** Интеллектуальный анализ через AI

```python
# Endpoints
POST   /analyze-object         # Анализ объекта от OpenAI
POST   /find-alternatives      # Поиск похожих объектов
POST   /split-analysis         # Анализ разбиений
GET    /analysis/{id}          # История анализов
```

**Используемые модули:**
- `services/analysis_service.py`
- OpenAI API integration

#### 4. 📊 Analytics Service (PORT 8004)
**Ответственность:** Визуализация и аналитика

```python
# Endpoints
GET    /price-distribution     # Распределение цен
GET    /feature-correlation    # Корреляция признаков
GET    /market-segments        # Сегментация рынка
GET    /poi-impact             # Влияние POI
GET    /undervalued-objects    # Недооцененные объекты
```

**Используемые модули:**
- `analytics/explore_data.py`
- Plotly charts

#### 5. 🔄 Conversion Service (PORT 8005)
**Ответственность:** Конвертация и валидация форматов

```python
# Endpoints
POST   /excel-to-json          # Excel → JSON
POST   /validate-data          # Валидация данных
POST   /clean-data             # Очистка данных
POST   /enrich-data            # Обогащение данных
```

**Используемые модули:**
- `dataset/collection/avito/fix_json_format.py`
- `dataset/collection/enrich_from_excel.py`

### Миграционный путь

```
Фаза 1: Текущее состояние (DONE ✅)
├─ Layered Monolith в Streamlit
├─ Все функции в одном приложении
└─ Production Ready для MVP

Фаза 2: Service Extraction (PLANNED)
├─ Выделить бизнес логику в services/
├─ Создать REST API endpoints
└─ Docker контейнеризация

Фаза 3: API Gateway (PLANNED)
├─ FastAPI gateway для маршрутизации
├─ Load balancing
└─ Rate limiting

Фаза 4: Full Microservices (FUTURE)
├─ Независимые сервисы в разных контейнерах
├─ Kubernetes оркестрация
├─ Service mesh (Istio)
└─ Auto-scaling по нагрузке
```

---

## 🎯 Design Patterns (Паттерны проектирования)

### 1. Singleton Pattern
**Где используется:** Streamlit кэширование

```python
@st.cache_resource
def load_model():
    """Загружается один раз при старте приложения"""
    return get_model('catboost_regressor')

# Модель загружается один раз и переиспользуется
model = load_model()
```

**Преимущество:** Быстрая работа, экономия памяти

---

### 2. Factory Pattern
**Где используется:** Создание моделей

```python
from model.regressors.catboost_regressor import get_model

# Фабрика возвращает нужную модель по имени
model = get_model('catboost_regressor')
# В будущем:
# model = get_model('xgboost_regressor')
# model = get_model('lightgbm_regressor')
```

**Преимущество:** Легко добавлять новые модели без изменения кода

---

### 3. Pipeline Pattern
**Где используется:** Обработка данных

```
Excel файл
   ↓
[Парсинг]       (parse_excel.py)
   ↓
[Нормализация]  (process_prodam.py)
   ↓
[Обогащение]    (enrich_from_excel.py)
   ↓
[Форматирование](fix_json_format.py)
   ↓
JSON для модели
```

**Код:**
```python
# Pipeline в convert/fix_json_format.py
for idx, obj in enumerate(data):
    fixed_obj = fix_object(obj)  # Преобразование
    formatted_data.append(fixed_obj)
```

**Преимущество:** Модульность, переиспользуемость

---

### 4. Decorator Pattern
**Где используется:** Streamlit кэширование

```python
@st.cache_data
def load_data():
    """Кэшируется автоматически"""
    return json.load(open('dataset.json'))

@st.cache_resource
def load_model():
    """Кэшируется на весь сеанс"""
    return get_model('catboost')
```

**Преимущество:** Упрощение кода, автоматическое кэширование

---

### 5. Strategy Pattern
**Где используется:** Разные алгоритмы предсказания

```python
# В будущем:
predictor = ModelFactory.create('catboost')
# или
predictor = ModelFactory.create('xgboost')
# или
predictor = ModelFactory.create('ensemble')

prediction = predictor.predict(X)
```

**Текущее:** CatBoost используется напрямую

---

### 6. Repository Pattern
**Где используется:** Работа с данными

```python
# В будущем:
class DataRepository:
    def load_json(path): ...
    def load_pickle(path): ...
    def load_parquet(path): ...

# Унифицированный интерфейс
data = repository.load('data.json')
data = repository.load('data.pkl')
```

**Текущее:** Прямая работа с файлами в utils

---

### 7. MVC Pattern
**Где используется:** Streamlit приложение

```
Model         → model/ (CatBoost, логика предсказания)
View          → interface/ (UI, отображение)
Controller    → interface/app.py (логика взаимодействия)
```

**Архитектура:**
```
interface/app.py
├─ Model слой (load_model, predict)
├─ View слой (st.title, st.dataframe)
└─ Controller (обработка кнопок, фильтры)
```

---

### 8. Chain of Responsibility
**Где используется:** Обработка ошибок в конвертации

```python
try:
    # Шаг 1
    df = pd.read_excel(file)
except:
    st.error("Ошибка чтения Excel")
    return

try:
    # Шаг 2
    data = df.to_dict('records')
except:
    st.error("Ошибка парсинга")
    return

try:
    # Шаг 3
    formatted = format_json_for_model(data)
except:
    st.error("Ошибка форматирования")
    return
```

**Преимущество:** Гибкая обработка ошибок на каждом этапе

---

## 🔄 Design Principles

### SOLID принципы

#### Single Responsibility (SRP)
```python
# ✅ ХОРОШО
class PricePredictor:
    def predict(self, obj) -> float: ...

class DataFormatter:
    def format(self, obj) -> dict: ...

# ❌ ПЛОХО
class UtilityClass:
    def predict(self, obj): ...
    def format(self, obj): ...
    def analyze(self, obj): ...
```

#### Open/Closed (OCP)
```python
# ✅ ХОРОШО - можно добавить новую модель без изменения кода
ModelFactory.create('new_model')

# ❌ ПЛОХО - надо менять код при добавлении модели
if model_type == 'catboost':
    model = CatBoost()
elif model_type == 'xgboost':
    model = XGBoost()
```

#### Dependency Inversion (DIP)
```python
# ✅ ХОРОШО - зависит от абстракции
predictor = ModelFactory.create(model_type)

# ❌ ПЛОХО - прямая зависимость
from model.catboost_regressor import CatBoost
model = CatBoost()
```

---

## 🔧 Конфигурация

Главные параметры в `config.py`:
- `PRICE_COL` - колонка цены
- `SQUARE_COL` - колонка площади

### Планирование микросервисов
Для развертывания микросервисов потребуется:
- Docker & Docker Compose
- API Gateway (FastAPI)
- Message Queue (RabbitMQ)
- Cache (Redis)
- Monitoring (Prometheus)

---

## 📚 Дополнительно

- [📊 Dataset](dataset/README.md) - работа с данными
- [🎨 Interface](interface/README.md) - описание UI
- [🤖 Model](model/README.md) - модели и алгоритмы
- [📈 Analytics](analytics/README.md) - анализ и выводы
- [🔧 Services](services/README.md) - микросервисы

---

**Версия:** 1.0.0  
**Статус:** ✅ Production Ready
