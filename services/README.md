# 🔧 Services - Микросервисы

Модуль для разделения функциональности приложения на независимые сервисы.

## 📁 Структура

```
services/
├── analysis_service.py   # Сервис анализа объектов
└── README.md
```

## 🎯 Архитектура микросервисов

Текущее приложение может быть разделено на 5 независимых сервисов:

### 1. 📊 Data Service (PORT 8001)
**Функция:** Загрузка и обработка данных
**Endpoints:**
```
POST /load-json          - загрузить JSON
POST /convert-excel      - конвертировать Excel
GET /data/{id}           - получить объект
```

### 2. 🤖 Prediction Service (PORT 8002)
**Функция:** Предсказание цен
**Endpoints:**
```
POST /predict            - получить предсказание
POST /predict-batch      - пакетные предсказания
GET /model-info          - информация о модели
```

### 3. 🔍 Analysis Service (PORT 8003)
**Функция:** Анализ объектов
**Endpoints:**
```
POST /analyze-object     - анализ от AI
POST /find-alternatives  - похожие объекты
POST /split-analysis     - анализ разбиений
```

### 4. 🏙️ Analytics Service (PORT 8004)
**Функция:** Визуализация и аналитика
**Endpoints:**
```
GET /price-distribution  - распределение цен
GET /feature-importance  - важность признаков
GET /market-segments     - сегментация рынка
GET /poi-impact          - влияние POI
```

### 5. 📥 Conversion Service (PORT 8005)
**Функция:** Конвертация форматов
**Endpoints:**
```
POST /excel-to-json      - Excel → JSON
POST /format-for-model   - JSON → Model-ready
POST /validate-data      - валидация данных
```

## 📋 Текущая реализация: Analysis Service

### Описание

Сервис для выполнения AI-анализа объектов с использованием OpenAI.

### Использование

```python
from services.analysis_service import AnalysisService

service = AnalysisService(api_key="your_api_key")

analysis = service.analyze_object({
    'Цена': 29000000,
    'Общая площадь': 149,
    'Район': 'Кировский',
    ...
})
```

### Функции

```python
def analyze_object(obj_data: dict) -> dict:
    """Анализ объекта через OpenAI"""
    
def get_ai_recommendation(analysis: dict) -> str:
    """Получить рекомендацию"""
```

## 🏗️ Миграция на микросервисы

### Этап 1: Выделение логики (DONE)
- ✅ Analysis Service выделена
- ✅ Separation of concerns
- ✅ Готово к deploy

### Этап 2: REST API (PLANNED)
- [ ] FastAPI endpoints
- [ ] Docker контейнеры
- [ ] Kubernetes orchestration

### Этап 3: Масштабирование (FUTURE)
- [ ] Load balancing
- [ ] Service mesh (Istio)
- [ ] Auto-scaling по нагрузке

## 🔄 API Contracts

### Prediction Service

**Request:**
```json
{
  "object": {
    "Цена": null,
    "Общая площадь": 149.0,
    "Район": "Кировский",
    ...
  }
}
```

**Response:**
```json
{
  "predicted_price_per_sqm": 194631,
  "predicted_total_price": 28960000,
  "confidence": 0.55,
  "model_version": "1.0"
}
```

### Analysis Service

**Request:**
```json
{
  "object": {...},
  "depth": "full"
}
```

**Response:**
```json
{
  "undervaluation_pct": 15,
  "key_factors": ["avg500", "Школа1000"],
  "potential": "medium",
  "recommendation": "BUY",
  "ai_analysis": "..."
}
```

## 🚀 Развертывание

### Локально (Development)

```bash
# Запустить текущее приложение
streamlit run interface/app.py

# В будущем - запустить микросервисы
docker-compose up -d
```

### В облаке (Production)

```bash
# Kubernetes deployment
kubectl apply -f k8s/

# Service endpoints
prediction.api.example.com
analysis.api.example.com
data.api.example.com
```

## 🔐 Безопасность

### API Authentication
```
Authorization: Bearer {JWT_TOKEN}
```

### Rate Limiting
```
100 запросов / минуту на юзера
1000 запросов / минуту на сервис
```

### Data Protection
```
- HTTPS only
- Input validation
- SQL injection prevention
- CORS rules
```

## 📊 Мониторинг

### Metrics

```
- Latency (p50, p95, p99)
- Error rate
- Throughput (req/sec)
- Cache hit rate
```

### Logging

```
- JSON format
- Structured logging
- ELK stack integration
```

### Alerting

```
- 99.9% uptime SLA
- Alert on errors > 1%
- Alert on latency p95 > 1000ms
```

## 🔗 Интеграция с основным приложением

**Сейчас (Monolith):**
```
Streamlit UI → interface/app.py → все модули
```

**После миграции (Microservices):**
```
Streamlit UI → API Gateway → Microservices
                               ├─ Prediction Service
                               ├─ Analysis Service
                               ├─ Data Service
                               ├─ Analytics Service
                               └─ Conversion Service
```

## 📚 Дополнительные ресурсы

- [Kubernetes docs](https://kubernetes.io/)
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [Docker Compose](https://docs.docker.com/compose/)

## 🔗 Связанные модули

- [Interface](../interface/README.md) - клиент сервисов
- [Model](../model/README.md) - используется в Prediction Service
- [Dataset](../dataset/README.md) - используется в Data Service

---

**Версия:** 0.5.0  
**Статус:** Design Phase → Implementation  
**Следующий шаг:** REST API endpoints
