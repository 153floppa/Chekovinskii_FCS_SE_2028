"""
AI Analysis Microservice
Extracts analyze_object.py into independent REST service
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.analytics.analyze_object import analyze_object

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Property Analysis Service",
    description="AI-powered analysis for real estate objects",
    version="1.0.0"
)

# ============ SCHEMAS ============

class PropertyObject(BaseModel):
    """Real estate object for analysis"""
    Адрес: Optional[str] = Field(None, description="Address")
    Район: Optional[str] = Field(None, description="District name")
    Вид_объекта: Optional[str] = Field(None, description="Object type")
    Площадь: Optional[float] = Field(None, description="Area in m²")
    Общая_площадь: Optional[float] = Field(None, description="Total area in m²")
    Цена: Optional[float] = Field(None, description="Current price per m²")
    Этаж: Optional[int] = Field(None, description="Floor number")
    Этажность_здания: Optional[int] = Field(None, description="Building floors")

    # POI fields (example)
    Школа_MIN: Optional[float] = Field(None, description="Min distance to school (m)")
    Школа500: Optional[int] = Field(None, description="Schools within 500m")
    Квартир500: Optional[int] = Field(None, description="Households within 500m")

    class Config:
        extra = "allow"  # Allow additional fields from JSON


class AnalysisRequest(BaseModel):
    """Request for property analysis"""
    object: Dict[str, Any] = Field(..., description="Property to analyze")
    predicted_price: float = Field(..., description="Predicted fair price per m²")
    real_price: float = Field(..., description="Current price per m²")
    margin_pct: float = Field(..., description="Profit margin %")
    model: str = Field("gpt-4o-mini", description="OpenAI model to use")
    max_tokens: int = Field(8000, description="Max tokens for response")


class AnalysisResponse(BaseModel):
    """Response with analysis"""
    status: str = Field("success", description="Response status")
    analysis: str = Field(..., description="AI-generated analysis in markdown")
    processing_time_ms: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response"""
    status: str = "error"
    error_code: str
    message: str


# ============ ENDPOINTS ============

@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    summary="Analyze real estate property",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "API error"},
    }
)
async def analyze_property(request: AnalysisRequest):
    """
    Analyze a real estate property using AI.

    Returns investment potential, risk factors, and recommendations.
    """
    import time
    start_time = time.time()

    try:
        logger.info(f"Analyzing property: {request.object.get('Адрес', 'Unknown')}")

        # Test mode: if PROXYAPI_KEY not set, return mock analysis
        if not os.getenv("PROXYAPI_KEY"):
            logger.warning("PROXYAPI_KEY not set - returning test analysis")
            test_analysis = f"""# Объект: {request.object.get('Адрес', 'тестовый')}, {request.object.get('Район', 'тест')}, {request.object.get('Общая площадь', 'N/A')} м²

## Краткий вердикт
🟡 СРЕДНЕ - Объект имеет потенциал, но требует тщательного анализа.
Потенциальная прибыль {request.margin_pct:.1f}% говорит о возможной недооценке.

**ВНИМАНИЕ: Это тестовый анализ (без API key). Используй реальный PROXYAPI_KEY для полного анализа.**

## Ключевые цифры
- Площадь: {request.object.get('Общая площадь', 'N/A')} м²
- Реальная цена: ₽{request.real_price:,.0f}
- Предсказанная цена: ₽{request.predicted_price:,.0f}
- Потенциальная недооценка: {request.margin_pct:.1f}%

## Разбор по факторам
### 1. Ценовая позиция [🟡]
Потенциальная недооценка составляет {request.margin_pct:.1f}%, что требует проверки.

### 2. Спрос vs Предложение [🟡]
Требуется детальный анализ местности для подтверждения спроса.

### 3. Риски [🟡]
- Малые объемы данных для глубокого анализа
- Требуется валидация через реальный OpenAI API
- Рекомендуется осмотр объекта

## Статус
🔴 MOCK MODE: Требуется PROXYAPI_KEY для полного анализа.
Используй: `export PROXYAPI_KEY='your_key'`
"""
            processing_time = int((time.time() - start_time) * 1000)
            return AnalysisResponse(
                status="success",
                analysis=test_analysis,
                processing_time_ms=processing_time
            )

        # Call the original analyze_object function
        analysis = analyze_object(
            full_object=request.object,
            predicted_price=request.predicted_price,
            real_price=request.real_price,
            margin_pct=request.margin_pct,
            model=request.model,
            max_tokens=request.max_tokens
        )

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(f"Analysis completed in {processing_time}ms")

        return AnalysisResponse(
            status="success",
            analysis=analysis,
            processing_time_ms=processing_time
        )

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "message": str(e)
            }
        )
    except Exception as e:
        error_str = str(e)
        # Check for common errors
        if "401" in error_str or "Invalid API Key" in error_str:
            logger.error(f"API Key error: {e}")
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "error_code": "API_KEY_ERROR",
                    "message": "PROXYAPI_KEY not set or invalid. Set it: export PROXYAPI_KEY='your_key'"
                }
            )
        elif "PROXYAPI_KEY" in error_str:
            logger.error(f"Missing API Key: {e}")
            raise HTTPException(
                status_code=401,
                detail={
                    "status": "error",
                    "error_code": "API_KEY_MISSING",
                    "message": "PROXYAPI_KEY environment variable not set. Please configure it before running analysis."
                }
            )
        else:
            logger.error(f"Analysis error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "error_code": "ANALYSIS_ERROR",
                    "message": f"Failed to analyze property: {str(e)}"
                }
            )


@app.get("/health", summary="Health check")
async def health_check():
    """Check if service is running"""
    return {
        "status": "healthy",
        "service": "Property Analysis",
        "version": "1.0.0"
    }


@app.get("/api/v1/models", summary="List available models")
async def list_models():
    """List available OpenAI models"""
    return {
        "available_models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-5.5"
        ],
        "default": "gpt-4o-mini"
    }


# ============ STARTUP/SHUTDOWN ============

@app.on_event("startup")
async def startup():
    logger.info("🚀 Property Analysis Service starting...")
    logger.info(f"API Key present: {'PROXYAPI_KEY' in os.environ}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("🛑 Property Analysis Service shutting down...")


# ============ RUN ============

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ANALYSIS_SERVICE_PORT", "8001"))

    logger.info(f"Starting service on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
