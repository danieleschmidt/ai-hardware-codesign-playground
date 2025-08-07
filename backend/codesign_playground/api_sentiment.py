"""
FastAPI routes for sentiment analysis.

Provides REST API endpoints for sentiment analysis functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import io
import csv
from datetime import datetime

from .sentiment_analyzer import SentimentAnalyzerAPI, SentimentResult, SentimentLabel
from .utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/sentiment", tags=["sentiment-analysis"])

# Global analyzer instance (singleton pattern)
_analyzer_instance: Optional[SentimentAnalyzerAPI] = None


def get_analyzer() -> SentimentAnalyzerAPI:
    """Get or create sentiment analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzerAPI()
        logger.info("Initialized sentiment analyzer")
    return _analyzer_instance


# Pydantic models for request/response
class TextAnalysisRequest(BaseModel):
    """Request model for single text analysis."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    include_scores: bool = Field(True, description="Include detailed sentiment scores")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I absolutely love this product! It's amazing!",
                "include_scores": True
            }
        }


class BatchAnalysisRequest(BaseModel):
    """Request model for batch text analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to analyze")
    include_scores: bool = Field(True, description="Include detailed sentiment scores")
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I love this!",
                    "This is terrible.",
                    "It's okay."
                ],
                "include_scores": True
            }
        }


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    text: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: Optional[Dict[str, float]] = None
    processing_time_ms: float
    timestamp: datetime
    
    @classmethod
    def from_result(cls, result: SentimentResult, include_scores: bool = True) -> "SentimentResponse":
        """Create response from SentimentResult."""
        return cls(
            text=result.text,
            label=result.label.value,
            confidence=result.confidence,
            scores=result.scores if include_scores else None,
            processing_time_ms=result.processing_time_ms,
            timestamp=result.timestamp
        )


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment analysis."""
    results: List[SentimentResponse]
    total_count: int
    processing_time_ms: float
    summary: Dict[str, Any]
    
    @classmethod
    def from_results(cls, results: List[SentimentResult], include_scores: bool = True) -> "BatchSentimentResponse":
        """Create batch response from list of SentimentResults."""
        start_time = min(r.timestamp for r in results) if results else datetime.now()
        end_time = max(r.timestamp for r in results) if results else datetime.now()
        total_time = (end_time - start_time).total_seconds() * 1000 + sum(r.processing_time_ms for r in results)
        
        # Calculate summary statistics
        positive_count = sum(1 for r in results if r.label == SentimentLabel.POSITIVE)
        negative_count = sum(1 for r in results if r.label == SentimentLabel.NEGATIVE)
        neutral_count = sum(1 for r in results if r.label == SentimentLabel.NEUTRAL)
        avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        
        summary = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_percentage": (positive_count / len(results)) * 100 if results else 0,
            "negative_percentage": (negative_count / len(results)) * 100 if results else 0,
            "neutral_percentage": (neutral_count / len(results)) * 100 if results else 0,
            "average_confidence": avg_confidence,
            "average_processing_time_ms": sum(r.processing_time_ms for r in results) / len(results) if results else 0
        }
        
        return cls(
            results=[SentimentResponse.from_result(r, include_scores) for r in results],
            total_count=len(results),
            processing_time_ms=total_time,
            summary=summary
        )


class AnalyzerStatsResponse(BaseModel):
    """Response model for analyzer statistics."""
    analyzer_type: str
    positive_words_count: int
    negative_words_count: int
    intensifiers_count: int
    negations_count: int
    version: str = "1.0.0"
    
    class Config:
        schema_extra = {
            "example": {
                "analyzer_type": "SimpleSentimentAnalyzer",
                "positive_words_count": 32,
                "negative_words_count": 33,
                "intensifiers_count": 9,
                "negations_count": 7,
                "version": "1.0.0"
            }
        }


# API Endpoints

@router.post("/analyze", response_model=SentimentResponse, summary="Analyze Single Text")
async def analyze_text(
    request: TextAnalysisRequest,
    analyzer: SentimentAnalyzerAPI = Depends(get_analyzer)
) -> SentimentResponse:
    """
    Analyze sentiment of a single text.
    
    - **text**: The text to analyze (required)
    - **include_scores**: Whether to include detailed sentiment scores
    
    Returns sentiment label, confidence score, and processing metrics.
    """
    try:
        result = analyzer.analyze_text(request.text)
        logger.info(f"Analyzed text sentiment: {result.label.value} (confidence: {result.confidence:.3f})")
        return SentimentResponse.from_result(result, request.include_scores)
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch", response_model=BatchSentimentResponse, summary="Analyze Multiple Texts")
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    analyzer: SentimentAnalyzerAPI = Depends(get_analyzer)
) -> BatchSentimentResponse:
    """
    Analyze sentiment of multiple texts in batch.
    
    - **texts**: List of texts to analyze (max 1000)
    - **include_scores**: Whether to include detailed sentiment scores
    
    Returns analysis results for all texts with summary statistics.
    """
    try:
        if len(request.texts) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 texts allowed per batch")
        
        results = analyzer.analyze_batch(request.texts)
        
        # Log batch analysis in background
        background_tasks.add_task(
            _log_batch_analysis, 
            len(request.texts), 
            results
        )
        
        return BatchSentimentResponse.from_results(results, request.include_scores)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


@router.get("/stats", response_model=AnalyzerStatsResponse, summary="Get Analyzer Statistics")
async def get_analyzer_stats(
    analyzer: SentimentAnalyzerAPI = Depends(get_analyzer)
) -> AnalyzerStatsResponse:
    """
    Get sentiment analyzer statistics and configuration.
    
    Returns information about the analyzer including word counts and version.
    """
    try:
        stats = analyzer.get_stats()
        return AnalyzerStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting analyzer stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/analyze/stream", summary="Streaming Analysis")
async def analyze_stream(
    request: BatchAnalysisRequest,
    analyzer: SentimentAnalyzerAPI = Depends(get_analyzer)
) -> StreamingResponse:
    """
    Stream sentiment analysis results as they are processed.
    
    Returns JSON Lines format with one result per line.
    """
    try:
        def generate_results():
            for text in request.texts:
                try:
                    result = analyzer.analyze_text(text)
                    response = SentimentResponse.from_result(result, request.include_scores)
                    yield json.dumps(response.dict()) + "\n"
                except Exception as e:
                    error_response = {
                        "text": text,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield json.dumps(error_response) + "\n"
        
        return StreamingResponse(
            generate_results(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=sentiment_results.jsonl"}
        )
    except Exception as e:
        logger.error(f"Error in streaming analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming analysis failed: {str(e)}")


@router.post("/export/csv", summary="Export Results to CSV")
async def export_csv(
    request: BatchAnalysisRequest,
    analyzer: SentimentAnalyzerAPI = Depends(get_analyzer)
) -> StreamingResponse:
    """
    Analyze texts and export results as CSV file.
    
    Returns CSV file with sentiment analysis results.
    """
    try:
        results = analyzer.analyze_batch(request.texts)
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = ["text", "label", "confidence", "processing_time_ms", "timestamp"]
        if request.include_scores:
            headers.extend(["positive_score", "negative_score", "neutral_score"])
        writer.writerow(headers)
        
        # Write data
        for result in results:
            row = [
                result.text,
                result.label.value,
                f"{result.confidence:.3f}",
                f"{result.processing_time_ms:.2f}",
                result.timestamp.isoformat()
            ]
            if request.include_scores:
                row.extend([
                    f"{result.scores['positive']:.3f}",
                    f"{result.scores['negative']:.3f}",
                    f"{result.scores['neutral']:.3f}"
                ])
            writer.writerow(row)
        
        # Create response
        output.seek(0)
        return StreamingResponse(
            io.StringIO(output.getvalue()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentiment_analysis.csv"}
        )
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(e)}")


@router.get("/health", summary="Health Check")
async def health_check() -> Dict[str, str]:
    """
    Check if sentiment analysis service is healthy.
    
    Returns service status and version information.
    """
    try:
        # Test analyzer initialization
        analyzer = get_analyzer()
        
        # Quick test analysis
        test_result = analyzer.analyze_text("test")
        
        return {
            "status": "healthy",
            "service": "sentiment-analysis",
            "version": "1.0.0",
            "analyzer_type": "SimpleSentimentAnalyzer",
            "test_processing_time_ms": f"{test_result.processing_time_ms:.2f}"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Background task functions
def _log_batch_analysis(batch_size: int, results: List[SentimentResult]) -> None:
    """Log batch analysis statistics in background."""
    try:
        positive_count = sum(1 for r in results if r.label == SentimentLabel.POSITIVE)
        negative_count = sum(1 for r in results if r.label == SentimentLabel.NEGATIVE)
        neutral_count = sum(1 for r in results if r.label == SentimentLabel.NEUTRAL)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        total_time = sum(r.processing_time_ms for r in results)
        
        logger.info(
            f"Batch analysis completed: {batch_size} texts, "
            f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral, "
            f"avg confidence: {avg_confidence:.3f}, total time: {total_time:.2f}ms"
        )
    except Exception as e:
        logger.error(f"Error logging batch analysis: {e}")
