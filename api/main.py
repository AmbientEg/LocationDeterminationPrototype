import os
import sys
import logging
from datetime import datetime
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from .websocket import router as websocket_router
from .logger_config import setup_logging


####### NOTE BY Rodyna : This file is a simplified example of a FastAPI WebSocket server
#######       with production-ready logging, error handling, and middleware.
#######       It does not include advanced features like authentication,
#######       rate limiting, or database integration for brevity.

# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------

logger = setup_logging()

# ----------------------------------------------------
# Lifespan for Startup & Shutdown
# ----------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services or resources on startup."""
    try:
        logger.info("🚀 WebSocket server starting up...")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("🛑 WebSocket server shutting down...")

# ----------------------------------------------------
# FastAPI App
# ----------------------------------------------------
app = FastAPI(
    title="WebSocket Service",
    description="Lightweight FastAPI WebSocket server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ----------------------------------------------------
# CORS Middleware
# ----------------------------------------------------
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Security Middleware
# ----------------------------------------------------
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# ----------------------------------------------------
# Request Logging Middleware
# ----------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.utcnow()
    correlation_id = f"req_{start_time.timestamp()}"
    request.state.correlation_id = correlation_id
    logger.info(f"Incoming request: {request.method} {request.url} [{correlation_id}]")
    try:
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Request completed in {duration:.3f}s [{correlation_id}]")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e} [{correlation_id}]", exc_info=True)
        raise



# ----------------------------------------------------
# Health Check & Root
# ----------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
async def root():
    return {
        "message": "WebSocket Server Running",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket_endpoint": "/ws",
    }

app.include_router(websocket_router)


# ----------------------------------------------------
# Run Server
# ----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")


