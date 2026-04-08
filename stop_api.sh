#!/bin/bash

echo "Stopping FastAPI app..."

pkill -f "uvicorn main:app" || true
