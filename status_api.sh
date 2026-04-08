#!/bin/bash

if pgrep -f "uvicorn main:app" > /dev/null; then
  echo "App is running"
  exit 0
else
  echo "App is not running"
  exit 1
fi
