#!/bin/bash
# Shell script to run Classical ML Pipeline

echo "========================================"
echo "Classical ML Pipeline Runner"
echo "========================================"
echo

# Check if python is available
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python not found"
        echo "Please install Python or activate your virtual environment"
        exit 1
    fi
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

# Check if we're in the right directory
if [ ! -f "run.py" ]; then
    echo "ERROR: run.py not found"
    echo "Please run this script from src/classical_ml directory"
    exit 1
fi

echo "Running Classical ML Pipeline..."
echo

$PYTHON_CMD run.py

echo
echo "========================================"
echo "Pipeline execution completed"
echo "========================================"
