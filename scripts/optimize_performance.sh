#!/bin/bash

# Performance optimization script for ML Classifier
echo "=== Performance Optimization Report ==="
echo "Date: $(date)"
echo ""

# Check system resources
echo "1. System Resources:"
echo "   Memory: $(free -h 2>/dev/null || echo 'N/A (macOS)')"
echo "   CPU: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'N/A')"
echo "   Disk: $(df -h . | tail -1 | awk '{print $4}' | sed 's/Available/Free/')"
echo ""

# Check Python environment
echo "2. Python Environment:"
echo "   Python version: $(uv run python --version 2>&1)"
echo "   Virtual env: ${VIRTUAL_ENV:-'Not activated'}"
echo "   UV version: $(uv --version 2>/dev/null || echo 'Not installed')"
echo ""

# Check model sizes
echo "3. Model Artifacts:"
if [ -d "models/artifacts" ]; then
    echo "   Model files:"
    ls -lh models/artifacts/*.pkl 2>/dev/null | awk '{print "     " $9 ": " $5}' || echo "     No models found"
else
    echo "   No models directory found"
fi
echo ""

# Check dependencies
echo "4. Key Dependencies:"
uv run python -c "
import sys
packages = ['pandas', 'numpy', 'sklearn', 'fastapi', 'uvicorn']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'   {pkg}: {version}')
    except ImportError:
        print(f'   {pkg}: not installed')
"
echo ""

# Performance recommendations
echo "5. Performance Recommendations:"
echo "   ✓ Use uv for fast dependency management"
echo "   ✓ Enable sklearn parallel processing with n_jobs=-1"
echo "   ✓ Use efficient data types (int32 vs int64 where possible)"
echo "   ✓ Cache preprocessed data for repeated training"
echo "   ✓ Use model registry to avoid reloading models"
echo "   ✓ Enable FastAPI async for I/O-bound operations"
echo ""

# Memory usage check
echo "6. Memory Usage Optimization:"
uv run python -c "
import psutil
import os
process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f'   Current process memory: {memory_mb:.1f} MB')
print('   Recommendations:')
print('     - Use pandas.read_csv with dtype specifications')
print('     - Clear intermediate variables with del')
print('     - Use sklearn partial_fit for large datasets')
print('     - Enable garbage collection for long-running processes')
"
echo ""

# Docker optimization
echo "7. Docker Optimization:"
echo "   ✓ Multi-stage builds to reduce image size"
echo "   ✓ .dockerignore to exclude unnecessary files"
echo "   ✓ Separate training and serving containers"
echo "   ✓ Use slim Python base images"
echo ""

echo "=== Optimization Complete ==="
