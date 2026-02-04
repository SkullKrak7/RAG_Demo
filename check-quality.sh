#!/bin/bash
# Quick local quality check before pushing

echo "🔍 Running local quality checks..."
echo ""

# Format
echo "1️⃣  Formatting code..."
black rag_demo/ tests/
echo "✅ Code formatted"
echo ""

# Tests
echo "2️⃣  Running tests..."
pytest tests/ -v --cov=rag_demo --cov-fail-under=90
if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi
echo "✅ Tests passed"
echo ""

# Lint
echo "3️⃣  Linting..."
pylint rag_demo/ --disable=C0114,C0115,C0116,R0913,R0914 --max-line-length=120 || true
echo "✅ Lint complete"
echo ""

# Security
echo "4️⃣  Security scan..."
bandit -r rag_demo/ -ll
if [ $? -ne 0 ]; then
    echo "❌ Security issues found"
    exit 1
fi
echo "✅ Security scan passed"
echo ""

echo "🎉 All checks passed! Safe to push."
