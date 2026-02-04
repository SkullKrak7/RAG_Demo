#!/bin/bash
# Setup git hooks for the project

HOOKS_DIR=".git/hooks"
HOOK_FILE="$HOOKS_DIR/pre-commit"

cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash
# Pre-commit hook for code quality checks

set -e

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

echo "Running pre-commit checks..."

# Format check
echo "1. Checking code formatting..."
black --check rag_demo/ tests/ || {
    echo "❌ Code not formatted. Run: black rag_demo/ tests/"
    exit 1
}

# Tests
echo "2. Running tests..."
pytest tests/ -q --cov=rag_demo --cov-fail-under=90 || {
    echo "❌ Tests failed or coverage < 90%"
    exit 1
}

# Security scan
echo "3. Running security scan..."
bandit -r rag_demo/ -ll -q || {
    echo "❌ Security issues found"
    exit 1
}

echo "✅ All pre-commit checks passed!"
EOF

chmod +x "$HOOK_FILE"

echo "✅ Git hooks installed successfully!"
echo ""
echo "Pre-commit hook will run:"
echo "  - Code formatting check"
echo "  - Full test suite with coverage"
echo "  - Security scan"
echo ""
echo "To skip hooks (not recommended): git commit --no-verify"
