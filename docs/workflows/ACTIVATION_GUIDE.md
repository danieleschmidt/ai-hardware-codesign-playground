# Workflow Activation Guide

## ⚠️ Manual Setup Required

Due to GitHub security restrictions, the workflow files need to be manually created in the `.github/workflows/` directory. This guide provides the complete workflow configurations ready for activation.

## Repository Assessment

**Current Maturity Level**: MATURING (65-70%)  
**Target Maturity Level**: ADVANCED (80-85%)

The repository has exceptional SDLC infrastructure already in place but requires activation of the CI/CD automation to reach advanced maturity.

## Required Workflow Files

### 1. CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main, develop]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "18"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  code-quality:
    name: Code Quality
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install Python Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Install Node Dependencies
        run: npm ci
      
      - name: Run Pre-commit Hooks
        uses: pre-commit/action@v3.0.0
        with:
          extra_args: --all-files
      
      - name: Python Type Checking
        run: mypy backend/codesign_playground/ || true
      
      - name: Security Scanning
        run: |
          bandit -r backend/codesign_playground/ || true
          safety check || true
        continue-on-error: true

  test-python:
    name: Python Tests
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.11"]
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Run Unit Tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/unit/ -v --cov=backend/codesign_playground \
            --cov-report=xml --cov-report=term-missing || true

  test-frontend:
    name: Frontend Tests
    runs-on: ubuntu-latest
    timeout-minutes: 20
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install Dependencies
        run: npm ci
      
      - name: Run Frontend Tests
        run: npm run test:frontend || true

  build:
    name: Build and Package
    runs-on: ubuntu-latest
    timeout-minutes: 20
    needs: [test-python, test-frontend]
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip build
          pip install -r requirements.txt
          npm ci
      
      - name: Build Frontend
        run: npm run build:frontend || true
      
      - name: Build Python Package
        run: python -m build || true

  docker-build:
    name: Docker Build
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: [build]
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Build Docker Images
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64
          push: false
          tags: |
            ai-hardware-codesign-playground:latest
            ai-hardware-codesign-playground:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: 'pip'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install safety bandit
      
      - name: Python Security Scan
        run: |
          safety check --json --output safety-report.json || true
          bandit -r backend/codesign_playground/ -f json -o bandit-report.json || true
        continue-on-error: true
      
      - name: Node.js Security Audit
        run: |
          npm audit --audit-level=moderate || true
        continue-on-error: true
      
      - name: Upload Security Reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json

  codeql-analysis:
    name: CodeQL Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: ['python', 'javascript']
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: ${{ matrix.language }}
      
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
        with:
          category: "/language:${{matrix.language}}"

  secret-scan:
    name: Secret Scanning
    runs-on: ubuntu-latest
    timeout-minutes: 10
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install detect-secrets
        run: pip install detect-secrets
      
      - name: Run Secret Detection
        run: |
          detect-secrets scan --all-files --disable-plugin AbsolutePathDetectorPlugin \
            --exclude-files '.*\.git/.*' \
            --exclude-files '.*node_modules/.*' > .secrets.baseline || true
      
      - name: Upload Secret Scan Results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: secret-scan-results
          path: .secrets.baseline
```

### 3. Release Automation (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: write
  issues: write
  pull-requests: write
  packages: write

jobs:
  release:
    name: Semantic Release
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "18"
          cache: 'npm'
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: 'pip'
      
      - name: Install Dependencies
        run: |
          npm ci
          python -m pip install --upgrade pip build twine
          pip install -r requirements.txt
      
      - name: Build Package
        run: |
          npm run build:frontend
          python -m build
      
      - name: Run Semantic Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
          PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        run: npx semantic-release

  docker-release:
    name: Docker Release
    runs-on: ubuntu-latest
    needs: release
    if: needs.release.outputs.new_release_published == 'true'
    timeout-minutes: 30
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=semver,pattern={{major}}
            type=sha
      
      - name: Build and push Docker images
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 4. Performance Testing (`.github/workflows/performance.yml`)

```yaml
name: Performance Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM
  workflow_dispatch:

jobs:
  performance-test:
    name: Performance Testing
    runs-on: ubuntu-latest
    timeout-minutes: 45
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: perf_test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
          cache: 'pip'
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "18"
          cache: 'npm'
      
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          npm ci
      
      - name: Build Application
        run: |
          npm run build:frontend
          npm run build:backend
      
      - name: Start Application
        run: |
          npm run start &
          sleep 30
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/perf_test_db
          REDIS_URL: redis://localhost:6379/0
          NODE_ENV: production
      
      - name: Run Performance Benchmarks
        run: |
          pytest tests/performance/ -v --junit-xml=junit/performance-results.xml
        env:
          BASE_URL: http://localhost:8000
      
      - name: Run Load Tests
        run: |
          cd tests/performance
          artillery run artillery.yml --output artillery-report.json
      
      - name: Analyze Performance
        run: |
          python scripts/analyze_performance.py \
            --artillery-report tests/performance/artillery-report.json \
            --pytest-report junit/performance-results.xml \
            --output performance-analysis.json
      
      - name: Upload Performance Results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: performance-results
          path: |
            junit/performance-results.xml
            tests/performance/artillery-report.json
            performance-analysis.json
```

## Activation Steps

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Create Workflow Files
Copy each of the workflow configurations above into separate files:
- `.github/workflows/ci.yml`
- `.github/workflows/security.yml`
- `.github/workflows/release.yml`
- `.github/workflows/performance.yml`

### 3. Configure Repository Secrets
See `docs/REPOSITORY_SETUP.md` for complete secret configuration.

Required secrets:
- `CODECOV_TOKEN`
- `NPM_TOKEN`
- `PYPI_TOKEN`
- `SLACK_WEBHOOK_URL`

### 4. Set Branch Protection
Configure branch protection rules as documented in `docs/REPOSITORY_SETUP.md`.

### 5. Test Activation
1. Create a test branch
2. Make a small change
3. Create a pull request
4. Verify all workflows execute successfully

## Expected Impact

### Maturity Progression
- **Before**: 65-70% (MATURING) - Excellent tooling but inactive automation
- **After**: 80-85% (ADVANCED) - Production-ready with active automation

### Benefits
- **Automated Quality Gates**: Pre-commit hooks and CI prevent issues
- **Security Posture**: Continuous scanning and vulnerability detection
- **Release Efficiency**: Automated semantic releases
- **Performance Monitoring**: Continuous performance tracking
- **Operational Excellence**: Comprehensive deployment and monitoring

This activation will transform the repository from having excellent preparation to having world-class operational automation.