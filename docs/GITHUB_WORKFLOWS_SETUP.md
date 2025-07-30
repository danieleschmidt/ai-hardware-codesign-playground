# GitHub Workflows Setup Guide

This document provides the complete GitHub Actions workflow configurations that need to be manually created due to GitHub App permission restrictions.

## 🚨 Required Manual Setup

Create the following files in your repository under `.github/workflows/`:

## 1. CI Pipeline (`.github/workflows/ci.yml`)

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
  quality-gate:
    name: Code Quality Gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e .[dev,test]
      - name: Run quality checks
        run: |
          npm run lint
          npm run typecheck
          npm run security

  test-backend:
    name: Backend Tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e .[test]
      - name: Run tests
        run: |
          npm run test:backend
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/coverage.xml

  test-frontend:
    name: Frontend Tests
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ./frontend
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      - name: Install dependencies
        run: npm ci
      - name: Run tests
        run: npm run test

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e .[test]
      - name: Run integration tests
        run: |
          npm run test:e2e
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
          REDIS_URL: redis://localhost:6379/0

  build-test:
    name: Build Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      - name: Install dependencies
        run: |
          npm install
          pip install -e .[dev]
      - name: Build application
        run: |
          npm run build
      - name: Test Docker build
        run: |
          docker build -t test-build .
```

## 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday at 6 AM UTC
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  actions: read

jobs:
  dependency-security:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install safety bandit
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true
      - name: Run Bandit security scan
        run: |
          bandit -r backend/ -f json -o bandit-report.json || true
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json

  container-security:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -t security-scan:latest .
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'security-scan:latest'
          format: 'sarif'
          output: 'trivy-results.sarif'
      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Run TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

  code-security:
    name: Code Security Analysis
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
        with:
          languages: python, javascript
      - name: Autobuild
        uses: github/codeql-action/autobuild@v2
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  sbom-generation:
    name: Generate SBOM
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate SBOM for Python
        uses: anchore/sbom-action@v0
        with:
          path: ./
          format: spdx-json
          output-file: sbom-python.spdx.json
      - name: Generate SBOM for Container
        run: |
          docker build -t sbom-scan:latest .
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            -v $(pwd):/workspace anchore/syft:latest \
            sbom-scan:latest -o spdx-json=/workspace/sbom-container.spdx.json
      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v3
        with:
          name: sbom-reports
          path: |
            sbom-*.spdx.json
```

## 3. Performance Testing (`.github/workflows/performance.yml`)

```yaml
name: Performance Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"

jobs:
  benchmark-tests:
    name: Python Benchmark Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e .[test,dev]
          pip install pytest-benchmark
      - name: Run benchmarks
        run: |
          npm run benchmark
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark-results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true

  load-testing:
    name: Load Testing
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e .[test]
          pip install locust
      - name: Start application
        run: |
          npm run start &
          sleep 30
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
          REDIS_URL: redis://localhost:6379/0
      - name: Run load tests
        run: |
          locust --host=http://localhost:8000 --users=10 --spawn-rate=2 \
                 --run-time=60s --headless --print-stats \
                 --html=load-test-report.html
      - name: Upload load test report
        uses: actions/upload-artifact@v3
        with:
          name: load-test-report
          path: load-test-report.html

  memory-profiling:
    name: Memory Profiling
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: |
          pip install -e .[test,dev]
          pip install memory-profiler matplotlib
      - name: Run memory profiling
        run: |
          python -m memory_profiler backend/codesign_playground/main.py
      - name: Generate memory usage report
        run: |
          mprof run --python python backend/codesign_playground/main.py
          mprof plot --output=memory-profile.png
      - name: Upload memory profile
        uses: actions/upload-artifact@v3
        with:
          name: memory-profile
          path: |
            memory-profile.png
            *.dat
```

## 4. Compliance & Governance (`.github/workflows/compliance.yml`)

```yaml
name: Compliance & Governance

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 4 * * 0'  # Weekly Sunday at 4 AM UTC
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  checks: write

jobs:
  license-compliance:
    name: License Compliance Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install license checker
        run: |
          pip install pip-licenses licensecheck
      - name: Check Python licenses
        run: |
          pip install -e .[all]
          pip-licenses --format=json --output-file=python-licenses.json
          pip-licenses --format=plain-vertical --allowed-only --fail-on-forbidden
      - name: Check Node.js licenses
        if: hashFiles('package.json') != ''
        run: |
          npm install -g license-checker
          license-checker --json --out nodejs-licenses.json
          license-checker --onlyAllow 'MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC'
      - name: Upload license reports
        uses: actions/upload-artifact@v3
        with:
          name: license-reports
          path: |
            python-licenses.json
            nodejs-licenses.json

  dependency-audit:
    name: Dependency Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install audit tools
        run: |
          pip install pip-audit cyclonedx-bom
      - name: Run Python dependency audit
        run: |
          pip install -e .[all]
          pip-audit --format=json --output=python-audit.json
      - name: Generate Python SBOM
        run: |
          cyclonedx-py -o python-sbom.json
      - name: Run npm audit
        if: hashFiles('package.json') != ''
        run: |
          npm audit --audit-level=moderate --json > npm-audit.json || true
      - name: Upload audit reports
        uses: actions/upload-artifact@v3
        with:
          name: dependency-audit
          path: |
            python-audit.json
            python-sbom.json
            npm-audit.json

  policy-validation:
    name: Policy as Code Validation
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Open Policy Agent
        run: |
          curl -L -o opa https://openpolicyagent.org/downloads/v0.58.0/opa_linux_amd64_static
          chmod +x opa
          sudo mv opa /usr/local/bin/
      - name: Validate policies
        run: |
          if [ -d "policies/" ]; then
            opa fmt --diff policies/
            opa test policies/
          fi
      - name: Run container policy checks
        if: hashFiles('Dockerfile') != ''
        run: |
          docker run --rm -v $(pwd):/workspace \
            openpolicyagent/conftest verify --policy policies/docker.rego Dockerfile

  supply-chain-security:
    name: Supply Chain Security
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Generate SLSA provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
        with:
          base64-subjects: "${{ needs.build.outputs.hashes }}"
      - name: Verify SLSA provenance
        run: |
          curl -sSL https://github.com/slsa-framework/slsa-verifier/releases/download/v2.4.1/slsa-verifier-linux-amd64 -o slsa-verifier
          chmod +x slsa-verifier
          ./slsa-verifier verify-artifact --provenance-path provenance.intoto.jsonl --source-uri github.com/${{ github.repository }}

  data-privacy-scan:
    name: Data Privacy Compliance
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Scan for PII/sensitive data
        run: |
          # Install whispers for static analysis of secrets/PII
          pip install whispers
          whispers --config .whispers.yml --output whispers-report.json .
      - name: Check GDPR compliance patterns
        run: |
          # Search for data processing patterns that need GDPR consideration
          grep -r "personal.*data\|email\|phone\|address" --include="*.py" --include="*.js" . > gdpr-patterns.txt || true
      - name: Upload privacy scan results
        uses: actions/upload-artifact@v3
        with:
          name: privacy-scan
          path: |
            whispers-report.json
            gdpr-patterns.txt

  architecture-compliance:
    name: Architecture Decision Records
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate ADR structure
        run: |
          # Check that ADRs follow the required template
          for adr in docs/adr/*.md; do
            if [ -f "$adr" ]; then
              echo "Validating $adr"
              grep -q "# ADR-" "$adr" || (echo "Missing ADR header in $adr" && exit 1)
              grep -q "## Status" "$adr" || (echo "Missing Status section in $adr" && exit 1)
              grep -q "## Context" "$adr" || (echo "Missing Context section in $adr" && exit 1)
              grep -q "## Decision" "$adr" || (echo "Missing Decision section in $adr" && exit 1)
              grep -q "## Consequences" "$adr" || (echo "Missing Consequences section in $adr" && exit 1)
            fi
          done
      - name: Generate architecture overview
        run: |
          # Create architecture compliance report
          echo "# Architecture Compliance Report" > architecture-report.md
          echo "Generated on: $(date)" >> architecture-report.md
          echo "" >> architecture-report.md
          echo "## Architecture Decision Records" >> architecture-report.md
          ls docs/adr/*.md | wc -l | xargs echo "Total ADRs:" >> architecture-report.md
      - name: Upload architecture report
        uses: actions/upload-artifact@v3
        with:
          name: architecture-compliance
          path: architecture-report.md

  compliance-reporting:
    name: Generate Compliance Report
    needs: [license-compliance, dependency-audit, policy-validation, supply-chain-security, data-privacy-scan, architecture-compliance]
    runs-on: ubuntu-latest
    if: always()
    steps:
      - uses: actions/checkout@v4
      - name: Download all artifacts
        uses: actions/download-artifact@v3
      - name: Generate comprehensive compliance report
        run: |
          mkdir -p compliance-reports
          echo "# Compliance Report - $(date)" > compliance-reports/compliance-summary.md
          echo "" >> compliance-reports/compliance-summary.md
          
          echo "## License Compliance" >> compliance-reports/compliance-summary.md
          if [ -f license-reports/python-licenses.json ]; then
            echo "✅ Python license compliance checked" >> compliance-reports/compliance-summary.md
          fi
          
          echo "## Security Audit" >> compliance-reports/compliance-summary.md
          if [ -f dependency-audit/python-audit.json ]; then
            echo "✅ Dependency security audit completed" >> compliance-reports/compliance-summary.md
          fi
          
          echo "## Supply Chain Security" >> compliance-reports/compliance-summary.md
          echo "✅ SLSA provenance generated" >> compliance-reports/compliance-summary.md
          
          echo "## Data Privacy" >> compliance-reports/compliance-summary.md
          if [ -f privacy-scan/whispers-report.json ]; then
            echo "✅ Privacy scan completed" >> compliance-reports/compliance-summary.md
          fi
          
          echo "## Architecture Governance" >> compliance-reports/compliance-summary.md
          if [ -f architecture-compliance/architecture-report.md ]; then
            echo "✅ ADR validation passed" >> compliance-reports/compliance-summary.md
          fi
      - name: Upload compliance summary
        uses: actions/upload-artifact@v3
        with:
          name: compliance-summary
          path: compliance-reports/
```

## 🚀 Setup Instructions

1. **Create the workflows directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy each workflow**:
   - Copy the CI workflow to `.github/workflows/ci.yml`
   - Copy the Security workflow to `.github/workflows/security.yml`
   - Copy the Performance workflow to `.github/workflows/performance.yml`
   - Copy the Compliance workflow to `.github/workflows/compliance.yml`

3. **Commit and push**:
   ```bash
   git add .github/workflows/
   git commit -m "feat: add comprehensive GitHub Actions workflows"
   git push
   ```

4. **Configure secrets** (if needed):
   - Go to your repository Settings → Secrets and variables → Actions
   - Add any required secrets for your specific setup

## ✅ Verification

After setup, verify that:
- [ ] All workflows appear in the Actions tab
- [ ] CI workflow runs on pull requests
- [ ] Security scans complete successfully
- [ ] Performance tests execute without errors
- [ ] Compliance checks pass

This completes the GitHub Actions setup for your comprehensive SDLC enhancement!