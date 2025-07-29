# GitHub Workflows Templates

This directory contains comprehensive GitHub Actions workflow templates for the AI Hardware Co-Design Playground project. These templates provide a complete CI/CD and security scanning setup that can be customized for your specific needs.

## Available Templates

### 1. CI Pipeline Template (`ci-template.yml`)

A comprehensive continuous integration pipeline that includes:

- **Code Quality Checks**
  - Pre-commit hooks validation
  - Python linting (Ruff, Black, isort)
  - TypeScript/JavaScript linting (ESLint)
  - Type checking (mypy, TypeScript)
  - Security scanning (Bandit, Safety)

- **Multi-Platform Testing**
  - Unit tests across Python 3.9-3.12
  - Integration tests with PostgreSQL and Redis
  - Frontend tests with Jest/Vitest
  - Cross-platform testing (Ubuntu, Windows, macOS)
  - Hardware simulation tests (when tools available)

- **End-to-End Testing**
  - Full application testing
  - Browser-based testing support
  - Screenshot capture on failures

- **Build and Package**
  - Python package building
  - Frontend asset compilation
  - Docker multi-architecture builds
  - Artifact uploads

- **Deployment**
  - Conditional production deployment
  - Environment-based approvals
  - Rollback capabilities

### 2. Security Scanning Template (`security-template.yml`)

A comprehensive security scanning pipeline that includes:

- **Dependency Vulnerability Scanning**
  - Python: Safety, pip-audit
  - Node.js: npm audit
  - Automated SARIF uploads to GitHub Security

- **Static Application Security Testing (SAST)**
  - Bandit for Python security issues
  - Semgrep for multi-language security patterns
  - GitHub Security integration

- **Secret Detection**
  - detect-secrets baseline validation
  - TruffleHog entropy-based detection
  - Historical commit scanning

- **Container Security**
  - Trivy vulnerability scanning
  - Multi-layer security analysis
  - Base image security validation

- **License Compliance**
  - Python license analysis (pip-licenses)
  - Node.js license checking
  - Compliance reporting

- **Infrastructure as Code Security**
  - Dockerfile security scanning
  - Docker Compose validation
  - Kubernetes manifest security (when applicable)

- **SBOM Generation**
  - Software Bill of Materials creation
  - SPDX and CycloneDX formats
  - Dependency tracking

## Setup Instructions

### 1. Copy Templates to Workflows Directory

```bash
# Create the workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy the CI template
cp docs/workflows/ci-template.yml .github/workflows/ci.yml

# Copy the security template
cp docs/workflows/security-template.yml .github/workflows/security.yml
```

### 2. Required GitHub Secrets

Add the following secrets to your GitHub repository (Settings > Secrets and variables > Actions):

#### Deployment Secrets
```
DEPLOY_TOKEN          # Deployment authentication token
PRODUCTION_URL        # Production environment URL
```

#### Notification Secrets
```
SLACK_WEBHOOK_URL     # Slack webhook for CI notifications
```

#### Container Registry Secrets
```
DOCKER_USERNAME       # Docker registry username
DOCKER_PASSWORD       # Docker registry password/token
```

#### Cloud Provider Secrets (if using cloud deployment)
```
AWS_ACCESS_KEY_ID     # AWS access key
AWS_SECRET_ACCESS_KEY # AWS secret key
AZURE_CLIENT_ID       # Azure client ID
AZURE_CLIENT_SECRET   # Azure client secret
GCP_SERVICE_ACCOUNT   # GCP service account JSON
```

### 3. Required GitHub Repository Settings

#### Branch Protection Rules
Configure branch protection for `main` and `develop` branches:

1. Go to Settings > Branches
2. Add rule for `main` branch:
   - Require pull request reviews (2 reviewers recommended)
   - Require status checks to pass before merging
   - Required status checks:
     - `Code Quality`
     - `Python Tests (ubuntu-latest, 3.11)`
     - `Frontend Tests`
     - `Build and Package`
   - Require branches to be up to date before merging
   - Restrict pushes that create files over 100MB

#### Security Settings
1. Enable Dependabot alerts (Security > Dependabot alerts)
2. Enable Secret scanning (Security > Secret scanning)
3. Enable Code scanning (Security > Code scanning)

### 4. Environment Configuration

#### Production Environment
1. Go to Settings > Environments
2. Create `production` environment
3. Add protection rules:
   - Required reviewers (production team)
   - Deployment branches (main only)
   - Environment secrets as needed

### 5. Customization Guide

#### Modifying Test Matrix
Edit the `strategy.matrix` section in `ci-template.yml`:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.9", "3.10", "3.11", "3.12"]
    # Add or remove versions as needed
```

#### Adding New Test Steps
Add new steps to the appropriate job:

```yaml
- name: Custom Test Step
  run: |
    echo "Running custom tests..."
    # Your custom test commands
```

#### Modifying Security Scans
Enable/disable specific security tools by commenting out job steps or entire jobs:

```yaml
# Uncomment to disable a specific security scan
# secret-scan:
#   name: Secret Detection
#   runs-on: ubuntu-latest
#   # ... rest of job configuration
```

#### Customizing Notifications
Modify the notification step to use your preferred service:

```yaml
- name: Notify Teams
  uses: 8398a7/action-slack@v3  # Change to your notification service
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.TEAMS_WEBHOOK_URL }}  # Update secret name
```

## Workflow Triggers

### CI Pipeline Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch
- Scheduled runs (if configured)

### Security Pipeline Triggers
- Push to `main` or `develop` branches
- Pull requests to any branch
- Daily scheduled scans (2 AM UTC)
- Manual workflow dispatch

## Performance Optimization

### Caching Strategy
The templates include aggressive caching for:
- Python pip cache
- Node.js npm cache
- Docker layer cache
- GitHub Actions cache

### Parallel Execution
- Test jobs run in parallel across different OS/version combinations
- Security scans run independently
- Build steps are optimized for maximum parallelism

### Resource Management
- Appropriate timeouts for each job type
- Memory-efficient test execution
- Artifact cleanup and retention policies

## Monitoring and Reporting

### Test Results
- JUnit XML reports for all test suites
- Coverage reports uploaded to Codecov
- Artifact preservation for failed builds

### Security Reports
- SARIF uploads to GitHub Security tab
- Consolidated security summary reports
- Pull request comments with security status

### Build Artifacts
- Python packages (wheel and source)
- Frontend static assets
- Docker images with multi-arch support
- SBOM files for compliance

## Troubleshooting

### Common Issues

#### Test Failures
1. Check service health (PostgreSQL, Redis)
2. Verify environment variables are set correctly
3. Review test logs in the Actions tab

#### Security Scan Failures
1. Review SARIF uploads in Security tab
2. Check for new vulnerabilities in dependencies
3. Update security baselines if needed

#### Build Failures
1. Check dependency conflicts
2. Verify Docker image builds locally
3. Review build logs for missing assets

### Debug Mode
Enable debug logging by adding to workflow environment:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Contributing

When modifying workflow templates:

1. Test changes in a feature branch first
2. Validate YAML syntax using `yamllint`
3. Check workflow syntax using GitHub Actions validator
4. Document any new secrets or configuration requirements
5. Update this README with any changes

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Security Scanning Best Practices](https://docs.github.com/en/code-security)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [SARIF Format Specification](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html)