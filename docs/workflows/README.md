# CI/CD Workflow Documentation & Templates

This directory contains comprehensive CI/CD workflow documentation and templates for the AI Hardware Co-Design Playground. Due to GitHub App permission limitations, repository maintainers must manually create the actual workflow files from these templates.

## Directory Structure

```
docs/workflows/
├── README.md                    # This file
├── examples/                    # Example workflow files
│   ├── ci.yml                  # Continuous Integration
│   ├── cd.yml                  # Continuous Deployment
│   ├── security-scan.yml       # Security scanning
│   ├── dependency-update.yml   # Automated dependency updates
│   ├── release.yml             # Automated releases
│   └── performance.yml         # Performance testing
├── security/                   # Security-related workflows
│   ├── slsa.yml               # SLSA compliance
│   ├── sbom.yml               # Software Bill of Materials
│   └── vulnerability-scan.yml  # Vulnerability scanning
├── documentation.md            # Workflow documentation
├── branch-protection.md        # Branch protection setup
└── manual-setup.md             # Manual setup instructions
```

## Quick Setup

### 1. Copy Workflow Files

Repository maintainers must manually copy the workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# From repository root
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/
```

### 2. Configure Repository Settings

Refer to [`manual-setup.md`](manual-setup.md) for detailed instructions on:
- Branch protection rules
- Required status checks
- GitHub App permissions
- Secrets configuration

### 3. Enable Workflows

After copying the files:
1. Commit and push the workflow files
2. Go to your repository's "Actions" tab
3. Enable the workflows that appear
4. Configure any required secrets in repository settings

## Workflow Overview

| Workflow | Trigger | Purpose | Duration |
|----------|---------|---------|----------|
| **CI** | PR, Push to main | Code quality, tests, security | ~15 min |
| **CD** | Push to main, tags | Automated deployment | ~10 min |
| **Security Scan** | Daily, PR | Vulnerability assessment | ~5 min |
| **Dependency Update** | Weekly | Automated dependency updates | ~20 min |
| **Release** | Tag creation | Automated release creation | ~15 min |
| **Performance** | Nightly | Performance regression testing | ~30 min |

## Required Secrets

Configure these secrets in your repository settings:

```bash
# Required for all workflows
GITHUB_TOKEN                 # Automatically provided by GitHub

# Docker and deployment
DOCKER_REGISTRY_TOKEN        # Container registry authentication
DOCKER_REGISTRY_USERNAME     # Container registry username

# Cloud deployment (if using)
AWS_ACCESS_KEY_ID           # AWS deployment credentials
AWS_SECRET_ACCESS_KEY       # AWS deployment credentials
AZURE_CLIENT_ID             # Azure deployment credentials
AZURE_CLIENT_SECRET         # Azure deployment credentials
AZURE_TENANT_ID             # Azure deployment credentials

# Security scanning
SONAR_TOKEN                 # SonarCloud integration
SNYK_TOKEN                  # Snyk security scanning
CODECOV_TOKEN              # Code coverage reporting

# Notifications
SLACK_WEBHOOK_URL          # Slack notifications
DISCORD_WEBHOOK_URL        # Discord notifications

# Release management
NPM_TOKEN                  # NPM package publishing
PYPI_TOKEN                 # PyPI package publishing
```

## Workflow Features

### Continuous Integration (CI)
- **Multi-platform testing**: Linux, macOS, Windows
- **Python matrix**: 3.9, 3.10, 3.11, 3.12
- **Node.js versions**: 16, 18, 20
- **Code quality**: Linting, formatting, type checking
- **Security**: Vulnerability scanning, secret detection
- **Performance**: Benchmark regression testing
- **Documentation**: Build and deploy docs

### Continuous Deployment (CD)
- **Environment promotion**: dev → staging → production
- **Blue-green deployment**: Zero-downtime deployments
- **Rollback capability**: Automatic rollback on failure
- **Health checks**: Post-deployment verification
- **Monitoring**: Integration with observability stack

### Security
- **SLSA Level 3**: Supply chain security compliance
- **SBOM generation**: Software Bill of Materials
- **Vulnerability scanning**: Dependencies and container images
- **Secret scanning**: Prevent credential leaks
- **License compliance**: Open source license verification

## Environment Strategy

### Development
- **Trigger**: Every push to feature branches
- **Tests**: Unit, integration, security scans
- **Deployment**: Development environment
- **Notifications**: Slack/Discord on failure

### Staging
- **Trigger**: Push/PR to `main` branch
- **Tests**: Full test suite including E2E
- **Deployment**: Staging environment
- **Approval**: Automatic (with rollback)

### Production
- **Trigger**: Git tags matching `v*`
- **Tests**: Full validation including performance
- **Deployment**: Production environment
- **Approval**: Manual approval required
- **Monitoring**: Enhanced monitoring and alerting

## Performance Testing

### Benchmarks
- **Model Analysis**: Profiling performance benchmarks
- **Hardware Generation**: RTL generation speed tests
- **Optimization**: Algorithm convergence benchmarks
- **Simulation**: Cycle-accurate simulation performance
- **Memory Usage**: Memory leak detection
- **Load Testing**: Concurrent user simulation

### Regression Detection
- **Baseline Comparison**: Compare against previous runs
- **Threshold Alerts**: Alert on performance degradation >10%
- **Historical Tracking**: Long-term performance trends
- **Automated Reports**: Weekly performance summaries

## Monitoring Integration

### Metrics Collection
- **Build Metrics**: Success rate, duration, failure reasons
- **Test Metrics**: Coverage, flakiness, execution time
- **Deployment Metrics**: Frequency, success rate, rollback rate
- **Security Metrics**: Vulnerability count, resolution time

### Alerting
- **Build Failures**: Immediate Slack/Discord notification
- **Security Issues**: High/Critical vulnerabilities
- **Performance Degradation**: >20% performance drop
- **Deployment Failures**: Production deployment issues

## Compliance & Governance

### Code Quality Gates
- **Test Coverage**: Minimum 80% overall, 95% for critical paths
- **Security Scan**: No high/critical vulnerabilities
- **Code Review**: At least one approval required
- **Documentation**: Updated documentation for new features

### Audit Trail
- **All Changes**: Full audit log of deployments
- **Approval Records**: Who approved what and when
- **Rollback History**: Complete rollback tracking
- **Security Events**: All security-related actions logged

## Troubleshooting

### Common Issues

#### Workflow Permissions
```yaml
# Add to workflow file if needed
permissions:
  contents: read
  security-events: write
  actions: read
  checks: write
  pull-requests: write
```

#### Docker Build Failures
```bash
# Check Docker setup
docker --version
docker-compose --version

# Verify Dockerfile syntax
docker build --dry-run .
```

#### Test Failures
```bash
# Run tests locally
make test

# Check specific test categories
make test-unit
make test-integration
```

### Debug Information

Enable debug logging in workflows:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Best Practices

### Workflow Design
1. **Fail Fast**: Run quick tests first
2. **Parallel Execution**: Maximize parallelism
3. **Resource Optimization**: Use appropriate runner sizes
4. **Caching**: Cache dependencies and build artifacts
5. **Secrets Management**: Use environment-specific secrets

### Security
1. **Principle of Least Privilege**: Minimal required permissions
2. **Secret Rotation**: Regular rotation of all secrets
3. **Vulnerability Response**: SLA for security issue resolution
4. **Supply Chain Security**: Verify all dependencies

### Maintenance
1. **Regular Updates**: Keep actions and dependencies current
2. **Performance Monitoring**: Track workflow execution times
3. **Cost Optimization**: Monitor and optimize CI/CD costs
4. **Documentation**: Keep workflow documentation current

## Migration Guide

### From Other CI/CD Systems

#### Jenkins
- Convert Jenkinsfile pipelines to GitHub Actions
- Migrate Jenkins plugins to GitHub Actions equivalents
- Update deployment scripts for GitHub Actions environment

#### GitLab CI
- Convert `.gitlab-ci.yml` to GitHub Actions workflows
- Migrate GitLab Runner configurations
- Update registry and deployment configurations

#### CircleCI
- Convert `.circleci/config.yml` to GitHub Actions
- Migrate CircleCI orbs to GitHub Actions
- Update environment variable configurations

## Support

### Getting Help
- **Documentation**: [GitHub Actions Documentation](https://docs.github.com/en/actions)
- **Community**: [GitHub Community Discussions](https://github.com/community)
- **Support**: [GitHub Support](https://support.github.com/)

### Internal Resources
- **Team Slack**: `#ci-cd-support` channel
- **Documentation**: Internal CI/CD knowledge base
- **Office Hours**: Weekly CI/CD office hours (Thursdays 2-3 PM UTC)

For specific questions about workflow implementation, please refer to the detailed documentation files in this directory or reach out through our community channels.