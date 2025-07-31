# Repository Setup Guide

This document provides instructions for setting up the repository with proper branch protection, secrets, and configuration.

## Branch Protection Rules

Configure branch protection for the `main` branch with the following settings:

### Required Settings
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "code-quality",
      "test-python (3.9)",
      "test-python (3.11)",
      "test-frontend",
      "build",
      "docker-build"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "restrict_pushes": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
```

### GitHub CLI Setup
```bash
# Install GitHub CLI
gh auth login

# Set branch protection
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["code-quality","test-python (3.9)","test-python (3.11)","test-frontend","build","docker-build"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

## Required Secrets

Configure the following secrets in GitHub repository settings:

### CI/CD Secrets
- `CODECOV_TOKEN`: Code coverage reporting token
- `SONAR_TOKEN`: SonarCloud analysis token (if using)

### Deployment Secrets
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `DEPLOY_TOKEN`: Deployment authentication token
- `PRODUCTION_URL`: Production deployment URL

### Release Secrets
- `NPM_TOKEN`: NPM publishing token
- `PYPI_TOKEN`: PyPI publishing token

### Notification Secrets
- `SLACK_WEBHOOK_URL`: Slack notifications webhook

### Security Scanning
- `SNYK_TOKEN`: Snyk security scanning token (if using)

## Repository Labels

Create the following labels for issue and PR management:

```bash
# Bug-related labels
gh label create "bug" --description "Something isn't working" --color "d73a4a"
gh label create "critical" --description "Critical priority" --color "b60205"
gh label create "high-priority" --description "High priority" --color "d93f0b"

# Feature-related labels
gh label create "enhancement" --description "New feature or request" --color "a2eeef"
gh label create "feature" --description "New feature" --color "0052cc"
gh label create "hardware" --description "Hardware/RTL related" --color "5319e7"
gh label create "ml" --description "Machine learning related" --color "0e8a16"

# Process labels
gh label create "documentation" --description "Improvements or additions to documentation" --color "0075ca"
gh label create "testing" --description "Testing related" --color "f9d0c4"
gh label create "performance" --description "Performance related" --color "fbca04"
gh label create "security" --description "Security related" --color "d4c5f9"

# Status labels
gh label create "needs-review" --description "Needs code review" --color "fbca04"
gh label create "needs-testing" --description "Needs testing" --color "fef2c0"
gh label create "work-in-progress" --description "Work in progress" --color "ededed"
gh label create "blocked" --description "Blocked by external dependency" --color "d73a4a"

# Size labels
gh label create "size/xs" --description "Extra small change" --color "3cbf00"
gh label create "size/s" --description "Small change" --color "5d9801"
gh label create "size/m" --description "Medium change" --color "7f6f00"
gh label create "size/l" --description "Large change" --color "a14f00"
gh label create "size/xl" --description "Extra large change" --color "b21800"
```

## Automated Issue Templates

The repository includes automated issue templates for:

- **Bug Reports**: Structured bug reporting with environment details
- **Feature Requests**: Feature proposal template
- **Hardware Issues**: Hardware-specific issue template
- **Performance Issues**: Performance problem reporting
- **Security Issues**: Security vulnerability reporting

## Repository Rulesets

Configure repository rulesets for additional protection:

### Commit Message Rules
```json
{
  "name": "Commit Message Format",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/main", "refs/heads/develop"]
    }
  },
  "rules": [
    {
      "type": "commit_message_pattern",
      "parameters": {
        "pattern": "^(feat|fix|docs|style|refactor|test|chore)(\\(.+\\))?: .{1,50}",
        "operator": "regex"
      }
    }
  ]
}
```

### File Protection Rules
```json
{
  "name": "Protect Configuration Files",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "include": ["refs/heads/**"]
    }
  },
  "rules": [
    {
      "type": "file_path_restriction",
      "parameters": {
        "restricted_file_paths": [
          ".github/workflows/**",
          "pyproject.toml",
          "package.json",
          "docker-compose.yml"
        ]
      }
    }
  ]
}
```

## Webhooks Configuration

Configure webhooks for external integrations:

### Slack Integration
```json
{
  "name": "slack-notifications",
  "active": true,
  "events": ["push", "pull_request", "issues", "release"],
  "config": {
    "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "content_type": "json"
  }
}
```

### Discord Integration
```json
{
  "name": "discord-notifications",
  "active": true,
  "events": ["push", "pull_request", "release"],
  "config": {
    "url": "https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK",
    "content_type": "json"
  }
}
```

## Dependabot Configuration

The repository includes Dependabot configuration in `.github/dependabot.yml`:

- **Python dependencies**: Daily updates
- **Node.js dependencies**: Daily updates  
- **Docker base images**: Weekly updates
- **GitHub Actions**: Weekly updates

## Code Scanning Setup

### CodeQL Configuration
```yaml
# .github/workflows/codeql-analysis.yml already included in security.yml
# Additional configuration can be added to .github/codeql/codeql-config.yml
```

### Dependency Scanning
```yaml
# dependency-review.yml
name: 'Dependency Review'
on: [pull_request]

permissions:
  contents: read

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - name: 'Checkout Repository'
        uses: actions/checkout@v4
      - name: 'Dependency Review'
        uses: actions/dependency-review-action@v3
```

## Performance Monitoring

### Repository Insights
- Enable repository insights for traffic analytics
- Configure automated performance monitoring
- Set up alerts for repository health metrics

### Code Quality Metrics
- Integrate with CodeClimate or SonarCloud
- Configure quality gates in CI/CD pipeline
- Set up automated code quality reports

## Compliance and Governance

### GDPR Compliance
- Configure data retention policies
- Set up automated data export capabilities
- Implement user consent management

### SOC 2 Compliance
- Enable audit logging
- Configure access controls
- Set up automated compliance reporting

### Open Source Governance
- Configure CLA (Contributor License Agreement) bot
- Set up automated license scanning
- Configure trademark and copyright protection

## Monitoring and Alerting

### GitHub Repository Monitoring
```bash
# Set up monitoring for repository health
gh api repos/:owner/:repo/stats/contributors
gh api repos/:owner/:repo/stats/commit_activity
gh api repos/:owner/:repo/stats/participation
```

### Security Monitoring
```bash
# Enable security advisories
gh api repos/:owner/:repo/vulnerability-alerts --method PUT

# Configure secret scanning
gh api repos/:owner/:repo/secret-scanning/alerts
```

## Backup and Disaster Recovery

### Repository Backup
```bash
#!/bin/bash
# Repository backup script
gh repo clone owner/repo backup-$(date +%Y%m%d)
tar -czf repo-backup-$(date +%Y%m%d).tar.gz backup-$(date +%Y%m%d)/
```

### Disaster Recovery Plan
1. **Repository Recovery**: Clone from backup
2. **CI/CD Recovery**: Restore workflow configurations
3. **Secret Recovery**: Restore from secure backup
4. **Access Recovery**: Restore team and permission settings

## Automation Scripts

### Repository Setup Script
```bash
#!/bin/bash
# setup-repository.sh

# Set repository variables
OWNER="terragon-labs"
REPO="ai-hardware-codesign-playground"

echo "Setting up repository: $OWNER/$REPO"

# Enable repository features
gh api repos/$OWNER/$REPO --method PATCH \
  --field has_issues=true \
  --field has_projects=true \
  --field has_wiki=true \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field allow_rebase_merge=true \
  --field delete_branch_on_merge=true

echo "Repository setup complete!"
```

This configuration ensures the repository follows best practices for security, quality, and collaboration while maintaining the flexibility needed for AI hardware co-design development.