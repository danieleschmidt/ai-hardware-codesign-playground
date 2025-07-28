# Manual Workflow Setup Instructions

Due to GitHub App permission limitations, repository maintainers must manually set up the CI/CD workflows and repository configuration. This guide provides step-by-step instructions.

## 🚨 Required Actions

### 1. Copy Workflow Files

**Action Required**: Copy workflow files from documentation to `.github/workflows/`

```bash
# From repository root
mkdir -p .github/workflows

# Copy all example workflows
cp docs/workflows/examples/*.yml .github/workflows/

# OR copy individual workflows as needed
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Configure Repository Secrets

**Action Required**: Add required secrets in repository settings

Go to: `Settings > Secrets and variables > Actions`

#### Essential Secrets
```bash
# Docker Registry (for container deployment)
DOCKER_REGISTRY_TOKEN        # GitHub Container Registry token
DOCKER_REGISTRY_USERNAME     # Your GitHub username

# Code Quality & Security
CODECOV_TOKEN               # From codecov.io for coverage reporting
SONAR_TOKEN                 # From sonarcloud.io for code analysis
SNYK_TOKEN                  # From snyk.io for security scanning
SEMGREP_APP_TOKEN           # From semgrep.dev for security scanning

# Package Publishing (optional)
NPM_TOKEN                   # For publishing to npm registry
PYPI_TOKEN                  # For publishing to PyPI

# Notifications (optional)
SLACK_WEBHOOK_URL          # For Slack notifications
DISCORD_WEBHOOK_URL        # For Discord notifications
```

#### Cloud Deployment Secrets (if applicable)
```bash
# AWS
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION

# Azure
AZURE_CLIENT_ID
AZURE_CLIENT_SECRET
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID

# Google Cloud
GCP_PROJECT_ID
GCP_SERVICE_ACCOUNT_KEY     # Base64 encoded service account JSON
```

### 3. Set Up Branch Protection Rules

**Action Required**: Configure branch protection for `main` branch

Go to: `Settings > Branches > Add rule`

#### Branch Protection Configuration
```yaml
Branch name pattern: main

Restrictions:
☑️ Restrict pushes that create files larger than 100MB
☑️ Require a pull request before merging
  ☑️ Require approvals: 1
  ☑️ Dismiss stale PR approvals when new commits are pushed
  ☑️ Require review from code owners
  ☑️ Restrict pushes that create files larger than 100MB

☑️ Require status checks to pass before merging
  ☑️ Require branches to be up to date before merging
  Required status checks:
    - Code Quality
    - Security Scan  
    - Python Tests (ubuntu-latest, 3.11)  # At minimum
    - Frontend Tests
    - Integration Tests
    - Docker Build
    - Documentation

☑️ Require conversation resolution before merging
☑️ Require signed commits
☑️ Include administrators
☑️ Restrict pushes that create files larger than 100MB
```

### 4. Configure Repository Settings

**Action Required**: Update repository settings for optimal workflow operation

#### General Settings
Go to: `Settings > General`

```yaml
Features:
☑️ Issues
☑️ Discussions  
☑️ Projects
☑️ Wiki
☑️ Releases
☑️ Packages
☑️ Security and analysis

Pull Requests:
☑️ Allow merge commits
☑️ Allow squash merging
☑️ Allow rebase merging
☑️ Always suggest updating pull request branches
☑️ Allow auto-merge
☑️ Automatically delete head branches
```

#### Actions Settings
Go to: `Settings > Actions > General`

```yaml
Actions permissions:
● Allow all actions and reusable workflows

Fork pull request workflows:
☑️ Run workflows from fork pull requests
● Require approval for first-time contributors

Workflow permissions:
● Read and write permissions
☑️ Allow GitHub Actions to create and approve pull requests
```

### 5. Enable Security Features

**Action Required**: Enable security and analysis features

Go to: `Settings > Security & analysis`

```yaml
Security:
☑️ Dependency graph
☑️ Dependabot alerts
☑️ Dependabot security updates
☑️ Dependabot version updates
☑️ Code scanning alerts
☑️ Secret scanning alerts
☑️ Push protection
```

### 6. Create Issue and PR Templates

**Action Required**: Set up issue and PR templates

```bash
# Create templates directory
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/PULL_REQUEST_TEMPLATE

# Copy templates from docs (if available) or create new ones
cp docs/templates/bug_report.yml .github/ISSUE_TEMPLATE/
cp docs/templates/feature_request.yml .github/ISSUE_TEMPLATE/
cp docs/templates/pull_request_template.md .github/PULL_REQUEST_TEMPLATE/
```

### 7. Configure Dependabot

**Action Required**: Create Dependabot configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/backend"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    reviewers:
      - "@terragon-labs/backend-team"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "deps"
      include: "scope"
  
  # Node.js dependencies
  - package-ecosystem: "npm"
    directory: "/frontend"
    schedule:
      interval: "weekly"
      day: "monday" 
      time: "09:00"
    open-pull-requests-limit: 5
    reviewers:
      - "@terragon-labs/frontend-team"
    labels:
      - "dependencies"
      - "javascript"
    commit-message:
      prefix: "deps"
      include: "scope"
  
  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    reviewers:
      - "@terragon-labs/devops-team"
    labels:
      - "dependencies"
      - "docker"
    commit-message:
      prefix: "deps"
      include: "scope"
  
  # GitHub Actions dependencies
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    reviewers:
      - "@terragon-labs/devops-team"
    labels:
      - "dependencies"
      - "github-actions"
    commit-message:
      prefix: "deps"
      include: "scope"
```

## 🔍 Verification Steps

### 1. Test Workflow Execution

```bash
# After copying workflows, commit and push
git add .github/
git commit -m "feat: add CI/CD workflows and repository configuration"
git push origin main

# Create a test PR to verify workflows
git checkout -b test-workflows
echo "# Test" >> test.md
git add test.md
git commit -m "test: verify workflow execution"
git push origin test-workflows

# Create PR through GitHub UI and verify workflows run
```

### 2. Verify Branch Protection

```bash
# Try to push directly to main (should fail)
git checkout main
echo "# Direct push test" >> test-direct.md
git add test-direct.md
git commit -m "test: direct push should fail"
git push origin main  # This should be rejected
```

### 3. Check Security Scanning

- Go to `Security > Code scanning alerts`
- Verify that security scans are running
- Check for any existing vulnerabilities

### 4. Test Dependabot

- Go to `Insights > Dependency graph > Dependabot`
- Verify Dependabot is monitoring dependencies
- Check for any pending security updates

## 🚫 Common Issues & Solutions

### Issue: Workflows not running

**Symptoms**: No Actions appearing in the Actions tab

**Solutions**:
1. Verify workflow files are in `.github/workflows/` directory
2. Check YAML syntax: `yamllint .github/workflows/*.yml`
3. Ensure Actions are enabled in repository settings
4. Check file permissions (should be readable)

### Issue: Permission denied errors

**Symptoms**: Workflows fail with permission errors

**Solutions**:
1. Check workflow permissions in repository settings
2. Verify required secrets are configured
3. Update workflow permissions in YAML files:
   ```yaml
   permissions:
     contents: read
     actions: read
     checks: write
     pull-requests: write
     security-events: write
   ```

### Issue: Status checks not required

**Symptoms**: PRs can be merged despite failing checks

**Solutions**:
1. Verify branch protection rules are enabled
2. Check that workflow job names match required status checks
3. Ensure "Require branches to be up to date" is enabled

### Issue: Secret scanning alerts

**Symptoms**: Alerts about secrets in code

**Solutions**:
1. Remove secrets from code immediately
2. Rotate any exposed secrets
3. Use environment variables or GitHub secrets instead
4. Add secrets to `.gitignore` and `.gitsecrets`

## 📞 Support

### Getting Help

1. **Documentation**: Check GitHub Actions documentation
2. **Community**: Ask in GitHub Community Discussions
3. **Internal**: Contact the DevOps team via Slack `#devops-support`
4. **Issues**: Create an issue in this repository with the `workflow` label

### Escalation Path

1. **Level 1**: Team lead or senior developer
2. **Level 2**: DevOps team or platform engineering
3. **Level 3**: GitHub Support (for platform issues)

### Office Hours

- **When**: Thursdays 2-3 PM UTC
- **Where**: Team video call (link in calendar)
- **What**: CI/CD questions, workflow debugging, best practices

## 📝 Next Steps

After completing the manual setup:

1. ✅ Copy workflow files to `.github/workflows/`
2. ✅ Configure repository secrets
3. ✅ Set up branch protection rules
4. ✅ Update repository settings
5. ✅ Enable security features
6. ✅ Create issue/PR templates
7. ✅ Configure Dependabot
8. ✅ Test workflow execution
9. ✅ Verify all checks are working
10. ✅ Document any customizations

**Final Step**: Update this document with any repository-specific modifications or lessons learned during setup.
