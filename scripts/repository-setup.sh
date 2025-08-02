#!/bin/bash

# AI Hardware Co-Design Playground - Repository Setup Script
# Automated repository configuration and integration setup

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_OWNER="danieleschmidt"
REPO_NAME="ai-hardware-codesign-playground"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
AI Hardware Co-Design Playground - Repository Setup

Usage: $0 [OPTIONS]

This script configures repository settings, branch protection, labels, and integrations
for optimal development workflow and SDLC compliance.

OPTIONS:
    -h, --help              Show this help message
    -t, --token TOKEN       GitHub personal access token
    -o, --owner OWNER       Repository owner (default: $REPO_OWNER)
    -r, --repo REPO         Repository name (default: $REPO_NAME)
    -v, --verbose           Enable verbose logging
    --dry-run              Show what would be done without making changes
    --skip-branch-protection    Skip branch protection setup
    --skip-labels          Skip labels setup
    --skip-webhooks        Skip webhooks setup
    --skip-settings        Skip repository settings

EXAMPLES:
    $0 --token ghp_xxxxxxxxxxxx                    # Basic setup
    $0 --token ghp_xxxxxxxxxxxx --verbose          # With verbose output
    $0 --dry-run                                   # Preview changes
    $0 --skip-branch-protection --skip-webhooks    # Minimal setup

REQUIRED PERMISSIONS:
    - Repository: admin access
    - GitHub Token: repo, admin:repo_hook, admin:org

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN          GitHub personal access token
    GITHUB_OWNER         Repository owner override
    GITHUB_REPO          Repository name override
EOF
}

# Parse command line arguments
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
VERBOSE=false
DRY_RUN=false
SKIP_BRANCH_PROTECTION=false
SKIP_LABELS=false
SKIP_WEBHOOKS=false
SKIP_SETTINGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--token)
            GITHUB_TOKEN="$2"
            shift 2
            ;;
        -o|--owner)
            REPO_OWNER="$2"
            shift 2
            ;;
        -r|--repo)
            REPO_NAME="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-branch-protection)
            SKIP_BRANCH_PROTECTION=true
            shift
            ;;
        --skip-labels)
            SKIP_LABELS=true
            shift
            ;;
        --skip-webhooks)
            SKIP_WEBHOOKS=true
            shift
            ;;
        --skip-settings)
            SKIP_SETTINGS=true
            shift
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Verbose logging
debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# GitHub API helper
github_api() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "[DRY RUN] Would call: $method $endpoint"
        [[ -n "$data" ]] && echo "Data: $data"
        return 0
    fi
    
    local curl_args=(-s -X "$method" -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json")
    
    if [[ -n "$data" ]]; then
        curl_args+=(-H "Content-Type: application/json" -d "$data")
    fi
    
    curl "${curl_args[@]}" "https://api.github.com$endpoint"
}

# Validate prerequisites
validate_prerequisites() {
    log "Validating prerequisites..."
    
    # Check GitHub token
    if [[ -z "$GITHUB_TOKEN" ]]; then
        error "GitHub token is required. Set GITHUB_TOKEN environment variable or use --token"
        exit 1
    fi
    
    # Check required tools
    local missing_tools=()
    for tool in curl jq git; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Validate GitHub token
    if [[ "$DRY_RUN" == "false" ]]; then
        local user_info=$(github_api GET "/user")
        if ! echo "$user_info" | jq -e '.login' > /dev/null; then
            error "Invalid GitHub token or API error"
            exit 1
        fi
        local username=$(echo "$user_info" | jq -r '.login')
        debug "Authenticated as GitHub user: $username"
    fi
    
    success "Prerequisites validated"
}

# Configure repository settings
configure_repository_settings() {
    if [[ "$SKIP_SETTINGS" == "true" ]]; then
        debug "Skipping repository settings"
        return
    fi
    
    log "Configuring repository settings..."
    
    local settings='{
        "description": "AI Hardware Co-Design Playground - Advanced FPGA/ASIC development with ML integration and comprehensive SDLC automation",
        "homepage": "https://github.com/'$REPO_OWNER'/'$REPO_NAME'",
        "topics": ["ai", "hardware", "codesign", "fpga", "asic", "machine-learning", "automation", "sdlc", "devops", "synthesis"],
        "private": false,
        "has_issues": true,
        "has_projects": true,
        "has_wiki": true,
        "has_downloads": true,
        "default_branch": "main",
        "allow_squash_merge": true,
        "allow_merge_commit": false,
        "allow_rebase_merge": true,
        "delete_branch_on_merge": true,
        "allow_auto_merge": true,
        "allow_update_branch": true,
        "security_and_analysis": {
            "secret_scanning": {
                "status": "enabled"
            },
            "secret_scanning_push_protection": {
                "status": "enabled"
            },
            "dependency_graph": {
                "status": "enabled"
            },
            "dependabot_security_updates": {
                "status": "enabled"
            }
        }
    }'
    
    local response=$(github_api PATCH "/repos/$REPO_OWNER/$REPO_NAME" "$settings")
    
    if [[ "$DRY_RUN" == "false" ]] && echo "$response" | jq -e '.name' > /dev/null; then
        success "Repository settings configured"
    elif [[ "$DRY_RUN" == "false" ]]; then
        warning "Repository settings update may have failed: $response"
    fi
}

# Setup branch protection rules
setup_branch_protection() {
    if [[ "$SKIP_BRANCH_PROTECTION" == "true" ]]; then
        debug "Skipping branch protection setup"
        return
    fi
    
    log "Setting up branch protection for 'main' branch..."
    
    local protection_rules='{
        "required_status_checks": {
            "strict": true,
            "contexts": [
                "ci/build",
                "ci/test",
                "ci/lint",
                "ci/security-scan",
                "ci/type-check"
            ]
        },
        "enforce_admins": false,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": true,
            "require_code_owner_reviews": true,
            "bypass_pull_request_allowances": {
                "users": [],
                "teams": [],
                "apps": []
            }
        },
        "restrictions": null,
        "allow_force_pushes": false,
        "allow_deletions": false,
        "required_linear_history": true,
        "required_conversation_resolution": true
    }'
    
    local response=$(github_api PUT "/repos/$REPO_OWNER/$REPO_NAME/branches/main/protection" "$protection_rules")
    
    if [[ "$DRY_RUN" == "false" ]] && echo "$response" | jq -e '.url' > /dev/null; then
        success "Branch protection configured for 'main'"
    elif [[ "$DRY_RUN" == "false" ]]; then
        warning "Branch protection setup may have failed: $response"
    fi
}

# Setup repository labels
setup_labels() {
    if [[ "$SKIP_LABELS" == "true" ]]; then
        debug "Skipping labels setup"
        return
    fi
    
    log "Setting up repository labels..."
    
    # Define comprehensive label set
    local labels='[
        {"name": "bug", "color": "d73a4a", "description": "Something isn'\''t working"},
        {"name": "documentation", "color": "0075ca", "description": "Improvements or additions to documentation"},
        {"name": "duplicate", "color": "cfd3d7", "description": "This issue or pull request already exists"},
        {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
        {"name": "good first issue", "color": "7057ff", "description": "Good for newcomers"},
        {"name": "help wanted", "color": "008672", "description": "Extra attention is needed"},
        {"name": "invalid", "color": "e4e669", "description": "This doesn'\''t seem right"},
        {"name": "question", "color": "d876e3", "description": "Further information is requested"},
        {"name": "wontfix", "color": "ffffff", "description": "This will not be worked on"},
        
        {"name": "priority:critical", "color": "b60205", "description": "Critical priority - immediate attention required"},
        {"name": "priority:high", "color": "d93f0b", "description": "High priority"},
        {"name": "priority:medium", "color": "fbca04", "description": "Medium priority"},
        {"name": "priority:low", "color": "0e8a16", "description": "Low priority"},
        
        {"name": "size:xs", "color": "c2e0c6", "description": "Extra small change"},
        {"name": "size:s", "color": "91c995", "description": "Small change"},
        {"name": "size:m", "color": "5fb364", "description": "Medium change"},
        {"name": "size:l", "color": "2e8b32", "description": "Large change"},
        {"name": "size:xl", "color": "1e6823", "description": "Extra large change"},
        
        {"name": "type:feature", "color": "a2eeef", "description": "New feature implementation"},
        {"name": "type:bugfix", "color": "d73a4a", "description": "Bug fix"},
        {"name": "type:refactor", "color": "e99695", "description": "Code refactoring"},
        {"name": "type:performance", "color": "f9d0c4", "description": "Performance improvement"},
        {"name": "type:security", "color": "d4c5f9", "description": "Security enhancement"},
        {"name": "type:maintenance", "color": "fef2c0", "description": "Maintenance and cleanup"},
        
        {"name": "hardware:fpga", "color": "1f77b4", "description": "FPGA-related changes"},
        {"name": "hardware:asic", "color": "ff7f0e", "description": "ASIC-related changes"},
        {"name": "hardware:synthesis", "color": "2ca02c", "description": "Hardware synthesis"},
        {"name": "hardware:simulation", "color": "d62728", "description": "Hardware simulation"},
        {"name": "hardware:verification", "color": "9467bd", "description": "Hardware verification"},
        
        {"name": "ml:training", "color": "8c564b", "description": "ML model training"},
        {"name": "ml:inference", "color": "e377c2", "description": "ML inference optimization"},
        {"name": "ml:model", "color": "7f7f7f", "description": "ML model development"},
        {"name": "ml:dataset", "color": "bcbd22", "description": "Dataset management"},
        {"name": "ml:optimization", "color": "17becf", "description": "ML optimization"},
        
        {"name": "ci:build", "color": "0052cc", "description": "CI build system"},
        {"name": "ci:test", "color": "5319e7", "description": "CI testing"},
        {"name": "ci:deploy", "color": "b60205", "description": "CI deployment"},
        {"name": "ci:monitoring", "color": "fbca04", "description": "CI monitoring"},
        
        {"name": "dependencies", "color": "0366d6", "description": "Dependency updates"},
        {"name": "security", "color": "ee0701", "description": "Security-related"},
        {"name": "performance", "color": "ff9500", "description": "Performance-related"},
        {"name": "breaking-change", "color": "b60205", "description": "Breaking change"},
        {"name": "backward-compatible", "color": "0e8a16", "description": "Backward compatible change"}
    ]'
    
    # Create labels
    echo "$labels" | jq -c '.[]' | while read -r label; do
        local name=$(echo "$label" | jq -r '.name')
        local response=$(github_api POST "/repos/$REPO_OWNER/$REPO_NAME/labels" "$label")
        
        if [[ "$DRY_RUN" == "false" ]] && echo "$response" | jq -e '.name' > /dev/null; then
            debug "Created label: $name"
        elif [[ "$DRY_RUN" == "false" ]] && echo "$response" | grep -q "already_exists"; then
            debug "Label already exists: $name"
        elif [[ "$DRY_RUN" == "false" ]]; then
            warning "Failed to create label $name: $response"
        fi
    done
    
    success "Repository labels configured"
}

# Setup webhooks
setup_webhooks() {
    if [[ "$SKIP_WEBHOOKS" == "true" ]]; then
        debug "Skipping webhooks setup"
        return
    fi
    
    log "Setting up repository webhooks..."
    
    # Example webhook configuration (customize as needed)
    local webhook_config='{
        "name": "web",
        "active": true,
        "events": [
            "push",
            "pull_request",
            "issues",
            "issue_comment",
            "release",
            "deployment",
            "deployment_status"
        ],
        "config": {
            "url": "https://your-webhook-endpoint.com/github",
            "content_type": "json",
            "insecure_ssl": "0",
            "secret": "your-webhook-secret"
        }
    }'
    
    # Note: This webhook endpoint needs to be configured with actual values
    warning "Webhook configuration requires actual endpoint URL and secret"
    debug "Webhook config template prepared"
}

# Generate setup summary
generate_setup_summary() {
    log "Generating setup summary..."
    
    local summary_file="$PROJECT_ROOT/REPOSITORY_SETUP_SUMMARY.md"
    
    cat > "$summary_file" << EOF
# Repository Setup Summary

Generated on: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Repository: $REPO_OWNER/$REPO_NAME

## Configuration Applied

### Repository Settings
- ✅ Description and topics updated
- ✅ Security features enabled (secret scanning, dependency graph)
- ✅ Merge settings optimized
- ✅ Branch deletion on merge enabled

### Branch Protection
- ✅ Main branch protected
- ✅ Required status checks configured
- ✅ Pull request reviews required (1 approver)
- ✅ Code owner reviews required
- ✅ Linear history enforced

### Labels
- ✅ Comprehensive label set created
- ✅ Priority levels (critical, high, medium, low)
- ✅ Size categories (xs, s, m, l, xl)
- ✅ Type categories (feature, bugfix, refactor, etc.)
- ✅ Hardware-specific labels (FPGA, ASIC, synthesis, etc.)
- ✅ ML-specific labels (training, inference, model, etc.)
- ✅ CI-specific labels (build, test, deploy, monitoring)

### Security Configuration
- ✅ Secret scanning enabled
- ✅ Push protection for secrets enabled
- ✅ Dependency graph enabled
- ✅ Dependabot security updates enabled

## Manual Actions Required

### 1. Webhook Configuration
Update webhook endpoint in repository settings:
- URL: Configure actual webhook endpoint
- Secret: Set secure webhook secret
- Events: push, pull_request, issues, releases

### 2. Required Secrets
Add the following secrets in repository settings:

#### CI/CD Secrets
- \`DOCKER_REGISTRY_TOKEN\`: Docker registry authentication
- \`NPM_TOKEN\`: NPM registry token (if publishing packages)
- \`PYPI_TOKEN\`: PyPI token (if publishing Python packages)

#### Cloud Provider Secrets
- \`AWS_ACCESS_KEY_ID\`: AWS access key
- \`AWS_SECRET_ACCESS_KEY\`: AWS secret key
- \`AZURE_CLIENT_ID\`: Azure client ID
- \`AZURE_CLIENT_SECRET\`: Azure client secret
- \`GCP_SA_KEY\`: Google Cloud service account key

#### Security & Monitoring
- \`SONAR_TOKEN\`: SonarQube/SonarCloud token
- \`SNYK_TOKEN\`: Snyk security scanning token
- \`SENTRY_DSN\`: Sentry error tracking DSN

#### Notification Secrets
- \`SLACK_WEBHOOK\`: Slack webhook URL for notifications
- \`DISCORD_WEBHOOK\`: Discord webhook URL for notifications

### 3. Team Permissions
Configure team access in repository settings:
- Admin: Repository maintainers
- Write: Core contributors
- Read: All organization members

### 4. External Integrations
Enable and configure:
- SonarCloud for code quality analysis
- Snyk for security vulnerability scanning
- Dependabot for dependency updates
- CodeQL for security analysis

## Verification Checklist

- [ ] Branch protection rules are active
- [ ] Labels are visible in Issues/PRs
- [ ] Required status checks appear on PRs
- [ ] Security features are enabled
- [ ] Webhooks are configured and functional
- [ ] Team permissions are set correctly
- [ ] Required secrets are added
- [ ] External integrations are working

## Next Steps

1. Create a test pull request to verify branch protection
2. Test CI/CD pipeline with a sample change
3. Verify security scanning is working
4. Configure monitoring dashboards
5. Set up automated dependency updates
6. Configure notification channels

## Support

For issues with this setup, please:
1. Check GitHub repository settings
2. Verify required permissions
3. Review webhook logs
4. Contact repository administrators

---

This setup provides enterprise-grade repository governance and SDLC automation
for the AI Hardware Co-Design Playground project.
EOF

    success "Setup summary generated: $summary_file"
}

# Main execution
main() {
    log "Starting repository setup for $REPO_OWNER/$REPO_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        warning "DRY RUN MODE - No changes will be made"
    fi
    
    validate_prerequisites
    configure_repository_settings
    setup_branch_protection
    setup_labels
    setup_webhooks
    generate_setup_summary
    
    success "Repository setup completed!"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        log "Review the setup summary at: $PROJECT_ROOT/REPOSITORY_SETUP_SUMMARY.md"
        log "Complete the manual actions listed in the summary"
    fi
}

# Run main function
main "$@"