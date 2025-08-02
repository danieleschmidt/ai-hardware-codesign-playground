#!/bin/bash

# AI Hardware Co-Design Playground - Repository Health Check
# Comprehensive health monitoring and automated issue detection

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HEALTH_REPORT="$PROJECT_ROOT/health-report.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Health check results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    ((FAILED_CHECKS++))
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_CHECKS++))
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNING_CHECKS++))
}

# Increment total checks
check() {
    ((TOTAL_CHECKS++))
}

# Initialize health report
init_health_report() {
    cat > "$HEALTH_REPORT" << EOF
# Repository Health Check Report

**Generated:** $TIMESTAMP  
**Repository:** ai-hardware-codesign-playground  
**Checker:** $(whoami)@$(hostname)

## Summary

EOF
}

# Check Git repository health
check_git_health() {
    log "Checking Git repository health..."
    
    cd "$PROJECT_ROOT"
    
    # Check if we're in a git repository
    check
    if git rev-parse --git-dir > /dev/null 2>&1; then
        success "Git repository is properly initialized"
        echo "- ✅ Git repository properly initialized" >> "$HEALTH_REPORT"
    else
        error "Not in a Git repository"
        echo "- ❌ Not in a Git repository" >> "$HEALTH_REPORT"
        return
    fi
    
    # Check for uncommitted changes
    check
    if git diff-index --quiet HEAD --; then
        success "No uncommitted changes in working directory"
        echo "- ✅ Working directory clean" >> "$HEALTH_REPORT"
    else
        warning "Uncommitted changes detected"
        echo "- ⚠️ Uncommitted changes detected" >> "$HEALTH_REPORT"
    fi
    
    # Check remote configuration
    check
    if git remote get-url origin > /dev/null 2>&1; then
        success "Remote origin configured"
        echo "- ✅ Remote origin configured" >> "$HEALTH_REPORT"
    else
        warning "No remote origin configured"
        echo "- ⚠️ No remote origin configured" >> "$HEALTH_REPORT"
    fi
    
    # Check branch protection (simulated - would need GitHub API)
    check
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" == "main" ]] || [[ "$current_branch" == "master" ]]; then
        warning "Working directly on main branch - consider using feature branches"
        echo "- ⚠️ Working on main branch (consider feature branches)" >> "$HEALTH_REPORT"
    else
        success "Working on feature branch: $current_branch"
        echo "- ✅ Working on feature branch: $current_branch" >> "$HEALTH_REPORT"
    fi
    
    echo "" >> "$HEALTH_REPORT"
}

# Check SDLC files and structure
check_sdlc_structure() {
    log "Checking SDLC structure..."
    
    echo "### SDLC Structure" >> "$HEALTH_REPORT"
    
    # Required files
    local required_files=(
        "README.md"
        "LICENSE"
        "CONTRIBUTING.md"
        "CODE_OF_CONDUCT.md"
        "SECURITY.md"
        ".gitignore"
        "requirements.txt"
        "package.json"
    )
    
    for file in "${required_files[@]}"; do
        check
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            success "Found required file: $file"
            echo "- ✅ $file" >> "$HEALTH_REPORT"
        else
            error "Missing required file: $file"
            echo "- ❌ Missing: $file" >> "$HEALTH_REPORT"
        fi
    done
    
    # Required directories
    local required_dirs=(
        ".github"
        "docs"
        "tests"
        "scripts"
    )
    
    for dir in "${required_dirs[@]}"; do
        check
        if [[ -d "$PROJECT_ROOT/$dir" ]]; then
            success "Found required directory: $dir"
            echo "- ✅ $dir/" >> "$HEALTH_REPORT"
        else
            error "Missing required directory: $dir"
            echo "- ❌ Missing: $dir/" >> "$HEALTH_REPORT"
        fi
    done
    
    echo "" >> "$HEALTH_REPORT"
}

# Check CI/CD workflows
check_cicd_workflows() {
    log "Checking CI/CD workflows..."
    
    echo "### CI/CD Workflows" >> "$HEALTH_REPORT"
    
    local workflow_dir="$PROJECT_ROOT/.github/workflows"
    
    check
    if [[ -d "$workflow_dir" ]]; then
        success "GitHub Actions workflows directory exists"
        echo "- ✅ Workflows directory exists" >> "$HEALTH_REPORT"
        
        local workflow_count=$(find "$workflow_dir" -name "*.yml" -o -name "*.yaml" | wc -l)
        if [[ $workflow_count -gt 0 ]]; then
            success "Found $workflow_count workflow files"
            echo "- ✅ $workflow_count workflow files found" >> "$HEALTH_REPORT"
            
            # List workflows
            find "$workflow_dir" -name "*.yml" -o -name "*.yaml" | while read -r workflow; do
                local name=$(basename "$workflow" .yml)
                name=$(basename "$name" .yaml)
                echo "  - $name" >> "$HEALTH_REPORT"
            done
        else
            warning "No workflow files found"
            echo "- ⚠️ No workflow files found" >> "$HEALTH_REPORT"
        fi
    else
        error "GitHub Actions workflows directory missing"
        echo "- ❌ Workflows directory missing" >> "$HEALTH_REPORT"
    fi
    
    # Check for essential workflows
    local essential_workflows=("ci.yml" "security.yml" "release.yml")
    
    for workflow in "${essential_workflows[@]}"; do
        check
        if [[ -f "$workflow_dir/$workflow" ]]; then
            success "Essential workflow found: $workflow"
            echo "- ✅ Essential workflow: $workflow" >> "$HEALTH_REPORT"
        else
            warning "Essential workflow missing: $workflow"
            echo "- ⚠️ Missing essential workflow: $workflow" >> "$HEALTH_REPORT"
        fi
    done
    
    echo "" >> "$HEALTH_REPORT"
}

# Check dependencies and security
check_dependencies() {
    log "Checking dependencies and security..."
    
    echo "### Dependencies & Security" >> "$HEALTH_REPORT"
    
    cd "$PROJECT_ROOT"
    
    # Check Python dependencies
    check
    if [[ -f "requirements.txt" ]]; then
        success "Python requirements.txt found"
        echo "- ✅ Python requirements.txt found" >> "$HEALTH_REPORT"
        
        local python_deps=$(wc -l < requirements.txt)
        echo "  - $python_deps Python dependencies" >> "$HEALTH_REPORT"
        
        # Check for security vulnerabilities (if safety is installed)
        if command -v safety > /dev/null 2>&1; then
            check
            if safety check -r requirements.txt > /dev/null 2>&1; then
                success "No known Python security vulnerabilities"
                echo "- ✅ No known Python vulnerabilities" >> "$HEALTH_REPORT"
            else
                error "Python security vulnerabilities detected"
                echo "- ❌ Python vulnerabilities detected" >> "$HEALTH_REPORT"
            fi
        else
            check
            warning "Safety not installed - cannot check Python vulnerabilities"
            echo "- ⚠️ Cannot check Python vulnerabilities (safety not installed)" >> "$HEALTH_REPORT"
        fi
    else
        error "Python requirements.txt missing"
        echo "- ❌ Python requirements.txt missing" >> "$HEALTH_REPORT"
    fi
    
    # Check Node.js dependencies
    check
    if [[ -f "package.json" ]]; then
        success "Node.js package.json found"
        echo "- ✅ Node.js package.json found" >> "$HEALTH_REPORT"
        
        if [[ -f "package-lock.json" ]]; then
            success "Package lock file found"
            echo "- ✅ package-lock.json found" >> "$HEALTH_REPORT"
        else
            warning "Package lock file missing"
            echo "- ⚠️ package-lock.json missing" >> "$HEALTH_REPORT"
        fi
        
        # Check for security vulnerabilities
        if command -v npm > /dev/null 2>&1; then
            check
            if npm audit > /dev/null 2>&1; then
                success "No critical Node.js security vulnerabilities"
                echo "- ✅ No critical Node.js vulnerabilities" >> "$HEALTH_REPORT"
            else
                warning "Node.js security issues detected"
                echo "- ⚠️ Node.js security issues detected" >> "$HEALTH_REPORT"
            fi
        fi
    else
        warning "Node.js package.json missing"
        echo "- ⚠️ Node.js package.json missing" >> "$HEALTH_REPORT"
    fi
    
    echo "" >> "$HEALTH_REPORT"
}

# Check test coverage and quality
check_testing() {
    log "Checking testing setup..."
    
    echo "### Testing & Quality" >> "$HEALTH_REPORT"
    
    cd "$PROJECT_ROOT"
    
    # Check test directory
    check
    if [[ -d "tests" ]]; then
        success "Tests directory found"
        echo "- ✅ Tests directory exists" >> "$HEALTH_REPORT"
        
        local test_files=$(find tests -name "*.py" -o -name "*.js" -o -name "*.ts" | wc -l)
        echo "  - $test_files test files found" >> "$HEALTH_REPORT"
        
        # Check for different types of tests
        local test_types=("unit" "integration" "e2e" "performance")
        for test_type in "${test_types[@]}"; do
            if [[ -d "tests/$test_type" ]]; then
                success "Found $test_type tests"
                echo "  - ✅ $test_type tests" >> "$HEALTH_REPORT"
            else
                warning "Missing $test_type tests directory"
                echo "  - ⚠️ Missing $test_type tests" >> "$HEALTH_REPORT"
            fi
        done
    else
        error "Tests directory missing"
        echo "- ❌ Tests directory missing" >> "$HEALTH_REPORT"
    fi
    
    # Check for test configuration
    local test_configs=("pytest.ini" "jest.config.js" ".coveragerc")
    for config in "${test_configs[@]}"; do
        check
        if [[ -f "$config" ]]; then
            success "Test configuration found: $config"
            echo "- ✅ Test config: $config" >> "$HEALTH_REPORT"
        else
            warning "Test configuration missing: $config"
            echo "- ⚠️ Missing test config: $config" >> "$HEALTH_REPORT"
        fi
    done
    
    echo "" >> "$HEALTH_REPORT"
}

# Check documentation
check_documentation() {
    log "Checking documentation..."
    
    echo "### Documentation" >> "$HEALTH_REPORT"
    
    cd "$PROJECT_ROOT"
    
    # Check README quality
    check
    if [[ -f "README.md" ]]; then
        local readme_size=$(wc -l < README.md)
        if [[ $readme_size -gt 20 ]]; then
            success "README.md is comprehensive ($readme_size lines)"
            echo "- ✅ Comprehensive README.md ($readme_size lines)" >> "$HEALTH_REPORT"
        else
            warning "README.md is too brief ($readme_size lines)"
            echo "- ⚠️ Brief README.md ($readme_size lines)" >> "$HEALTH_REPORT"
        fi
    fi
    
    # Check docs directory
    check
    if [[ -d "docs" ]]; then
        success "Documentation directory found"
        echo "- ✅ Documentation directory exists" >> "$HEALTH_REPORT"
        
        local doc_files=$(find docs -name "*.md" | wc -l)
        echo "  - $doc_files documentation files" >> "$HEALTH_REPORT"
    else
        warning "Documentation directory missing"
        echo "- ⚠️ Documentation directory missing" >> "$HEALTH_REPORT"
    fi
    
    # Check for API documentation
    local api_docs=("docs/api" "api.md" "API.md")
    local found_api_docs=false
    for api_doc in "${api_docs[@]}"; do
        if [[ -f "$api_doc" ]] || [[ -d "$api_doc" ]]; then
            found_api_docs=true
            break
        fi
    done
    
    check
    if [[ "$found_api_docs" == "true" ]]; then
        success "API documentation found"
        echo "- ✅ API documentation found" >> "$HEALTH_REPORT"
    else
        warning "API documentation missing"
        echo "- ⚠️ API documentation missing" >> "$HEALTH_REPORT"
    fi
    
    echo "" >> "$HEALTH_REPORT"
}

# Check Docker setup
check_docker_setup() {
    log "Checking Docker setup..."
    
    echo "### Docker Setup" >> "$HEALTH_REPORT"
    
    cd "$PROJECT_ROOT"
    
    # Check Dockerfile
    check
    if [[ -f "Dockerfile" ]]; then
        success "Dockerfile found"
        echo "- ✅ Dockerfile found" >> "$HEALTH_REPORT"
        
        # Check for best practices
        if grep -q "FROM.*:latest" Dockerfile; then
            warning "Dockerfile uses :latest tag (not recommended)"
            echo "  - ⚠️ Uses :latest tag" >> "$HEALTH_REPORT"
        else
            success "Dockerfile uses specific tags"
            echo "  - ✅ Uses specific image tags" >> "$HEALTH_REPORT"
        fi
        
        if grep -q "USER" Dockerfile; then
            success "Dockerfile specifies non-root user"
            echo "  - ✅ Non-root user specified" >> "$HEALTH_REPORT"
        else
            warning "Dockerfile doesn't specify non-root user"
            echo "  - ⚠️ No non-root user specified" >> "$HEALTH_REPORT"
        fi
    else
        warning "Dockerfile missing"
        echo "- ⚠️ Dockerfile missing" >> "$HEALTH_REPORT"
    fi
    
    # Check docker-compose
    check
    if [[ -f "docker-compose.yml" ]] || [[ -f "docker-compose.yaml" ]]; then
        success "Docker Compose configuration found"
        echo "- ✅ Docker Compose found" >> "$HEALTH_REPORT"
    else
        warning "Docker Compose configuration missing"
        echo "- ⚠️ Docker Compose missing" >> "$HEALTH_REPORT"
    fi
    
    # Check .dockerignore
    check
    if [[ -f ".dockerignore" ]]; then
        success ".dockerignore found"
        echo "- ✅ .dockerignore found" >> "$HEALTH_REPORT"
    else
        warning ".dockerignore missing"
        echo "- ⚠️ .dockerignore missing" >> "$HEALTH_REPORT"
    fi
    
    echo "" >> "$HEALTH_REPORT"
}

# Generate final health summary
generate_health_summary() {
    log "Generating health summary..."
    
    local pass_rate=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))
    local health_status="Unknown"
    local health_color=""
    
    if [[ $pass_rate -ge 90 ]]; then
        health_status="Excellent"
        health_color="🟢"
    elif [[ $pass_rate -ge 75 ]]; then
        health_status="Good"
        health_color="🟡"
    elif [[ $pass_rate -ge 50 ]]; then
        health_status="Fair"
        health_color="🟠"
    else
        health_status="Poor"
        health_color="🔴"
    fi
    
    # Update the summary section
    sed -i "/## Summary/a\\
\\
$health_color **Overall Health: $health_status ($pass_rate%)**\\
\\
- **Total Checks:** $TOTAL_CHECKS\\
- **Passed:** $PASSED_CHECKS\\
- **Warnings:** $WARNING_CHECKS\\
- **Failed:** $FAILED_CHECKS\\
\\
---\\
" "$HEALTH_REPORT"
    
    # Add recommendations
    cat >> "$HEALTH_REPORT" << EOF

## Recommendations

EOF
    
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        echo "### Critical Issues (Fix Immediately)" >> "$HEALTH_REPORT"
        echo "- Address all failed checks above" >> "$HEALTH_REPORT"
        echo "- Failed checks indicate missing essential components" >> "$HEALTH_REPORT"
        echo "" >> "$HEALTH_REPORT"
    fi
    
    if [[ $WARNING_CHECKS -gt 0 ]]; then
        echo "### Improvements (Recommended)" >> "$HEALTH_REPORT"
        echo "- Review warnings to enhance repository quality" >> "$HEALTH_REPORT"
        echo "- Warnings indicate areas for improvement" >> "$HEALTH_REPORT"
        echo "" >> "$HEALTH_REPORT"
    fi
    
    echo "### General Recommendations" >> "$HEALTH_REPORT"
    echo "- Run this health check regularly (weekly recommended)" >> "$HEALTH_REPORT"
    echo "- Keep dependencies updated and secure" >> "$HEALTH_REPORT"
    echo "- Maintain comprehensive test coverage" >> "$HEALTH_REPORT"
    echo "- Document new features and changes" >> "$HEALTH_REPORT"
    echo "- Follow semantic versioning for releases" >> "$HEALTH_REPORT"
    
    success "Health check completed - Overall status: $health_status ($pass_rate%)"
    log "Detailed report available at: $HEALTH_REPORT"
}

# Main execution
main() {
    log "Starting repository health check..."
    
    # Initialize report
    init_health_report
    
    # Run all health checks
    check_git_health
    check_sdlc_structure
    check_cicd_workflows
    check_dependencies
    check_testing
    check_documentation
    check_docker_setup
    
    # Generate summary
    generate_health_summary
    
    # Exit with appropriate code
    if [[ $FAILED_CHECKS -gt 0 ]]; then
        exit 1
    elif [[ $WARNING_CHECKS -gt 0 ]]; then
        exit 2
    else
        exit 0
    fi
}

# Run main function
main "$@"