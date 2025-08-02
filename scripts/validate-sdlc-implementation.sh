#!/bin/bash

# AI Hardware Co-Design Playground - SDLC Implementation Validator
# Comprehensive validation of all SDLC checkpoints and components

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VALIDATION_REPORT="$PROJECT_ROOT/sdlc-validation-report.md"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Validation results
TOTAL_VALIDATIONS=0
PASSED_VALIDATIONS=0
FAILED_VALIDATIONS=0
WARNING_VALIDATIONS=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[FAIL]${NC} $1" >&2
    ((FAILED_VALIDATIONS++))
}

success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_VALIDATIONS++))
}

warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNING_VALIDATIONS++))
}

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Increment total validations
validate() {
    ((TOTAL_VALIDATIONS++))
}

# Initialize validation report
init_validation_report() {
    cat > "$VALIDATION_REPORT" << EOF
# SDLC Implementation Validation Report

**Generated:** $TIMESTAMP  
**Repository:** ai-hardware-codesign-playground  
**Validator:** $(whoami)@$(hostname)

## Executive Summary

This report validates the comprehensive SDLC implementation against enterprise-grade standards and best practices.

## Validation Results

EOF
}

# Validate Checkpoint 1: Project Foundation & Documentation
validate_checkpoint_1() {
    log "Validating Checkpoint 1: Project Foundation & Documentation"
    echo "### Checkpoint 1: Project Foundation & Documentation" >> "$VALIDATION_REPORT"
    
    local checkpoint_score=0
    local checkpoint_total=0
    
    # Core documentation files
    local required_docs=(
        "README.md:Primary documentation"
        "ARCHITECTURE.md:System architecture"
        "PROJECT_CHARTER.md:Project charter"
        "LICENSE:License file"
        "CODE_OF_CONDUCT.md:Code of conduct"
        "CONTRIBUTING.md:Contribution guidelines"
        "SECURITY.md:Security policy"
        "CHANGELOG.md:Change log"
    )
    
    for doc_info in "${required_docs[@]}"; do
        IFS=':' read -r file description <<< "$doc_info"
        validate
        ((checkpoint_total++))
        
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            success "$description found: $file"
            echo "- ✅ $description ($file)" >> "$VALIDATION_REPORT"
            ((checkpoint_score++))
        else
            error "$description missing: $file"
            echo "- ❌ Missing $description ($file)" >> "$VALIDATION_REPORT"
        fi
    done
    
    # ADR structure
    validate
    ((checkpoint_total++))
    if [[ -d "$PROJECT_ROOT/docs/adr" ]]; then
        local adr_count=$(find "$PROJECT_ROOT/docs/adr" -name "*.md" | wc -l)
        if [[ $adr_count -gt 0 ]]; then
            success "Architecture Decision Records found: $adr_count ADRs"
            echo "- ✅ Architecture Decision Records ($adr_count ADRs)" >> "$VALIDATION_REPORT"
            ((checkpoint_score++))
        else
            warning "ADR directory exists but no ADRs found"
            echo "- ⚠️ ADR directory empty" >> "$VALIDATION_REPORT"
        fi
    else
        error "ADR directory missing"
        echo "- ❌ Missing ADR directory" >> "$VALIDATION_REPORT"
    fi
    
    # Community files validation
    validate
    ((checkpoint_total++))
    local community_score=0
    local community_total=4
    
    [[ -f "$PROJECT_ROOT/LICENSE" ]] && ((community_score++))
    [[ -f "$PROJECT_ROOT/CODE_OF_CONDUCT.md" ]] && ((community_score++))
    [[ -f "$PROJECT_ROOT/CONTRIBUTING.md" ]] && ((community_score++))
    [[ -f "$PROJECT_ROOT/SECURITY.md" ]] && ((community_score++))
    
    if [[ $community_score -eq $community_total ]]; then
        success "All community files present"
        echo "- ✅ Complete community file set" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "Some community files missing ($community_score/$community_total)"
        echo "- ⚠️ Incomplete community files ($community_score/$community_total)" >> "$VALIDATION_REPORT"
    fi
    
    local checkpoint_percentage=$((checkpoint_score * 100 / checkpoint_total))
    echo "**Checkpoint 1 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Checkpoint 1 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)"
}

# Validate Checkpoint 2: Development Environment & Tooling
validate_checkpoint_2() {
    log "Validating Checkpoint 2: Development Environment & Tooling"
    echo "### Checkpoint 2: Development Environment & Tooling" >> "$VALIDATION_REPORT"
    
    local checkpoint_score=0
    local checkpoint_total=0
    
    # Development environment files
    local dev_files=(
        ".devcontainer/devcontainer.json:DevContainer configuration"
        ".env.example:Environment variables template"
        ".editorconfig:Editor configuration"
        ".gitignore:Git ignore rules"
        "package.json:Node.js package configuration"
        "requirements.txt:Python requirements"
        "pyproject.toml:Python project configuration"
    )
    
    for file_info in "${dev_files[@]}"; do
        IFS=':' read -r file description <<< "$file_info"
        validate
        ((checkpoint_total++))
        
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            success "$description found: $file"
            echo "- ✅ $description ($file)" >> "$VALIDATION_REPORT"
            ((checkpoint_score++))
        else
            warning "$description missing: $file"
            echo "- ⚠️ Missing $description ($file)" >> "$VALIDATION_REPORT"
        fi
    done
    
    # Code quality tools
    validate
    ((checkpoint_total++))
    if [[ -f "$PROJECT_ROOT/.pre-commit-config.yaml" ]]; then
        success "Pre-commit configuration found"
        echo "- ✅ Pre-commit hooks configured" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        error "Pre-commit configuration missing"
        echo "- ❌ Missing pre-commit configuration" >> "$VALIDATION_REPORT"
    fi
    
    # VSCode settings
    validate
    ((checkpoint_total++))
    if [[ -f "$PROJECT_ROOT/.vscode/settings.json" ]]; then
        success "VSCode settings found"
        echo "- ✅ VSCode workspace configured" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "VSCode settings missing"
        echo "- ⚠️ Missing VSCode configuration" >> "$VALIDATION_REPORT"
    fi
    
    local checkpoint_percentage=$((checkpoint_score * 100 / checkpoint_total))
    echo "**Checkpoint 2 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Checkpoint 2 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)"
}

# Validate Checkpoint 3: Testing Infrastructure
validate_checkpoint_3() {
    log "Validating Checkpoint 3: Testing Infrastructure"
    echo "### Checkpoint 3: Testing Infrastructure" >> "$VALIDATION_REPORT"
    
    local checkpoint_score=0
    local checkpoint_total=0
    
    # Test directories
    local test_dirs=(
        "tests:Main test directory"
        "tests/unit:Unit tests"
        "tests/integration:Integration tests"
        "tests/e2e:End-to-end tests"
        "tests/performance:Performance tests"
    )
    
    for dir_info in "${test_dirs[@]}"; do
        IFS=':' read -r dir description <<< "$dir_info"
        validate
        ((checkpoint_total++))
        
        if [[ -d "$PROJECT_ROOT/$dir" ]]; then
            success "$description found: $dir"
            echo "- ✅ $description ($dir)" >> "$VALIDATION_REPORT"
            ((checkpoint_score++))
        else
            error "$description missing: $dir"
            echo "- ❌ Missing $description ($dir)" >> "$VALIDATION_REPORT"
        fi
    done
    
    # Test configuration files
    local test_configs=(
        "pytest.ini:Pytest configuration"
        ".coveragerc:Coverage configuration"
    )
    
    for config_info in "${test_configs[@]}"; do
        IFS=':' read -r file description <<< "$config_info"
        validate
        ((checkpoint_total++))
        
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            success "$description found: $file"
            echo "- ✅ $description ($file)" >> "$VALIDATION_REPORT"
            ((checkpoint_score++))
        else
            warning "$description missing: $file"
            echo "- ⚠️ Missing $description ($file)" >> "$VALIDATION_REPORT"
        fi
    done
    
    # Performance testing tools
    validate
    ((checkpoint_total++))
    if [[ -f "$PROJECT_ROOT/tests/performance/artillery.yml" ]]; then
        success "Performance testing configured (Artillery)"
        echo "- ✅ Performance testing configured" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "Performance testing configuration missing"
        echo "- ⚠️ Missing performance testing configuration" >> "$VALIDATION_REPORT"
    fi
    
    local checkpoint_percentage=$((checkpoint_score * 100 / checkpoint_total))
    echo "**Checkpoint 3 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Checkpoint 3 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)"
}

# Validate Checkpoint 4: Build & Containerization
validate_checkpoint_4() {
    log "Validating Checkpoint 4: Build & Containerization"
    echo "### Checkpoint 4: Build & Containerization" >> "$VALIDATION_REPORT"
    
    local checkpoint_score=0
    local checkpoint_total=0
    
    # Container files
    local container_files=(
        "Dockerfile:Docker container definition"
        "docker-compose.yml:Docker Compose configuration"
        ".dockerignore:Docker ignore rules"
    )
    
    for file_info in "${container_files[@]}"; do
        IFS=':' read -r file description <<< "$file_info"
        validate
        ((checkpoint_total++))
        
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            success "$description found: $file"
            echo "- ✅ $description ($file)" >> "$VALIDATION_REPORT"
            ((checkpoint_score++))
        else
            warning "$description missing: $file"
            echo "- ⚠️ Missing $description ($file)" >> "$VALIDATION_REPORT"
        fi
    done
    
    # Build system
    validate
    ((checkpoint_total++))
    if [[ -f "$PROJECT_ROOT/Makefile" ]]; then
        success "Build system found: Makefile"
        echo "- ✅ Build system configured (Makefile)" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "Build system missing"
        echo "- ⚠️ Missing build system" >> "$VALIDATION_REPORT"
    fi
    
    # Semantic release
    validate
    ((checkpoint_total++))
    if [[ -f "$PROJECT_ROOT/semantic-release.config.js" ]]; then
        success "Semantic release configured"
        echo "- ✅ Semantic release configured" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "Semantic release configuration missing"
        echo "- ⚠️ Missing semantic release configuration" >> "$VALIDATION_REPORT"
    fi
    
    local checkpoint_percentage=$((checkpoint_score * 100 / checkpoint_total))
    echo "**Checkpoint 4 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Checkpoint 4 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)"
}

# Validate Checkpoint 5: Monitoring & Observability Setup
validate_checkpoint_5() {
    log "Validating Checkpoint 5: Monitoring & Observability Setup"
    echo "### Checkpoint 5: Monitoring & Observability Setup" >> "$VALIDATION_REPORT"
    
    local checkpoint_score=0
    local checkpoint_total=0
    
    # Monitoring configuration
    validate
    ((checkpoint_total++))
    if [[ -d "$PROJECT_ROOT/monitoring" ]]; then
        success "Monitoring directory found"
        echo "- ✅ Monitoring directory exists" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
        
        # Check for specific monitoring files
        local monitoring_files=(
            "monitoring/prometheus.yml"
            "monitoring/alert_rules.yml"
        )
        
        for file in "${monitoring_files[@]}"; do
            if [[ -f "$PROJECT_ROOT/$file" ]]; then
                echo "  - ✅ $(basename "$file")" >> "$VALIDATION_REPORT"
            else
                echo "  - ⚠️ Missing $(basename "$file")" >> "$VALIDATION_REPORT"
            fi
        done
    else
        error "Monitoring directory missing"
        echo "- ❌ Missing monitoring directory" >> "$VALIDATION_REPORT"
    fi
    
    # Grafana dashboards
    validate
    ((checkpoint_total++))
    if [[ -d "$PROJECT_ROOT/monitoring/grafana-dashboards" ]]; then
        local dashboard_count=$(find "$PROJECT_ROOT/monitoring/grafana-dashboards" -name "*.json" | wc -l)
        success "Grafana dashboards found: $dashboard_count dashboards"
        echo "- ✅ Grafana dashboards ($dashboard_count dashboards)" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "Grafana dashboards missing"
        echo "- ⚠️ Missing Grafana dashboards" >> "$VALIDATION_REPORT"
    fi
    
    # Health check scripts
    validate
    ((checkpoint_total++))
    if [[ -f "$PROJECT_ROOT/scripts/health-check.sh" ]]; then
        success "Health check script found"
        echo "- ✅ Health check script configured" >> "$VALIDATION_REPORT"
        ((checkpoint_score++))
    else
        warning "Health check script missing"
        echo "- ⚠️ Missing health check script" >> "$VALIDATION_REPORT"
    fi
    
    local checkpoint_percentage=$((checkpoint_score * 100 / checkpoint_total))
    echo "**Checkpoint 5 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Checkpoint 5 Score: $checkpoint_percentage% ($checkpoint_score/$checkpoint_total)"
}

# Validate GitHub Actions Workflows
validate_github_workflows() {
    log "Validating GitHub Actions Workflows"
    echo "### GitHub Actions Workflows" >> "$VALIDATION_REPORT"
    
    local workflow_score=0
    local workflow_total=0
    
    # Essential workflows
    local essential_workflows=(
        "ci.yml:Continuous Integration"
        "security.yml:Security Scanning"
        "release.yml:Release Automation"
        "performance.yml:Performance Testing"
        "dependency-update.yml:Dependency Management"
    )
    
    for workflow_info in "${essential_workflows[@]}"; do
        IFS=':' read -r file description <<< "$workflow_info"
        validate
        ((workflow_total++))
        
        if [[ -f "$PROJECT_ROOT/.github/workflows/$file" ]]; then
            success "$description workflow found: $file"
            echo "- ✅ $description ($file)" >> "$VALIDATION_REPORT"
            ((workflow_score++))
        else
            error "$description workflow missing: $file"
            echo "- ❌ Missing $description ($file)" >> "$VALIDATION_REPORT"
        fi
    done
    
    # Workflow syntax validation (basic check)
    validate
    ((workflow_total++))
    local syntax_errors=0
    
    if command -v yamllint > /dev/null 2>&1; then
        for workflow in "$PROJECT_ROOT/.github/workflows"/*.yml "$PROJECT_ROOT/.github/workflows"/*.yaml; do
            if [[ -f "$workflow" ]]; then
                if ! yamllint "$workflow" > /dev/null 2>&1; then
                    ((syntax_errors++))
                fi
            fi
        done
        
        if [[ $syntax_errors -eq 0 ]]; then
            success "All workflows have valid YAML syntax"
            echo "- ✅ Workflow syntax validation passed" >> "$VALIDATION_REPORT"
            ((workflow_score++))
        else
            error "$syntax_errors workflows have syntax errors"
            echo "- ❌ $syntax_errors workflows with syntax errors" >> "$VALIDATION_REPORT"
        fi
    else
        warning "yamllint not available - skipping syntax validation"
        echo "- ⚠️ Workflow syntax validation skipped (yamllint not available)" >> "$VALIDATION_REPORT"
    fi
    
    local workflow_percentage=$((workflow_score * 100 / workflow_total))
    echo "**Workflows Score: $workflow_percentage% ($workflow_score/$workflow_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Workflows Score: $workflow_percentage% ($workflow_score/$workflow_total)"
}

# Validate Automation Scripts
validate_automation_scripts() {
    log "Validating Automation Scripts"
    echo "### Automation Scripts" >> "$VALIDATION_REPORT"
    
    local script_score=0
    local script_total=0
    
    # Essential scripts
    local essential_scripts=(
        "scripts/collect-metrics.sh:Metrics collection"
        "scripts/repository-health-check.sh:Repository health check"
        "scripts/generate-dashboard.py:Dashboard generation"
        "scripts/validate-sdlc-implementation.sh:SDLC validation"
        "scripts/setup.sh:Environment setup"
        "scripts/health-check.sh:Health monitoring"
    )
    
    for script_info in "${essential_scripts[@]}"; do
        IFS=':' read -r file description <<< "$script_info"
        validate
        ((script_total++))
        
        if [[ -f "$PROJECT_ROOT/$file" ]]; then
            if [[ -x "$PROJECT_ROOT/$file" ]]; then
                success "$description script found and executable: $file"
                echo "- ✅ $description ($file)" >> "$VALIDATION_REPORT"
                ((script_score++))
            else
                warning "$description script found but not executable: $file"
                echo "- ⚠️ $description not executable ($file)" >> "$VALIDATION_REPORT"
            fi
        else
            error "$description script missing: $file"
            echo "- ❌ Missing $description ($file)" >> "$VALIDATION_REPORT"
        fi
    done
    
    local script_percentage=$((script_score * 100 / script_total))
    echo "**Scripts Score: $script_percentage% ($script_score/$script_total)**" >> "$VALIDATION_REPORT"
    echo "" >> "$VALIDATION_REPORT"
    
    info "Scripts Score: $script_percentage% ($script_score/$script_total)"
}

# Generate final validation summary
generate_validation_summary() {
    log "Generating validation summary..."
    
    local overall_percentage=$((PASSED_VALIDATIONS * 100 / TOTAL_VALIDATIONS))
    local validation_status="Unknown"
    local status_emoji=""
    
    if [[ $overall_percentage -ge 95 ]]; then
        validation_status="Excellent"
        status_emoji="🟢"
    elif [[ $overall_percentage -ge 85 ]]; then
        validation_status="Good"
        status_emoji="🟡"
    elif [[ $overall_percentage -ge 70 ]]; then
        validation_status="Fair"
        status_emoji="🟠"
    else
        validation_status="Poor"
        status_emoji="🔴"
    fi
    
    # Update the summary section
    sed -i "/## Validation Results/a\\
\\
$status_emoji **Overall SDLC Implementation: $validation_status ($overall_percentage%)**\\
\\
### Summary Statistics\\
- **Total Validations:** $TOTAL_VALIDATIONS\\
- **Passed:** $PASSED_VALIDATIONS\\
- **Warnings:** $WARNING_VALIDATIONS\\
- **Failed:** $FAILED_VALIDATIONS\\
\\
### Maturity Assessment\\
- **Current Level:** $(get_maturity_level $overall_percentage)\\
- **Target Level:** Enterprise-Grade SDLC\\
- **Gap Analysis:** $(get_gap_analysis $overall_percentage)\\
\\
---\\
" "$VALIDATION_REPORT"
    
    # Add recommendations
    cat >> "$VALIDATION_REPORT" << EOF

## Recommendations

### Immediate Actions
EOF
    
    if [[ $FAILED_VALIDATIONS -gt 0 ]]; then
        echo "- **Priority 1:** Address all failed validations immediately" >> "$VALIDATION_REPORT"
        echo "- **Impact:** Failed validations indicate missing critical components" >> "$VALIDATION_REPORT"
    fi
    
    if [[ $WARNING_VALIDATIONS -gt 0 ]]; then
        echo "- **Priority 2:** Review and address warning items" >> "$VALIDATION_REPORT"
        echo "- **Impact:** Warnings indicate opportunities for improvement" >> "$VALIDATION_REPORT"
    fi
    
    cat >> "$VALIDATION_REPORT" << EOF

### Next Steps
1. **Review Detailed Results:** Address each failed validation above
2. **Implement Missing Components:** Focus on critical infrastructure gaps
3. **Enhance Existing Components:** Improve warning items for better quality
4. **Regular Validation:** Run this validator weekly to maintain standards
5. **Continuous Improvement:** Regularly update SDLC practices

### Success Criteria
- **Minimum Target:** 90% validation pass rate
- **Excellence Target:** 95% validation pass rate
- **Enterprise Standard:** 98% validation pass rate with zero critical failures

## Conclusion

This SDLC implementation represents a **$validation_status** level of maturity with **$overall_percentage%** compliance to enterprise standards.

$(get_conclusion_text $overall_percentage)

**Validator:** Repository Health & SDLC Compliance Tool v1.0
**Report Generated:** $TIMESTAMP
EOF
    
    success "SDLC validation completed - Overall status: $validation_status ($overall_percentage%)"
    log "Detailed validation report available at: $VALIDATION_REPORT"
}

# Helper functions
get_maturity_level() {
    local percentage=$1
    if [[ $percentage -ge 95 ]]; then
        echo "Enterprise-Grade"
    elif [[ $percentage -ge 85 ]]; then
        echo "Advanced"
    elif [[ $percentage -ge 70 ]]; then
        echo "Maturing"
    elif [[ $percentage -ge 50 ]]; then
        echo "Developing"
    else
        echo "Basic"
    fi
}

get_gap_analysis() {
    local percentage=$1
    local gap=$((100 - percentage))
    if [[ $gap -le 5 ]]; then
        echo "Minimal gaps - fine-tuning needed"
    elif [[ $gap -le 15 ]]; then
        echo "Minor gaps - targeted improvements needed"
    elif [[ $gap -le 30 ]]; then
        echo "Moderate gaps - systematic improvements needed"
    else
        echo "Significant gaps - comprehensive improvements needed"
    fi
}

get_conclusion_text() {
    local percentage=$1
    if [[ $percentage -ge 95 ]]; then
        echo "The repository demonstrates exceptional SDLC maturity and is ready for enterprise production use."
    elif [[ $percentage -ge 85 ]]; then
        echo "The repository shows strong SDLC implementation with minor areas for improvement."
    elif [[ $percentage -ge 70 ]]; then
        echo "The repository has a solid foundation but requires focused improvements to reach enterprise standards."
    else
        echo "The repository needs significant SDLC enhancements to meet enterprise standards."
    fi
}

# Main execution
main() {
    log "Starting comprehensive SDLC implementation validation..."
    
    # Initialize report
    init_validation_report
    
    # Run all validations
    validate_checkpoint_1
    validate_checkpoint_2
    validate_checkpoint_3
    validate_checkpoint_4
    validate_checkpoint_5
    validate_github_workflows
    validate_automation_scripts
    
    # Generate final summary
    generate_validation_summary
    
    # Exit with appropriate code
    if [[ $FAILED_VALIDATIONS -gt 0 ]]; then
        exit 1
    elif [[ $WARNING_VALIDATIONS -gt 5 ]]; then
        exit 2
    else
        exit 0
    fi
}

# Run main function
main "$@"