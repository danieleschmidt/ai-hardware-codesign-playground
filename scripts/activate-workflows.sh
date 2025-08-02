#!/bin/bash

# AI Hardware Co-Design Playground - Workflow Activation Script
# One-command activation of production-ready SDLC workflows

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKFLOWS_SOURCE="$PROJECT_ROOT/docs/workflows/production-ready"
WORKFLOWS_TARGET="$PROJECT_ROOT/.github/workflows"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${CYAN}[DETAIL]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
🚀 AI Hardware Co-Design Playground - Workflow Activation

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -f, --force             Force overwrite existing workflows
    -d, --dry-run           Show what would be done without making changes
    -v, --verbose           Enable verbose output
    --check-permissions     Check if activation is possible
    --list-workflows        List available workflows

EXAMPLES:
    $0                              # Activate all workflows
    $0 --dry-run                    # Preview activation without changes
    $0 --force                      # Overwrite existing workflows
    $0 --check-permissions          # Check if activation is possible

WORKFLOW ACTIVATION PROCESS:
1. Checks for existing .github/workflows directory
2. Copies production-ready workflows from docs/workflows/production-ready/
3. Validates workflow syntax (if yamllint available)
4. Provides next steps for repository configuration

For detailed setup instructions, see: docs/WORKFLOW_ACTIVATION_GUIDE.md
EOF
}

# Parse command line arguments
FORCE=false
DRY_RUN=false
VERBOSE=false
CHECK_PERMISSIONS=false
LIST_WORKFLOWS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --check-permissions)
            CHECK_PERMISSIONS=true
            shift
            ;;
        --list-workflows)
            LIST_WORKFLOWS=true
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
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

# Check permissions
check_permissions() {
    log "Checking workflow activation permissions..."
    
    local can_activate=true
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        error "Not in a Git repository"
        can_activate=false
    else
        success "Git repository detected"
    fi
    
    # Check if source workflows exist
    if [[ ! -d "$WORKFLOWS_SOURCE" ]]; then
        error "Source workflows directory not found: $WORKFLOWS_SOURCE"
        can_activate=false
    else
        local workflow_count=$(find "$WORKFLOWS_SOURCE" -name "*.yml" -o -name "*.yaml" | wc -l)
        success "Source workflows directory found with $workflow_count workflows"
    fi
    
    # Check write permissions
    if [[ ! -w "$PROJECT_ROOT" ]]; then
        error "No write permission to project root: $PROJECT_ROOT"
        can_activate=false
    else
        success "Write permissions confirmed"
    fi
    
    # Check for existing workflows
    if [[ -d "$WORKFLOWS_TARGET" ]]; then
        local existing_count=$(find "$WORKFLOWS_TARGET" -name "*.yml" -o -name "*.yaml" | wc -l)
        if [[ $existing_count -gt 0 ]] && [[ "$FORCE" != "true" ]]; then
            warning "Existing workflows found ($existing_count files). Use --force to overwrite"
            info "Existing workflows:"
            find "$WORKFLOWS_TARGET" -name "*.yml" -o -name "*.yaml" | while read -r workflow; do
                info "  - $(basename "$workflow")"
            done
        else
            success "Target directory ready for workflows"
        fi
    else
        success "Target directory will be created: $WORKFLOWS_TARGET"
    fi
    
    if [[ "$can_activate" == "true" ]]; then
        success "✅ Workflow activation is possible"
        return 0
    else
        error "❌ Workflow activation is not possible - resolve issues above"
        return 1
    fi
}

# List available workflows
list_workflows() {
    log "Available production-ready workflows:"
    echo
    
    if [[ ! -d "$WORKFLOWS_SOURCE" ]]; then
        error "Source workflows directory not found: $WORKFLOWS_SOURCE"
        return 1
    fi
    
    find "$WORKFLOWS_SOURCE" -name "*.yml" -o -name "*.yaml" | while read -r workflow; do
        local name=$(basename "$workflow" .yml)
        name=$(basename "$name" .yaml)
        
        # Extract description from workflow file
        local description=""
        if grep -q "name:" "$workflow"; then
            description=$(grep "name:" "$workflow" | head -1 | sed 's/name: *//g' | tr -d '"')
        fi
        
        echo -e "${GREEN}📄 $name${NC}"
        if [[ -n "$description" ]]; then
            echo -e "   ${CYAN}Description:${NC} $description"
        fi
        
        # Show key features
        if [[ "$VERBOSE" == "true" ]]; then
            echo -e "   ${CYAN}File:${NC} $workflow"
            local triggers=$(grep -A 5 "^on:" "$workflow" | grep -E "  [a-z_]+:" | sed 's/://g' | sed 's/^  /    - /g' | tr '\n' ' ')
            if [[ -n "$triggers" ]]; then
                echo -e "   ${CYAN}Triggers:${NC}$triggers"
            fi
        fi
        echo
    done
}

# Validate workflow syntax
validate_workflow_syntax() {
    local workflow="$1"
    local filename=$(basename "$workflow")
    
    debug "Validating syntax for $filename"
    
    if command -v yamllint > /dev/null 2>&1; then
        if yamllint "$workflow" > /dev/null 2>&1; then
            success "✅ $filename - syntax valid"
            return 0
        else
            error "❌ $filename - syntax errors detected"
            if [[ "$VERBOSE" == "true" ]]; then
                yamllint "$workflow" 2>&1 | head -10
            fi
            return 1
        fi
    else
        warning "⚠️ $filename - yamllint not available, skipping validation"
        return 0
    fi
}

# Activate workflows
activate_workflows() {
    log "Activating production-ready SDLC workflows..."
    
    local activated_count=0
    local failed_count=0
    
    # Create target directory
    if [[ "$DRY_RUN" != "true" ]]; then
        mkdir -p "$WORKFLOWS_TARGET"
        debug "Created workflows directory: $WORKFLOWS_TARGET"
    else
        info "[DRY RUN] Would create directory: $WORKFLOWS_TARGET"
    fi
    
    # Process each workflow
    find "$WORKFLOWS_SOURCE" -name "*.yml" -o -name "*.yaml" | while read -r source_workflow; do
        local filename=$(basename "$source_workflow")
        local target_workflow="$WORKFLOWS_TARGET/$filename"
        
        debug "Processing workflow: $filename"
        
        # Check if target exists
        if [[ -f "$target_workflow" ]] && [[ "$FORCE" != "true" ]]; then
            warning "Skipping $filename - already exists (use --force to overwrite)"
            continue
        fi
        
        # Validate syntax before copying
        if validate_workflow_syntax "$source_workflow"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                cp "$source_workflow" "$target_workflow"
                success "✅ Activated: $filename"
                ((activated_count++))
            else
                info "[DRY RUN] Would activate: $filename"
                ((activated_count++))
            fi
        else
            error "❌ Failed to activate: $filename (syntax errors)"
            ((failed_count++))
        fi
    done
    
    echo
    if [[ "$DRY_RUN" != "true" ]]; then
        success "Workflow activation completed!"
        success "✅ Activated: $activated_count workflows"
        if [[ $failed_count -gt 0 ]]; then
            error "❌ Failed: $failed_count workflows"
        fi
    else
        info "[DRY RUN] Would activate: $activated_count workflows"
        if [[ $failed_count -gt 0 ]]; then
            info "[DRY RUN] Would fail: $failed_count workflows"
        fi
    fi
}

# Show next steps
show_next_steps() {
    echo
    log "🎯 Next Steps for Complete SDLC Activation:"
    echo
    
    echo -e "${CYAN}1. Configure Repository Secrets${NC}"
    echo "   Go to Settings > Secrets and variables > Actions"
    echo "   Required: GITHUB_TOKEN (auto-provided)"
    echo "   Optional: CODECOV_TOKEN, SLACK_WEBHOOK_URL, NPM_TOKEN, PYPI_TOKEN"
    echo
    
    echo -e "${CYAN}2. Set Branch Protection Rules${NC}"
    echo "   Go to Settings > Branches > Add rule for 'main'"
    echo "   Enable: 'Require status checks', 'Require pull request reviews'"
    echo "   Select: 'Continuous Integration' check"
    echo
    
    echo -e "${CYAN}3. Test Workflow Activation${NC}"
    echo "   Create test branch: git checkout -b test/workflow-activation"
    echo "   Make small change and push"
    echo "   Create PR and verify workflows run"
    echo
    
    echo -e "${CYAN}4. Review Detailed Setup Guide${NC}"
    echo "   See: docs/WORKFLOW_ACTIVATION_GUIDE.md"
    echo "   Contains: Advanced configuration, troubleshooting, monitoring"
    echo
    
    echo -e "${CYAN}5. Validate SDLC Implementation${NC}"
    echo "   Run: ./scripts/validate-sdlc-implementation.sh"
    echo "   Review: Generated validation report"
    echo
    
    success "🚀 Your repository is now enterprise-grade SDLC ready!"
}

# Generate activation report
generate_activation_report() {
    local report_file="$PROJECT_ROOT/workflow-activation-report.md"
    
    cat > "$report_file" << EOF
# Workflow Activation Report

**Generated:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")  
**Repository:** ai-hardware-codesign-playground  
**Activated By:** $(whoami)@$(hostname)

## Activation Summary

- **Source Directory:** $WORKFLOWS_SOURCE
- **Target Directory:** $WORKFLOWS_TARGET
- **Activation Mode:** $(if [[ "$DRY_RUN" == "true" ]]; then echo "DRY RUN"; else echo "LIVE"; fi)
- **Force Overwrite:** $FORCE

## Activated Workflows

EOF

    find "$WORKFLOWS_TARGET" -name "*.yml" -o -name "*.yaml" 2>/dev/null | while read -r workflow; do
        local name=$(basename "$workflow" .yml)
        name=$(basename "$name" .yaml)
        
        echo "### $name" >> "$report_file"
        echo "- **File:** $(basename "$workflow")" >> "$report_file"
        echo "- **Size:** $(wc -l < "$workflow") lines" >> "$report_file"
        echo "- **Status:** ✅ Activated" >> "$report_file"
        echo "" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

## Next Steps

1. **Configure Repository Secrets** (see docs/WORKFLOW_ACTIVATION_GUIDE.md)
2. **Set Branch Protection Rules** (require CI checks)
3. **Test Workflow Execution** (create test PR)
4. **Monitor Workflow Performance** (GitHub Actions tab)

## Support

For issues or questions:
- Review: docs/WORKFLOW_ACTIVATION_GUIDE.md
- Create issue: GitHub repository issues
- Validate: ./scripts/validate-sdlc-implementation.sh

---
Generated by Workflow Activation Script v1.0
EOF

    if [[ "$DRY_RUN" != "true" ]]; then
        success "📊 Activation report generated: $report_file"
    else
        info "[DRY RUN] Would generate report: $report_file"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}🚀 AI Hardware Co-Design Playground - Workflow Activation${NC}"
    echo
    
    # Handle special commands
    if [[ "$CHECK_PERMISSIONS" == "true" ]]; then
        check_permissions
        exit $?
    fi
    
    if [[ "$LIST_WORKFLOWS" == "true" ]]; then
        list_workflows
        exit 0
    fi
    
    # Check prerequisites
    if ! check_permissions; then
        exit 1
    fi
    
    # Activate workflows
    activate_workflows
    
    # Generate report
    if [[ "$DRY_RUN" != "true" ]]; then
        generate_activation_report
    fi
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"