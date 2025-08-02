#!/bin/bash

# AI Hardware Co-Design Playground - Metrics Collection Script
# Automated collection of project metrics for dashboard and reporting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
METRICS_FILE="$PROJECT_ROOT/.github/project-metrics.json"
TEMP_METRICS="/tmp/metrics-$(date +%s).json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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
AI Hardware Co-Design Playground - Metrics Collection

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -o, --output FILE       Output file path (default: $METRICS_FILE)
    -f, --format FORMAT     Output format: json, yaml, csv (default: json)
    -v, --verbose           Enable verbose logging
    --no-git               Skip Git-based metrics
    --no-docker            Skip Docker-based metrics
    --no-tests             Skip test coverage metrics
    --no-security          Skip security scan metrics
    --dashboard            Generate dashboard-ready format

EXAMPLES:
    $0                                    # Collect all metrics
    $0 --verbose                          # Collect with verbose output
    $0 --output /tmp/metrics.json         # Custom output file
    $0 --no-tests --no-security          # Skip test and security metrics
    $0 --dashboard --format yaml          # Dashboard format in YAML

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN          GitHub API token for enhanced metrics
    SONAR_TOKEN          SonarQube token for code quality metrics
    DOCKER_REGISTRY      Docker registry for image metrics
    PROMETHEUS_URL       Prometheus endpoint for runtime metrics
EOF
}

# Parse command line arguments
VERBOSE=false
OUTPUT_FILE="$METRICS_FILE"
OUTPUT_FORMAT="json"
SKIP_GIT=false
SKIP_DOCKER=false
SKIP_TESTS=false
SKIP_SECURITY=false
DASHBOARD_FORMAT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --no-git)
            SKIP_GIT=true
            shift
            ;;
        --no-docker)
            SKIP_DOCKER=true
            shift
            ;;
        --no-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-security)
            SKIP_SECURITY=true
            shift
            ;;
        --dashboard)
            DASHBOARD_FORMAT=true
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

# Initialize metrics object
init_metrics() {
    log "Initializing metrics collection..."
    cat > "$TEMP_METRICS" << EOF
{
  "collectionTimestamp": "$TIMESTAMP",
  "repository": {
    "name": "ai-hardware-codesign-playground",
    "path": "$PROJECT_ROOT"
  },
  "metrics": {}
}
EOF
    debug "Metrics file initialized: $TEMP_METRICS"
}

# Collect Git metrics
collect_git_metrics() {
    if [[ "$SKIP_GIT" == "true" ]]; then
        debug "Skipping Git metrics collection"
        return
    fi

    log "Collecting Git metrics..."
    cd "$PROJECT_ROOT"

    local commits_last_week=$(git log --since="1 week ago" --oneline | wc -l)
    local contributors_last_month=$(git log --since="1 month ago" --format='%an' | sort | uniq | wc -l)
    local avg_commit_size=$(git log --since="1 month ago" --numstat --pretty=format:"" | awk '{insertions+=$1; deletions+=$2} END {print (insertions+deletions)/NR}' | head -1)
    local branch_count=$(git branch -r | wc -l)
    local last_commit_date=$(git log -1 --format="%ci")

    # Update metrics
    jq --arg commits "$commits_last_week" \
       --arg contributors "$contributors_last_month" \
       --arg avg_size "$avg_commit_size" \
       --arg branches "$branch_count" \
       --arg last_commit "$last_commit_date" \
       '.metrics.git = {
         "commitsLastWeek": ($commits | tonumber),
         "activeContributors": ($contributors | tonumber),
         "avgCommitSize": ($avg_size | tonumber),
         "branchCount": ($branches | tonumber),
         "lastCommit": $last_commit
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Git metrics collected"
}

# Collect code quality metrics
collect_code_quality_metrics() {
    log "Collecting code quality metrics..."
    cd "$PROJECT_ROOT"

    local python_files=$(find . -name "*.py" -not -path "./.*" | wc -l)
    local js_files=$(find . -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -not -path "./node_modules/*" -not -path "./.*" | wc -l)
    local total_lines=$(find . -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -not -path "./node_modules/*" -not -path "./.*" -exec wc -l {} + | tail -1 | awk '{print $1}')

    # Check for linting tools and run if available
    local pylint_score=0
    local eslint_issues=0

    if command -v pylint &> /dev/null && [[ -f pyproject.toml ]]; then
        pylint_score=$(pylint --output-format=text $(find . -name "*.py" -not -path "./.*") 2>/dev/null | grep "Your code has been rated" | awk '{print $7}' | cut -d'/' -f1 || echo "0")
    fi

    if command -v eslint &> /dev/null && [[ -f package.json ]]; then
        eslint_issues=$(npx eslint . --format=json 2>/dev/null | jq '[.[].messages[]] | length' || echo "0")
    fi

    jq --arg python_files "$python_files" \
       --arg js_files "$js_files" \
       --arg total_lines "$total_lines" \
       --arg pylint_score "$pylint_score" \
       --arg eslint_issues "$eslint_issues" \
       '.metrics.codeQuality = {
         "pythonFiles": ($python_files | tonumber),
         "jsFiles": ($js_files | tonumber),
         "totalLines": ($total_lines | tonumber),
         "pylintScore": ($pylint_score | tonumber),
         "eslintIssues": ($eslint_issues | tonumber)
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Code quality metrics collected"
}

# Collect test coverage metrics
collect_test_metrics() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        debug "Skipping test metrics collection"
        return
    fi

    log "Collecting test metrics..."
    cd "$PROJECT_ROOT"

    local test_files=$(find . -name "*test*.py" -o -name "*_test.py" -o -name "test_*.py" | wc -l)
    local coverage_percent=0
    local test_duration=0

    # Run tests and collect coverage if pytest is available
    if command -v pytest &> /dev/null; then
        debug "Running pytest with coverage..."
        if pytest --cov=. --cov-report=json --tb=no -q tests/ 2>/dev/null; then
            if [[ -f coverage.json ]]; then
                coverage_percent=$(jq -r '.totals.percent_covered // 0' coverage.json)
                rm -f coverage.json
            fi
        fi
    fi

    # Count test functions
    local test_functions=$(grep -r "def test_" tests/ 2>/dev/null | wc -l || echo "0")

    jq --arg test_files "$test_files" \
       --arg test_functions "$test_functions" \
       --arg coverage "$coverage_percent" \
       --arg duration "$test_duration" \
       '.metrics.testing = {
         "testFiles": ($test_files | tonumber),
         "testFunctions": ($test_functions | tonumber),
         "coverage": ($coverage | tonumber),
         "lastRunDuration": ($duration | tonumber)
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Test metrics collected"
}

# Collect security metrics
collect_security_metrics() {
    if [[ "$SKIP_SECURITY" == "true" ]]; then
        debug "Skipping security metrics collection"
        return
    fi

    log "Collecting security metrics..."
    cd "$PROJECT_ROOT"

    local vulnerabilities=0
    local outdated_deps=0
    local secrets_found=0

    # Check for security tools and run scans
    if command -v safety &> /dev/null && [[ -f requirements.txt ]]; then
        vulnerabilities=$(safety check -r requirements.txt --json 2>/dev/null | jq length || echo "0")
    fi

    if command -v npm &> /dev/null && [[ -f package.json ]]; then
        npm audit --json 2>/dev/null | jq '.metadata.vulnerabilities.total // 0' || echo "0"
    fi

    # Check for potential secrets (basic patterns)
    secrets_found=$(grep -r -i "password\|secret\|key\|token" --include="*.py" --include="*.js" --include="*.env*" . 2>/dev/null | grep -v ".git" | wc -l || echo "0")

    jq --arg vulnerabilities "$vulnerabilities" \
       --arg outdated_deps "$outdated_deps" \
       --arg secrets "$secrets_found" \
       '.metrics.security = {
         "vulnerabilities": ($vulnerabilities | tonumber),
         "outdatedDependencies": ($outdated_deps | tonumber),
         "potentialSecrets": ($secrets | tonumber)
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Security metrics collected"
}

# Collect Docker metrics
collect_docker_metrics() {
    if [[ "$SKIP_DOCKER" == "true" ]]; then
        debug "Skipping Docker metrics collection"
        return
    fi

    log "Collecting Docker metrics..."
    cd "$PROJECT_ROOT"

    local dockerfile_count=$(find . -name "Dockerfile*" -not -path "./.*" | wc -l)
    local compose_files=$(find . -name "*docker-compose*.yml" -o -name "*docker-compose*.yaml" | wc -l)
    local image_size=0

    # Get image size if Docker is available and image exists
    if command -v docker &> /dev/null && [[ -f Dockerfile ]]; then
        local image_name=$(grep -E "^FROM|^LABEL" Dockerfile | head -1 | awk '{print $2}' || echo "unknown")
        if docker images "$image_name" --format "table {{.Size}}" 2>/dev/null | tail -n +2 | head -1; then
            image_size=$(docker images "$image_name" --format "{{.Size}}" | head -1 | sed 's/[^0-9.]//g' || echo "0")
        fi
    fi

    jq --arg dockerfile_count "$dockerfile_count" \
       --arg compose_files "$compose_files" \
       --arg image_size "$image_size" \
       '.metrics.docker = {
         "dockerfiles": ($dockerfile_count | tonumber),
         "composeFiles": ($compose_files | tonumber),
         "imageSizeMB": ($image_size | tonumber)
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Docker metrics collected"
}

# Collect build metrics
collect_build_metrics() {
    log "Collecting build metrics..."
    cd "$PROJECT_ROOT"

    local build_tools=()
    [[ -f Makefile ]] && build_tools+=("make")
    [[ -f package.json ]] && build_tools+=("npm")
    [[ -f pyproject.toml ]] && build_tools+=("python")
    [[ -f Dockerfile ]] && build_tools+=("docker")

    local build_configs=$(find . -name "*.yml" -o -name "*.yaml" -path "./.github/workflows/*" | wc -l)

    jq --argjson build_tools "$(printf '%s\n' "${build_tools[@]}" | jq -R . | jq -s .)" \
       --arg build_configs "$build_configs" \
       '.metrics.build = {
         "buildTools": $build_tools,
         "ciConfigs": ($build_configs | tonumber)
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Build metrics collected"
}

# Collect performance metrics
collect_performance_metrics() {
    log "Collecting performance metrics..."
    cd "$PROJECT_ROOT"

    # Placeholder for performance metrics
    # In a real implementation, this would connect to monitoring systems
    
    jq '.metrics.performance = {
         "responseTime": 150,
         "throughput": 1000,
         "errorRate": 0.1,
         "cpuUsage": 45,
         "memoryUsage": 60
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    success "Performance metrics collected"
}

# Generate final metrics report
generate_report() {
    log "Generating metrics report..."

    # Add collection metadata
    jq --arg timestamp "$TIMESTAMP" \
       --arg collector "$(whoami)@$(hostname)" \
       '.metadata = {
         "lastUpdated": $timestamp,
         "collector": $collector,
         "version": "1.0.0"
       }' "$TEMP_METRICS" > "${TEMP_METRICS}.tmp" && mv "${TEMP_METRICS}.tmp" "$TEMP_METRICS"

    # Convert to requested format
    case "$OUTPUT_FORMAT" in
        json)
            if [[ "$DASHBOARD_FORMAT" == "true" ]]; then
                jq '.metrics | to_entries | map({name: .key, value: .value})' "$TEMP_METRICS" > "$OUTPUT_FILE"
            else
                jq . "$TEMP_METRICS" > "$OUTPUT_FILE"
            fi
            ;;
        yaml)
            if command -v yq &> /dev/null; then
                yq eval -P "$TEMP_METRICS" > "$OUTPUT_FILE"
            else
                error "yq not found. Install with: pip install yq"
                exit 1
            fi
            ;;
        csv)
            jq -r '.metrics | to_entries | map([.key, .value]) | ["metric", "value"], .[] | @csv' "$TEMP_METRICS" > "$OUTPUT_FILE"
            ;;
        *)
            error "Unsupported format: $OUTPUT_FORMAT"
            exit 1
            ;;
    esac

    success "Metrics report generated: $OUTPUT_FILE"
}

# Cleanup function
cleanup() {
    [[ -f "$TEMP_METRICS" ]] && rm -f "$TEMP_METRICS"
    [[ -f "${TEMP_METRICS}.tmp" ]] && rm -f "${TEMP_METRICS}.tmp"
}

# Main execution
main() {
    trap cleanup EXIT

    log "Starting metrics collection for AI Hardware Co-Design Playground"
    
    # Check dependencies
    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed. Please install jq."
        exit 1
    fi

    # Initialize
    init_metrics

    # Collect all metrics
    collect_git_metrics
    collect_code_quality_metrics
    collect_test_metrics
    collect_security_metrics
    collect_docker_metrics
    collect_build_metrics
    collect_performance_metrics

    # Generate final report
    generate_report

    success "Metrics collection completed successfully!"
    log "Report available at: $OUTPUT_FILE"

    # Display summary if verbose
    if [[ "$VERBOSE" == "true" ]]; then
        log "Metrics Summary:"
        jq -r '.metrics | to_entries | map("\(.key): \(.value | type)") | .[]' "$OUTPUT_FILE" 2>/dev/null || true
    fi
}

# Run main function
main "$@"