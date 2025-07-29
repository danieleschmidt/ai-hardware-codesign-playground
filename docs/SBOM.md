# Software Bill of Materials (SBOM)

## Overview

This document outlines the SBOM (Software Bill of Materials) generation and management strategy for the AI Hardware Co-Design Playground project. SBOMs provide transparency into software supply chain components and are essential for security compliance and vulnerability management.

## SBOM Standards

### Supported Formats
- **SPDX** (Software Package Data Exchange) - Primary format
- **CycloneDX** - Alternative format for toolchain compatibility
- **SWID** (Software Identification) - For enterprise environments

### Generation Tools
- **syft** - Primary SBOM generation tool
- **cdxgen** - CycloneDX format generation
- **spdx-tools** - SPDX validation and conversion

## SBOM Generation

### Automated Generation
SBOMs are automatically generated during:
- CI/CD pipeline execution
- Container image builds
- Release packaging
- Security scanning workflows

### Manual Generation

#### For Python Components
```bash
# Install syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM for Python dependencies
syft packages . -o spdx-json=sbom-python.spdx.json
syft packages . -o cyclonedx-json=sbom-python.cyclonedx.json

# Include container images
syft packages docker:ai-hardware-codesign-playground:latest -o spdx-json=sbom-container.spdx.json
```

#### For Node.js Components
```bash
# Generate SBOM for frontend
cd frontend && syft packages . -o spdx-json=../sbom-frontend.spdx.json

# Alternative using cdxgen
npm install -g @cyclonedx/cdxgen
cdxgen -o sbom-frontend.cyclonedx.json frontend/
```

#### For Container Images
```bash
# Scan final container image
syft packages docker:ai-hardware-codesign-playground:latest -o spdx-json=sbom-full.spdx.json

# Include all layers and components
syft packages docker:ai-hardware-codesign-playground:latest -o table
```

## SBOM Contents

### Required Components
- **Direct dependencies** - All explicitly declared dependencies
- **Transitive dependencies** - All indirect dependencies
- **Base container images** - OS packages and system libraries
- **Build tools** - Compilers, build systems, and development tools
- **Runtime components** - Interpreters, virtual machines, and runtime libraries

### Metadata Fields
- **Package name and version**
- **License information**
- **Package supplier/originator**
- **Download location**
- **Verification codes and checksums**
- **Package relationships**
- **Vulnerability identifiers (if known)**

## Integration with Security Tools

### Vulnerability Scanning
```bash
# Scan SBOM for vulnerabilities using Grype
grype sbom:sbom-full.spdx.json

# Generate vulnerability report
grype sbom:sbom-full.spdx.json -o json > vulnerability-report.json
```

### License Compliance
```bash
# Check license compatibility
syft packages . -o json | jq '.artifacts[] | {name: .name, version: .version, licenses: .licenses}'

# Generate license report
syft packages . -o json | jq -r '.artifacts[] | "\(.name),\(.version),\(.licenses | join("; "))"' > license-report.csv
```

### Supply Chain Analysis
```bash
# Analyze supply chain risks
syft packages . -o json | jq '.artifacts[] | select(.type == "python") | {name: .name, locations: .locations}'

# Check for known malicious packages
grype sbom:sbom-python.spdx.json --only-fixed false
```

## Compliance Requirements

### SLSA (Supply-chain Levels for Software Artifacts)
- **Level 1**: Basic build integrity
- **Level 2**: Hosted build service
- **Level 3**: Hardened builds
- **Level 4**: Highest level of protection

### Executive Order on Cybersecurity (EO 14028)
- SBOM generation for all software components
- Vulnerability disclosure and management
- Supply chain risk management
- Software provenance tracking

### NIST Guidelines
- Following NIST SP 800-161 (Supply Chain Risk Management)
- Compliance with NIST Cybersecurity Framework
- Implementation of NIST Secure Software Development Framework (SSDF)

## SBOM Storage and Distribution

### Repository Integration
```bash
# Store SBOMs in releases
gh release create v1.0.0 sbom-full.spdx.json --notes "Release with SBOM"

# Tag SBOMs with git
git tag -a sbom-v1.0.0 -m "SBOM for version 1.0.0"
```

### Container Registry Integration
```bash
# Attach SBOM to container image
cosign attach sbom --sbom sbom-container.spdx.json ai-hardware-codesign-playground:latest

# Verify SBOM signature
cosign verify-attestation --type spdxjson ai-hardware-codesign-playground:latest
```

### API Integration
```bash
# Upload to dependency tracking system
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d @sbom-full.spdx.json \
  https://dependency-track.company.com/api/v1/bom
```

## Verification and Validation

### SBOM Validation
```bash
# Validate SPDX format
spdx-tools-python verify sbom-full.spdx.json

# Validate CycloneDX format
cyclonedx-cli validate --input-file sbom-full.cyclonedx.json
```

### Integrity Verification
```bash
# Sign SBOM with cosign
cosign sign-blob --key cosign.key sbom-full.spdx.json

# Verify SBOM signature
cosign verify-blob --key cosign.pub --signature sbom-full.spdx.json.sig sbom-full.spdx.json
```

### Completeness Checking
```bash
# Compare with dependency manifests
diff <(jq -r '.artifacts[].name' sbom-full.spdx.json | sort) \
     <(pip freeze | cut -d'=' -f1 | sort)

# Check for missing components
syft packages . -o json | jq '.artifacts | length'
```

## Automation and CI/CD Integration

### GitHub Actions Workflow
```yaml
name: Generate SBOM
on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          format: spdx-json
          output-file: sbom.spdx.json
      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.spdx.json
```

### Pre-commit Hook
```bash
# Add SBOM generation to pre-commit
cat >> .pre-commit-config.yaml << EOF
  - repo: local
    hooks:
      - id: generate-sbom
        name: Generate SBOM
        entry: syft packages . -o spdx-json=sbom.spdx.json
        language: system
        pass_filenames: false
EOF
```

## Monitoring and Alerting

### Vulnerability Monitoring
- Daily scans of generated SBOMs
- Integration with vulnerability databases (NVD, GitHub Advisory)
- Automated alerting for new vulnerabilities
- Prioritization based on CVSS scores and exploitability

### License Monitoring
- Track license changes in dependencies
- Alert on GPL or copyleft license introductions
- Monitor for license compatibility issues
- Generate compliance reports

### Supply Chain Monitoring
- Track new dependencies and version changes
- Monitor for dependency confusion attacks
- Alert on packages from untrusted sources
- Analyze supply chain risk scores

## Best Practices

### SBOM Generation
1. **Generate SBOMs early and often** - Include in every build
2. **Include all components** - Don't exclude development dependencies
3. **Use multiple formats** - Support different toolchain requirements
4. **Validate SBOMs** - Ensure format compliance and completeness
5. **Sign SBOMs** - Provide integrity and authenticity guarantees

### SBOM Management
1. **Version control SBOMs** - Track changes over time
2. **Store SBOMs securely** - Protect from tampering
3. **Distribute SBOMs** - Make available to consumers
4. **Update regularly** - Reflect dependency changes
5. **Archive SBOMs** - Maintain historical records

### Tool Selection
1. **Use industry-standard tools** - Prefer established solutions
2. **Validate tool output** - Don't trust blindly
3. **Keep tools updated** - Latest versions have best coverage
4. **Use multiple tools** - Cross-validate results
5. **Automate everything** - Reduce manual errors

## Troubleshooting

### Common Issues
- **Missing dependencies** - Check for dynamic loading
- **Incorrect versions** - Verify version detection logic
- **License detection failures** - Manual license specification
- **Large SBOM sizes** - Consider filtering strategies
- **Tool compatibility** - Format conversion utilities

### Resolution Strategies
- Use multiple SBOM generation tools
- Manual verification of critical components
- Custom parsing for non-standard packages
- Regular tool updates and calibration
- Documentation of known limitations

## References

- [NTIA SBOM Minimum Requirements](https://www.ntia.doc.gov/report/2021/minimum-elements-software-bill-materials-sbom)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [SLSA Framework](https://slsa.dev/)
- [NIST SP 800-161](https://csrc.nist.gov/publications/detail/sp/800-161/final)
- [Executive Order 14028](https://www.whitehouse.gov/briefing-room/presidential-actions/2021/05/12/executive-order-on-improving-the-nations-cybersecurity/)