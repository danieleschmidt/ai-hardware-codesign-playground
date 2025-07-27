# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

The AI Hardware Co-Design Playground team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them by email to: **security@codesign-playground.com**

If you don't receive a response within 48 hours, please follow up via:
- Discord: [AI Hardware Co-Design Community](https://discord.gg/ai-hardware-codesign)
- GitHub Security Advisory: Use the "Report a vulnerability" option in the Security tab

### What to Include

Please include the following information in your report:
- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Development**: Within 2-4 weeks (depending on complexity)
- **Public Disclosure**: After fix is released and deployed

### Security Update Process

1. **Vulnerability Assessment**: We confirm and assess the vulnerability
2. **Fix Development**: We develop and test a fix
3. **Release Preparation**: We prepare a security release
4. **Coordinated Disclosure**: We notify affected users and release the fix
5. **Public Disclosure**: We publish security advisory after users have time to update

## Security Measures

### Code Security
- All code changes require review by at least one maintainer
- Automated security scanning with CodeQL, Bandit, and Safety
- Regular dependency updates and vulnerability scanning
- Pre-commit hooks for security checks

### Infrastructure Security
- Container security scanning with Trivy
- Secrets scanning to prevent credential leaks
- Signed commits and releases
- Supply chain security with SBOM generation

### Data Security
- No user data is stored without explicit consent
- Model files and designs are kept confidential when marked as private
- Secure handling of API keys and credentials
- Encrypted communication for all external services

### Runtime Security
- Sandboxed execution of generated RTL code
- Input validation for all user-provided data
- Resource limits to prevent DoS attacks
- Secure temporary file handling

## Security Best Practices for Users

### For Developers
- Keep your development environment up to date
- Use virtual environments for Python dependencies
- Never commit secrets or credentials to version control
- Use the provided pre-commit hooks for security scanning
- Enable two-factor authentication on your GitHub account

### For Model and Hardware Designs
- Validate all inputs before processing
- Use quantization and other techniques to reduce attack surface
- Be cautious when sharing proprietary models or designs
- Review generated RTL code before deployment

### For Production Deployments
- Use the latest stable release
- Keep all dependencies updated
- Enable security monitoring and logging
- Use HTTPS for all web communications
- Implement proper access controls and authentication

## Vulnerability Disclosure Policy

We follow the principle of **Coordinated Vulnerability Disclosure**:

1. **Private Disclosure**: Security researchers privately notify us of vulnerabilities
2. **Assessment and Fix**: We assess the issue and develop a fix
3. **Coordinated Release**: We work with the researcher to time the public disclosure
4. **Public Disclosure**: After users have time to update, we publish details

### Recognition

We believe security research is important and valuable work. Researchers who responsibly disclose vulnerabilities will be:
- Credited in our security advisories (if desired)
- Listed in our Hall of Fame
- Eligible for our bug bounty program (when available)

## Security Hall of Fame

We thank the following researchers for their responsible disclosure:
- *Be the first to help secure AI Hardware Co-Design Playground!*

## Security Advisories

All security advisories are published at:
- [GitHub Security Advisories](https://github.com/terragon-labs/ai-hardware-codesign-playground/security/advisories)
- [Project Security Page](https://docs.codesign-playground.com/security)

## Security Tools and Dependencies

We use the following tools to maintain security:

### Static Analysis
- **Bandit**: Python security linter
- **CodeQL**: Semantic code analysis
- **Semgrep**: Static analysis for security bugs

### Dependency Scanning
- **Safety**: Python dependency vulnerability scanner
- **pip-audit**: Audit Python packages for known vulnerabilities
- **Dependabot**: Automated dependency updates

### Container Security
- **Trivy**: Container vulnerability scanner
- **Docker Scout**: Container image security analysis

### Secrets Detection
- **detect-secrets**: Prevent secrets in code
- **GitLeaks**: Git secrets scanner
- **TruffleHog**: Secrets scanner

## Contact

For general security questions or concerns, contact:
- **Email**: security@codesign-playground.com
- **Discord**: [AI Hardware Co-Design Community](https://discord.gg/ai-hardware-codesign)
- **GitHub**: [@terragon-labs](https://github.com/terragon-labs)

---

*Last updated: 2025-01-27*