# Development Guide

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd ai-hardware-codesign-playground
npm run setup

# Start development
npm run dev
```

## Prerequisites

* Node.js 18+
* Python 3.9+
* Git
* Docker (optional)

## Development Commands

```bash
npm run dev          # Start dev servers
npm run test         # Run all tests
npm run lint         # Run linting
npm run typecheck    # Type checking
npm run format       # Format code
```

## Architecture

See [ARCHITECTURE.md](../ARCHITECTURE.md) for system design details.

## Testing

Run tests with `npm run test`. See individual package directories for specific testing instructions.

## Documentation

* API docs: Generated from code annotations
* User guides: See `/docs` directory
* Contributing: See [CONTRIBUTING.md](../CONTRIBUTING.md)

For detailed setup instructions, see the main [README.md](../README.md).