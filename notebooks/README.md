# Jupyter Notebooks

This directory contains Jupyter notebooks for interactive development and experimentation with the AI Hardware Co-Design Playground.

## Getting Started

1. Start Jupyter Lab:
   ```bash
   npm run jupyter
   # or
   make jupyter
   ```

2. Navigate to http://localhost:8888 in your browser

3. Use token: `codesign-playground` (or check the console output)

## Notebook Categories

### Examples (`examples/`)
- Basic usage examples
- Tutorial notebooks
- Quick start guides

### Experiments (`experiments/`)
- Research experiments
- Performance analysis
- Design space exploration

### Analysis (`analysis/`)
- Model analysis notebooks
- Hardware profiling
- Results visualization

## Best Practices

1. **Name notebooks descriptively**: Use clear, descriptive names that indicate the purpose
2. **Add documentation**: Include markdown cells explaining the purpose and methodology
3. **Keep notebooks focused**: One notebook per experiment or analysis
4. **Version control**: Commit notebooks with cleared outputs using nbstripout
5. **Use relative paths**: Keep paths relative to the notebook location

## Environment

Notebooks run in the same environment as the main application with access to:
- All Python packages from requirements.txt
- Database and Redis connections
- Hardware simulation tools (if installed)
- ML frameworks (PyTorch, TensorFlow, etc.)

## Data Access

Notebooks can access:
- Sample datasets in `../data/`
- Generated models in `../generated/`
- Uploaded files in `../uploads/`
- Cached results in `../cache/`