# Graphtage Developer Guidelines

## Project Overview

Graphtage is a semantic diff/merge utility for tree-like structured data formats (JSON, XML, HTML, YAML, plist, CSS, CSV). It works as both a command-line tool and Python library.

Key capabilities:
- Semantic understanding of tree structures (recognizes key vs value changes)
- Cross-format diffing (e.g., JSON vs YAML with output in any format)
- Extensible architecture for custom node types and file formats
- HTML output support for visual diffs

## Architecture

### Core Protocols (graphtage/tree.py)
- `TreeNode`: Protocol for tree node implementations - all nodes must implement this
- `Edit`: Protocol for edit operations with cost bounds
- `GraphtageFormatter`: Base formatter for printing nodes and edits

### Concrete Implementations (graphtage/graphtage.py)
- `LeafNode`: Terminal nodes (strings, numbers, booleans, null)
- `ListNode`: Ordered sequences
- `DictNode`: Key-value mappings
- `KeyValuePairNode`: Individual dict entries

### Edit Operations (graphtage/edits.py)
- `Match`: No change needed
- `Replace`: Substitute one value for another
- `Insert`: Add new element
- `Remove`: Delete element
- `CompoundEdit`: Multiple edits grouped together

### Algorithms
- **matching.py**: Bipartite matching for optimal node correspondences
- **levenshtein.py**: String edit distance with Unicode combining marks
- **search.py**: Iterative tightening search for edit cost optimization
- **bounds.py**: Cost range calculations (Range class)
- **fibonacci.py**: Fibonacci search for optimization

### File Format Modules
Each format implements its own TreeNode subclasses and parser:
- json.py, yaml.py, xml.py, csv.py, plist.py, pickle.py

## Development Setup

### Installation
```bash
# Install with dev dependencies
pip install -e .[dev]

# Or just the package
pip install graphtage
```

### Running Tests
```bash
pytest                      # All tests
pytest test/test_graphtage.py  # Specific module
pytest -q                   # Quiet output
```

### Linting
```bash
# Ruff is configured in pyproject.toml
ruff check graphtage test
ruff check --fix graphtage test

# CI currently uses flake8
flake8 graphtage test --select=E9,F63,F7,F82
```

### Building Documentation
```bash
cd docs && make html
# Output in docs/_build/html/
```

## Code Patterns

### Adding a New File Format
1. Create `graphtage/newformat.py`
2. Define TreeNode subclasses for format-specific structures
3. Implement a `build_tree(content: str) -> TreeNode` function
4. Register the filetype in `graphtage/__init__.py`
5. Add tests in `test/test_newformat.py`

### Working with Edits
- Edit costs are computed lazily via `bounds()` method
- Use `has_non_zero_cost()` to check if an edit represents a change
- `initial_bounds` stores the first computed bounds for optimization

### Printing Protocol
The printing system is extensible:
1. Check for specialized formatter for the edit type
2. Fall back to edit's `print()` method
3. Fall back to node's `print()` method

## Key Conventions

- Line length: 120 characters (configured in ruff)
- Python version: 3.8+ compatibility required
- Type hints: Use typing_extensions for Protocol support
- Docstrings: Google style for public APIs
- Tests: Mirror package structure in test/ directory

## CLI Usage

```bash
# Basic diff
graphtage original.json modified.json

# Cross-format diff
graphtage file.json file.yaml --format yaml

# Condensed output
graphtage -j original.json modified.json

# Show only edits
graphtage -e original.json modified.json

# HTML output
graphtage --html original.json modified.json > diff.html
```

## Testing Guidelines

- Test files are in `test/` directory
- Use `test_*.py` naming convention
- Tests are organized by module (test_matching.py tests matching.py)
- Performance tests in timing.py (not run by default)
