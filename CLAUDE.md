# Graphtage Developer Guidelines

## Project Overview

Graphtage is a semantic diff/merge utility for tree-like structured data formats (JSON, JSON5, XML, HTML, YAML, TOML, INI, CSV, plist, Python pickle). It works as both a command-line tool and Python library.

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
- json.py, yaml.py, xml.py, csv.py, toml.py, ini.py, plist.py, pickle.py

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
Ruff is configured in `pyproject.toml` and gates CI, so the tree is expected to be clean.

```bash
ruff check graphtage test docs bindist
ruff check --fix graphtage test docs bindist
```

### Building Documentation
```bash
cd docs && make html
# Output in docs/_build/html/
```

## Code Patterns

### Adding a New File Format
1. Create `graphtage/newformat.py`.
2. Define a `Filetype` subclass with a zero-argument `__init__`. `FiletypeWatcher` instantiates it at
   class-definition time, so it must implement `build_tree`, `build_tree_handling_errors`, and
   `get_default_formatter`. Registration into `FILETYPES_BY_TYPENAME` and `FILETYPES_BY_MIME` is automatic, and
   that is what generates the `--from-*`, `--to-*`, and `--format` CLI flags.
3. Implement `build_tree(path: str, options: Optional[BuildOptions] = None) -> TreeNode`. Reusing
   `graphtage.json.build_tree` on a plain Python object gets you the whole `TreeNode` contract for free.
4. Add the module to the `from . import ...` line in `graphtage/__init__.py`. Nothing registers without it, and
   `docs/build_api.py` discovers API pages from this import.
5. Add the extension to `register_mimetypes()` in `graphtage/__main__.py`. `mimetypes` does not know most of these,
   and format detection is extension-based, so without this every diff fails with "Could not determine the filetype".
6. Add a `test_<typename>_formatting` method to `test/test_formatting.py`, or `test_formatter_coverage` fails. Note
   that `@filetype_test` only round-trips *unedited* trees, so it cannot catch a formatter that mishandles edits —
   add a separate diff-level test for insertions and removals.
7. Give the formatter `print_UnorderedListNode = print_ListNode`, or `test_unordered_list_renders_like_a_list`
   fails. A formatter that cannot resolve a node type bounces to `self.parent.print(...)` and recurses forever.
8. Mark helper formatters `is_partial = True` so they stay out of the global `FORMATTERS` list, where they could
   change how unrelated formats resolve node types.
9. Update the format lists in `README.md`, `docs/index.rst`, `CITATION.cff`, `pyproject.toml`, the `--help`
   description in `graphtage/__main__.py`, and this file.
10. Add a dependency to `pyproject.toml` only if the parser is third-party, and regenerate `uv.lock`.

Route printing through `SequenceFormatter.print_SequenceNode`, which is where insert and remove edits are applied;
iterating a node's children directly silently drops them. Only one formatter may define `print_<NodeType>` for a
given type, so distinguish nesting levels in the key/value formatter rather than by node type (see
`graphtage/yaml.py` and `graphtage/ini.py`).

Python 3.10 is the minimum supported version. Check `requires-python` in `pyproject.toml` and the CI matrix in
`.github/workflows/pythonpackage.yml` before using a feature from a newer release; a runtime-evaluated annotation
that the floor does not support fails at import, which takes down the whole package.

Never add `from __future__ import annotations` to this package. Two places read annotations at runtime, and PEP 563
turns both into silent no-ops rather than errors: `DataClassNode.__init_subclass__` derives `_SLOTS` from
`cls.__annotations__`, and `FormatterChecker` validates the `printer` parameter of every `print_*` method with
`inspect.signature`. For the same reason, a `DataClassNode` slot annotation must name a `TreeNode` subclass
directly; a subscripted generic is skipped, not turned into a slot.

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
- Python version: 3.10+ compatibility required; CI covers 3.10 through 3.14
- Type hints: `typing.Protocol`; `typing_extensions` is not a dependency of the library
- Annotations: PEP 585 and PEP 604 spellings (`list[str]`, `X | None`), not `typing.List` or `Optional`
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
