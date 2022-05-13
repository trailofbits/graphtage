# Graphtage

[![PyPI version](https://badge.fury.io/py/graphtage.svg)](https://badge.fury.io/py/graphtage)
[![Tests](https://github.com/trailofbits/graphtage/workflows/Python%20package/badge.svg)](https://github.com/trailofbits/graphtage/actions)
[![Slack Status](https://empireslacking.herokuapp.com/badge.svg)](https://empireslacking.herokuapp.com)

Graphtage is a command-line utility and [underlying library](https://trailofbits.github.io/graphtage/latest/library.html)
for semantically comparing and merging tree-like structures, such as JSON, XML, HTML, YAML, plist, and CSS files. Its name is a
portmanteau of “graph” and “graftage”—the latter being the horticultural practice of joining two trees together such
that they grow as one.

```console
$ echo Original: && cat original.json && echo Modified: && cat modified.json
```
```json
Original:
{
    "foo": [1, 2, 3, 4],
    "bar": "testing"
}
Modified:
{
    "foo": [2, 3, 4, 5],
    "zab": "testing",
    "woo": ["foobar"]
}
```
```console
$ graphtage original.json modified.json
```
```json
{
    "z̟b̶ab̟r̶": "testing",
    "foo": [
        1̶,̶
        2,
        3,
        4,̟
        5̟
    ],̟
    "̟w̟o̟o̟"̟:̟ ̟[̟
        "̟f̟o̟o̟b̟a̟r̟"̟
    ]̟
}
```

## Installation

```console
$ pip3 install graphtage
```

## Command Line Usage

### Output Formatting
Graphtage performs an analysis on an intermediate representation of the trees that is divorced from the filetypes of the
input files. This means, for example, that you can diff a JSON file against a YAML file. Also, the output format can be
different from the input format(s). By default, Graphtage will format the output diff in the same file format as the
first input file. But one could, for example, diff two JSON files and format the output in YAML. There are several
command-line arguments to specify these transformations, such as `--format`; please check the `--help` output for more
information.

By default, Graphtage pretty-prints its output with as many line breaks and indents as possible.
```json
{
    "foo": [
        1,
        2,
        3
    ],
    "bar": "baz"
}
```
Use the `--join-lists` or `-jl` option to suppress linebreaks after list items:
```json
{
    "foo": [1, 2, 3],
    "bar": "baz"
}
```
Likewise, use the `--join-dict-items` or `-jd` option to suppress linebreaks after key/value pairs in a dict:
```json
{"foo": [
    1,
    2,
    3
], "bar":  "baz"}
```
Use `--condensed` or `-j` to apply both of these options:
```json
{"foo": [1, 2, 3], "bar": "baz"}
```

The `--only-edits` or `-e` option will print out a list of edits rather than applying them to the input file in place.

### Matching Options
By default, Graphtage tries to match all possible pairs of elements in a dictionary.

Matching two dictionaries with each other is hard. Although computationally tractable, this can sometimes be onerous for 
input files with huge dictionaries. Graphtage has three different strategies for matching dictionaries:
1. `--dict-strategy match` (the most computationally expensive) tries to match all pairs of keys and values between the
   two dictionaries, resulting in a match of minimum edit distance;
2. `--dict-strategy none` (the least computationally expensive) will not attempt to match any key/value pairs unless
   they have the exact same key; and
3. `--dict-strategy auto` (the default) will automatically match the values of any key-value pairs that have identical
   keys and then use the `match` strategy for the remainder of key/value pairs.

See [Pull Request #51](https://github.com/trailofbits/graphtage/pull/51) for some examples of how these strategies
affect output.

The `--no-list-edits` or `-l` option will not consider interstitial insertions and removals when comparing two lists.
The `--no-list-edits-when-same-length` or `-ll` option is a less drastic version of `-l` that will behave normally for
lists that are of different lengths but behave like `-l` for lists that are of the same length.

### ANSI Color
By default, Graphtage will only use ANSI color in its output if it is run from a TTY. If, for example, you would like
to have Graphtage emit colorized output from a script or pipe, use the `--color` or `-c` argument. To disable color even
when running on a TTY, use `--no-color`.

### HTML Output
Graphtage can optionally emit the diff in HTML with the `--html` option.
```console
$ graphtage --html original.json modified.json > diff.html
```

### Status and Logging
By default, Graphtage prints status messages and a progress bar to STDERR. To suppress this, use the `--no-status`
option. To additionally suppress all but critical log messages, use `--quiet`. Fine-grained control of log messages is
via the `--log-level` option.

## Why does Graphtage exist?

Diffing tree-like structures with unordered elements is tough. Say you want to compare two JSON files.
There are [limited tools available](https://github.com/zgrossbart/jdd), which are effectively equivalent to
canonicalizing the JSON (_e.g._, sorting dictionary elements by key) and performing a standard diff. This is not always
sufficient. For example, if a key in a dictionary is changed but its value is not, a traditional diff
will conclude that the entire key/value pair was replaced by the new one, even though the only change was the key
itself. See [our documentation](https://trailofbits.github.io/graphtage/latest/howitworks.html) for more information.

## Using Graphtage as a Library

See [our documentation](https://trailofbits.github.io/graphtage/latest/library.html) for more information.

## Extending Graphtage

Graphtage is designed to be extensible: New filetypes can easily be defined, as well as new node types, edit types,
formatters, and printers. See [our documentation](https://trailofbits.github.io/graphtage/latest/extending.html) for
more information.

Complete API documentation is available [here](https://trailofbits.github.io/graphtage/latest/package.html).

## License and Acknowledgements

This research was developed by [Trail of Bits](https://www.trailofbits.com/) with partial funding from the Defense
Advanced Research Projects Agency (DARPA) under the SafeDocs program as a subcontractor to [Galois](https://galois.com).
It is licensed under the [GNU Lesser General Public License v3.0](LICENSE).
[Contact us](mailto:opensource@trailofbits.com) if you're looking for an exception to the terms.
© 2020, Trail of Bits.
