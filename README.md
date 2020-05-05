# Graphtage

[![PyPI version](https://badge.fury.io/py/graphtage.svg)](https://badge.fury.io/py/graphtage)
[![Tests](https://github.com/trailofbits/graphtage/workflows/Python%20package/badge.svg)](https://github.com/trailofbits/graphtage/actions)
[![Slack Status](https://empireslacking.herokuapp.com/badge.svg)](https://empireslacking.herokuapp.com)

Graphtage is a commandline utility and underlying library for semantically comparing and merging tree-like structures,
such as JSON, XML, HTML, and YAML files. Its name is a portmanteau of “graph” and “graftage”—the latter being the
practice of joining two trees together such that they grow as one. 

## Why does Graphtage exist?

Diffing tree-like structures with unordered elements is tough. Say you want to compare two JSON files.
There are [limited tools available](https://github.com/zgrossbart/jdd), which are effectively equivalent to
canonicalizing the JSON (_e.g._, sorting dictionary elements by key) and performing a standard diff. This is not always
sufficient. For exmaple, if a key in a dictionary is changed but its value is not, a traditional diff
will conclude that the entire key/value pair was replaced by the new one, even though the only change was the key
itself. 

Take this JSON as an example:
```json
{
	"foo": [1, 2, 3, 4],
	"bar": "testing"
}
```
Say it has been modified to look like this:
```json
{
	"foo": [2, 3, 4, 5],
	"zab": "testing",
	"woo": ["foobar"]
}
```

A traditional diff would look like this:
```console
$ cat original.json | jq -M --sort-keys > original.canonical.json
$ cat modified.json | jq -M --sort-keys > modified.canonical.json
$ diff -u original.canonical.json modified.canonical.json
```
```diff
 {
-  "bar": "testing",
   "foo": [
-    1,
     2,
     3,
-    4
-  ]
+    4,
+    5
+  ],
+  "woo": [
+    "foobar"
+  ],
+  "zab": "testing"
 }
```
Not entirely useful, particularly if the input files are large.

Here, on the other hand, is what Graphtage will output:
<p align="center">
  <img src="doc/example.png?raw=true" title="Graphtage Example">
</p>
 
## Installation
 
 ```console
$ pip3 install graphtage
```

## Usage

### Output Formatting
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
Finally, use `--condensed` or `-j` to apply both of these options:
```json
{"foo": [1, 2, 3], "bar": "baz"}
```

### Matching Options
By default, Graphtage tries to match all possible pairs of elements in a dictionary. While computationally tractable,
this can sometimes be onerous for input files with huge dictionaries. The `--no-key-edits` or `-k` option will instead
only attempt to match dictionary items that share the same key, drastically reducing computation.

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
option. To additionally suppress all but critical log messages, use `--quiet`. Fine grained control of log messages is
via the `--log-level` option.

## How does it work?

In general, optimally mapping one graph to another
[cannot be executed in polynomial time](https://en.wikipedia.org/wiki/Graph_isomorphism_problem), and is therefore not 
tractable for graphs of any useful size. However, trees and forests are a special case that _can_ be mapped in
polynomial time. Graphtage exploits this.

Ordered nodes in the tree (_e.g._, JSON lists) as well as mappings (_e.g._, JSON dicts) provide additional challenges.

Lists are matched using an “[online](https://en.wikipedia.org/wiki/Online_algorithm)”,
“[constructive](https://en.wikipedia.org/wiki/Constructive_proof)” implementation of the
[Levenshtein distance metric](https://en.wikipedia.org/wiki/Levenshtein_distance). The algorithm starts with an
unbounded mapping and iteratively improves it until the bounds converge, at which point the optimal edit sequence is
discovered.

Dicts are matched by solving the [minimum weight matching problem](https://en.wikipedia.org/wiki/Assignment_problem) on
the complete bipartite graph from key/value pairs in the source dict to key/value pairs in the destination dict.
