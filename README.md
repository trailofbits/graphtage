# Graphtage

[![PyPI version](https://badge.fury.io/py/graphtage.svg)](https://badge.fury.io/py/graphtage)
[![Tests](https://github.com/trailofbits/graphtage/workflows/Python%20package/badge.svg)](https://github.com/trailofbits/graphtage/actions)
[![Slack Status](https://slack.empirehacking.nyc/badge.svg)](https://slack.empirehacking.nyc)

Graphtage is a command-line utility and [underlying library](https://trailofbits.github.io/graphtage/latest/library.html)
for semantically comparing and merging tree-like structures, such as JSON, XML, HTML, YAML, TOML, plist, and CSS files. Its name is a
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

The `--edit-digest` or `-d` option is like `--only-edits` but prints a more concise context for each edit that is more
human-readable.

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

The `--ignore-list-order` option matches the elements of a list as an unordered collection, so moving an element within
a list is not an edit:
```console
$ graphtage --ignore-list-order original.json modified.json
```
Duplicate elements still count. With this option, `[1, 1, 2]` matches `[2, 1, 1]` but not `[1, 2, 2]`.

This option applies to the lists of every format Graphtage reads, with two exceptions: the rows of a CSV file and the
children of an XML or HTML element stay ordered. Those two node types ignore `--no-list-edits` as well.

Two lists whose elements all match each other cost nothing to compare, however long they are. Comparing two lists that
differ is more expensive than the ordered comparison, and grows faster: matching 30 dictionaries where every element
differs took about 30 seconds in one measurement, against 0.7 seconds without the option. Graphtage logs a warning when
a comparison is large enough for this to matter.

`--ignore-list-order` cannot be combined with `--no-list-edits` or `--no-list-edits-when-same-length`, because those two
options apply only to ordered lists.

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

### Git Integration
Graphtage installs a `graphtage-git-diff` command that implements git's external diff interface, so `git diff` can
render changes to structured files semantically.

Git passes a diff driver seven arguments: the path of the file in the repository, followed by the two revisions to
compare along with their hashes and modes. At least one of those revisions is a temporary copy whose name does not
necessarily carry the original file extension, and Graphtage detects file types from extensions, so
`graphtage-git-diff` takes the file type from the repository path rather than from the files it compares.

To set the driver up, define it in your git configuration:
```console
$ git config --global diff.graphtage.command graphtage-git-diff
```
Then assign it in `.gitattributes` to the file types you want Graphtage to handle:
```console
$ cat .gitattributes
```
```gitattributes
*.json diff=graphtage
*.yaml diff=graphtage
```
`git diff` now formats changes to those files with Graphtage:
```console
$ git diff
```
```json
original.json
{
    "foo": [
        ~~1~~,
        2,
        3,
        4,
        ++5++
    ],
    "++z++~~b~~a++b++~~r~~": "testing",
    ++"woo": [
        "foobar"
    ]++
}
```
```console
$ git diff -- deploy.yaml
```
```yaml
deploy.yaml
name: graphtage
ports: 
- 8080
- ++8443++
replicas: 2 -> 5
```

Assign the driver only to extensions that Graphtage supports. Git stops a diff at the first file its driver fails on,
so a driver assigned to every file fails as soon as it reaches one whose type Graphtage does not recognize.

To try the driver without changing any configuration, set `GIT_EXTERNAL_DIFF` for a single command:
```console
$ GIT_EXTERNAL_DIFF=graphtage-git-diff git diff
```

`git log` and `git show` do not run external diff drivers unless you pass `--ext-diff`:
```console
$ git log --patch --ext-diff
```

To pass Graphtage options through the driver, add them to the command. Spell the value of each option with an equals
sign, because git appends its own arguments and a value passed as a separate argument is indistinguishable from the
first of them:
```console
$ git config --global diff.graphtage.command 'graphtage-git-diff --color --format=yaml'
```
Use `--color` or `-c` when git sends the diff to a pager. Graphtage suppresses the Unicode marks that distinguish
the two sides of a change when its output is not a terminal, and a pager reads from a pipe. The marks survive the
pipe, but the ANSI colors do not.

Git stops the whole diff when a driver exits with a non-zero status, so `graphtage-git-diff` exits with a status of
zero whether or not it finds differences. It reserves a non-zero status for failures that leave it with nothing to
print, such as an unsupported file type or a file that does not parse. Graphtage compares two revisions, so the
driver reports a file that a commit adds or deletes instead of diffing it.

You can also run Graphtage from `git difftool`, which supplies the path of the file in `$MERGED` and the two
revisions in `$LOCAL` and `$REMOTE`. `graphtage-git-diff` ignores the hash and mode arguments, so a `.` stands in for
each of them:
```console
$ git config --global difftool.graphtage.cmd 'graphtage-git-diff "$MERGED" "$LOCAL" . . "$REMOTE" . .'
$ git difftool --no-prompt --tool=graphtage
```

## Why does Graphtage exist?

Diffing tree-like structures with unordered elements is tough. Say you want to compare two JSON files.
There are [limited tools available](https://github.com/zgrossbart/jdd), which are effectively equivalent to
canonicalizing the JSON (_e.g._, sorting dictionary elements by key) and performing a standard diff. This is not always
sufficient. For example, if a key in a dictionary is changed but its value is not, a traditional diff
will conclude that the entire key/value pair was replaced by the new one, even though the only change was the key
itself. See [our documentation](https://trailofbits.github.io/graphtage/latest/howitworks.html) for more information.

## Using Graphtage as a Library

Graphtage has a complete API for programmatically operating its diffing capabilities.
When using Graphtage as a library, it is also capable of diffing in-memory Python objects.
This can be useful for debugging Python code, for example, to determine a differential between two objects.
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
© 2020–2023, Trail of Bits.
