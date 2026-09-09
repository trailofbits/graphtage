# Graphtage

[![PyPI version](https://badge.fury.io/py/graphtage.svg)](https://badge.fury.io/py/graphtage)
[![Tests](https://github.com/trailofbits/graphtage/workflows/Python%20package/badge.svg)](https://github.com/trailofbits/graphtage/actions)
[![Slack Status](https://slack.empirehacking.nyc/badge.svg)](https://slack.empirehacking.nyc)

Graphtage is a command-line utility and [underlying library](https://trailofbits.github.io/graphtage/latest/library.html)
for semantically comparing and merging tree-like structures, such as JSON, JSON5, XML, HTML, YAML, TOML, INI, CSV,
plist, and Python pickle files, as well as flame graphs. Its name is a portmanteau of “graph” and “graftage”—the
latter being the horticultural practice of joining two trees together such that they grow as one.

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
    "foo": [
        1̶,̶
        2,
        3,
        4,̟
        5̟
    ],
    "b̶z̟ar̶b̟": "testing",̟
    "̟w̟o̟o̟"̟:̟ ̟[̟
        "̟f̟o̟o̟b̟a̟r̟"̟
    ]̟
}
```

## Installation

Graphtage requires Python 3.10 or later. It is tested against Python 3.10 through 3.14.

```console
$ pip3 install graphtage
```

Installing the package puts two commands on your `PATH`: `graphtage`, the diff utility, and `graphtage-git-diff`, the
external diff driver described in [Git Integration](#git-integration).

To work on Graphtage itself, install the `dev` extra, which adds pytest, Ruff, and Sphinx:

```console
$ pip3 install 'graphtage[dev]'
```

## Command Line Usage

### Input File Types
Graphtage infers the type of each input file from its extension. To state the type instead of inferring it, use
`--from-<type>` for the first file and `--to-<type>` for the second. Both flags exist for every format Graphtage
supports: `csv`, `flamegraph`, `html`, `ini`, `json`, `json5`, `pickle`, `plist`, `toml`, `xml`, and `yaml`. For
example, to read a JSON document whose name does not end in `.json`:
```console
$ graphtage --from-json config.txt config.json
```
`--from-mime` and `--to-mime` do the same thing but take a MIME type rather than a format name, which matters for the
formats that Graphtage registers under more than one type. Run `graphtage --help` for the accepted values.

#### Flame Graphs
Graphtage reads flame graphs in the folded stacks format that
[stackcollapse-perf.pl](https://github.com/brendangregg/FlameGraph) and its siblings emit, from files ending in
`.folded` or `.collapsed`. Each line is one stack trace, written as a `;`-delimited list of function names, followed by
a space and the integer number of times that stack trace was sampled.

Diffing two profiles of the same program shows where a performance regression came from: which stack traces gained or
lost samples, and which functions entered or left the call stack. Given `before.folded`:
```
main;work 100
main;work;parse 40
main;work;emit 30
```
and `after.folded`:
```
main;work 100
main;work;parse 55
main;work;flush 30
```
Graphtage reports the changed sample count and the changed frame, and leaves the unchanged stack trace alone:
```console
$ graphtage before.folded after.folded
main;work 100
main;work;parse 40 -> 55
main;work;~~emit~~++flush++ 30
```
Graphtage matches the stack traces that two profiles have in common by their function names, so only the stack traces
unique to one profile are compared against each other. Those are matched the same way Graphtage matches dictionary
entries, which is quadratic in their number: two profiles with a thousand distinct stack traces take about as long to
diff as two dictionaries of the same size.

### Output Formatting
Graphtage performs an analysis on an intermediate representation of the trees that is divorced from the filetypes of the
input files. This means, for example, that you can diff a JSON file against a YAML file. Also, the output format can be
different from the input format(s). By default, Graphtage will format the output diff in the same file format as the
first input file. But one could, for example, diff two JSON files and format the output in YAML. There are several
command-line arguments to specify these transformations, such as `--format`; please check the `--help` output for more
information.

Graphtage sorts the keys of every dictionary it reads, so the output orders keys alphabetically no matter how the
input files order them. The examples in this section all render the file `{"foo": [1, 2, 3], "bar": "baz"}` diffed
against itself.

By default, Graphtage pretty-prints its output with as many line breaks and indents as possible.
```json
{
    "bar": "baz",
    "foo": [
        1,
        2,
        3
    ]
}
```
Use the `--join-lists` or `-jl` option to suppress linebreaks after list items:
```json
{
    "bar": "baz",
    "foo": [1,2,3]
}
```
Likewise, use the `--join-dict-items` or `-jd` option to suppress linebreaks after key/value pairs in a dict:
```json
{"bar": "baz","foo": [
        1,
        2,
        3
    ]}
```
Use `--condensed` or `-j` to apply both of these options:
```json
{"bar": "baz","foo": [1,2,3]}
```

The `--only-edits` or `-e` option will print out a list of edits rather than applying them to the input file in place.

The `--edit-digest` or `-d` option is like `--only-edits` but prints a more concise context for each edit that is more
human-readable.

### Matching Options
By default, Graphtage matches the values of key/value pairs that share a key, and tries to match all possible
pairs of the remaining elements.

Matching two dictionaries with each other is hard. Although computationally tractable, this can sometimes be onerous for 
input files with huge dictionaries. Graphtage has three different strategies for matching dictionaries:
1. `--dict-strategy match` (the most computationally expensive) tries to match all pairs of keys and values between the
   two dictionaries, resulting in a match of minimum edit distance;
2. `--dict-strategy none` (the least computationally expensive) will not attempt to match any key/value pairs unless
   they have the exact same key; and
3. `--dict-strategy auto` (the default) will automatically match the values of any key-value pairs that have identical
   keys and then use the `match` strategy for the remainder of key/value pairs.

`--dict-strategy` also has the short form `-ds`. The `--no-key-edits` or `-k` option is equivalent to
`--dict-strategy none`.

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

### Match Constraints
The `--match-unless` or `-u` and `--match-if` or `-m` options take an expression that decides whether Graphtage may
pair two nodes. Graphtage evaluates the expression once for each pair of nodes it considers, with `from` bound to the
node from the first file and `to` bound to the node from the second. `--match-unless` refuses the pair when the
expression is true; `--match-if` refuses it unless the expression is true. A refused pair is reported as a wholesale
replacement rather than compared element by element.

Expressions are parsed by the `graphtage.expressions` module rather than by `eval`. They support arithmetic and
comparison operators, indexing, attribute lookup, and calls to a fixed set of builtins such as `len` and `sorted`.

Say two files describe the same two servers, each identified by an `id`:
```console
$ echo Original: && cat servers.json && echo Modified: && cat servers.new.json
```
```json
Original:
{
    "primary": {"id": 1, "host": "alpha"},
    "replica": {"id": 2, "host": "beta"}
}
Modified:
{
    "primary": {"id": 1, "host": "alphas"},
    "replica": {"id": 3, "host": "gamma"}
}
```
By default Graphtage pairs the two `replica` records and reports the differences between them, even though they
describe different servers:
```console
$ graphtage servers.json servers.new.json
```
```json
{
    "primary": {
        "host": "alpha++s++",
        "id": 1
    },
    "replica": {
        "host": "~~bet~~++gamm++a",
        "id": 2 -> 3
    }
}
```
Refusing to pair records whose `id` differs reports the second record as replaced instead:
```console
$ graphtage --match-unless "from['id'] != to['id']" servers.json servers.new.json
```
```json
{
    "primary": {
        "host": "alpha++s++",
        "id": 1
    },
    "replica": {
        "host": "beta",
        "id": 2
    } -> {
        "host": "gamma",
        "id": 3
    }
}
```
The two options differ in more than the sense of the test. `--match-unless` binds `from` and `to` to plain Python
values, and leaves a pair unconstrained when the expression raises an error, which is what makes the example above
work on the records without also constraining the strings and integers underneath them. `--match-if` binds `from` and
`to` to Graphtage node objects, and refuses a pair when the expression raises an error. Because the constraint applies
to every node in the tree, including the two roots, an expression that reads a key such as `from['id'] == to['id']`
refuses every pair and collapses the whole diff into one replacement. Prefer `--match-unless`.

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
via the `--log-level` option. `--debug` is equivalent to `--log-level=DEBUG`, and `--quiet` is equivalent to
`--log-level=CRITICAL --no-status`.

### Version Information
`--version` or `-v` writes a line such as `Graphtage version 0.3.1` to STDERR. If you pass it without any input files,
Graphtage prints the version and exits; if you pass input files as well, it prints the version and then computes the
diff. `-dumpversion` writes the raw version to STDOUT and exits without reading any input.

### Exit Status
`graphtage` exits with one of three statuses, so a script can tell the three outcomes apart:

| Status | Meaning |
|--------|---------|
| `0` | The two inputs are semantically identical. |
| `1` | The two inputs differ. |
| `2` | Graphtage could not compute a diff, for example because a file did not parse or its type was not recognized. |

Interrupting Graphtage with `SIGINT` returns `-2`, which a POSIX shell reports as `254`.

Because a status of `1` means "the inputs differ" rather than "something went wrong", a CI job that treats any non-zero
status as a failure will fail on every diff Graphtage finds. Test for `2` to detect an error.

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
    "~~b~~++z++a~~r~~++b++": "testing",
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
© 2020–2026, Trail of Bits.
