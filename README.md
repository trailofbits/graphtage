# Graphtage

[![PyPI version](https://badge.fury.io/py/graphtage.svg)](https://badge.fury.io/py/graphtage)
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
+  "zar": "testing"
 }
```

Here, on the other hand, is what Graphtage will output:
```console
$ graphtage original.json modified.json
```

<div style="margin: auto; background-color: black; color: gray;">
        <span style="font-weight: bold; opacity: 1.0;">{</span><div style="margin-left: 24pt; font-family: monospace; padding: 0;">
                <span style="color: blue;">"<span style="color: white;"><span style="background-color: red;"><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;"></span></span></span></span><span style="color: white;"><span style="background-color: green;"><span style="font-weight: bold; opacity: 1.0;"></span></span></span><span style="color: white;"><span style="background-color: red;"><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;"></span></span></span></span><span style="color: white;"><span style="background-color: green;"><span style="font-weight: bold; opacity: 1.0;">z̟</span></span></span><span style="color: white;"><span style="background-color: red;"><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;">b</span></span></span></span><span style="color: white;"><span style="background-color: green;"><span style="font-weight: bold; opacity: 1.0;"></span></span></span>a<span style="color: white;"><span style="background-color: red;"><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;"></span></span></span></span><span style="color: white;"><span style="background-color: green;"><span style="font-weight: bold; opacity: 1.0;"></span></span></span><span style="color: white;"><span style="background-color: red;"><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;"></span></span></span></span><span style="color: white;"><span style="background-color: green;"><span style="font-weight: bold; opacity: 1.0;">b̟</span></span></span><span style="color: white;"><span style="background-color: red;"><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;">r</span></span></span></span><span style="color: white;"><span style="background-color: green;"><span style="font-weight: bold; opacity: 1.0;"></span></span></span>"</span><span style="font-weight: bold; opacity: 1.0;">: </span><span style="color: green;">"testing"</span><span style="font-weight: bold; opacity: 1.0;">,</span><br />
                <span style="color: blue;">"foo"</span><span style="font-weight: bold; opacity: 1.0;">: </span><span style="font-weight: bold; opacity: 1.0;">[</span><div style="margin-left: 24pt; font-family: monospace; padding: 0;">
                    <span style="font-weight: bold; opacity: 1.0;"><span style="background-color: red;"><span style="color: white;"><span style="text-decoration: line-through;">1</span></span></span></span><span style="font-weight: bold; opacity: 1.0;"><span style="text-decoration: line-through;">,</span></span><br />
                    2<span style="font-weight: bold; opacity: 1.0;">,</span><br />
                    3<span style="font-weight: bold; opacity: 1.0;">,</span><br />
                    4<span style="font-weight: bold; opacity: 1.0;">,̟</span><br />
                    <span style="font-weight: bold; opacity: 1.0;"><span style="background-color: green;"><span style="color: white;">5̟</span></span></span>
                </div>
                <span style="font-weight: bold; opacity: 1.0;">]</span><span style="font-weight: bold; opacity: 1.0;">,̟</span><br />
                <span style="font-weight: bold; opacity: 1.0;"><span style="background-color: green;"><span style="color: white;"><span style="color: blue;">"̟w̟o̟o̟"̟</span>:̟ ̟[̟<div style="margin-left: 24pt; font-family: monospace; padding: 0;">
                    "̟f̟o̟o̟b̟a̟r̟"̟
                </div>
                ]̟</span></span></span>
            </div>
            <br />
            <span style="font-weight: bold; opacity: 1.0;">}</span>
 </div>
 
## Quickstart
 
 ```console
$ pip3 install graphtage
```

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