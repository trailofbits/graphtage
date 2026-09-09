How Graphtage Works
===================

In general, optimally mapping one graph to another
cannot be executed in polynomial time [#]_, and is therefore not
tractable for graphs of any useful size [*]_. This is true even for restricted classes of graphs like DAGs [#]_.
However, trees and forests are a special case that *can* be mapped in polynomial time, with reasonable constraints on
the types of edits possible. Graphtage exploits this.

Why Mapping Trees is Complex
----------------------------

Ordered nodes in the tree (*e.g.*, JSON lists) and, in particular, mappings (*e.g.*, JSON dicts) are challenging. Most
extant diffing algorithms and utilities assume that the structures are ordered. Take this JSON as an example:

.. list-table::
    :class: align-center

    * - Original JSON
      - Modified JSON
    * - .. code-block:: json

            {
                "foo": [1, 2, 3, 4],
                "bar": "testing"
            }

      - .. code-block:: json

            {
                "foo": [2, 3, 4, 5],
                "zab": "testing",
                "woo": ["foobar"]
            }

Existing tools effectively canonicalize the JSON (*e.g.*, sort dictionary elements by key and format lists with one
item per line), and then perform a traditional diff:

.. code-block:: console

    $ cat original.json | jq -M --sort-keys > original.canonical.json
    $ cat modified.json | jq -M --sort-keys > modified.canonical.json
    $ diff -u original.canonical.json modified.canonical.json

.. code-block:: diff
    :linenos:

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

Not entirely useful, particularly if the input files are large. The problem is that changing dict keys breaks the diff:
Since "bar" was changed to "zab", the canonical representation changes and they are considered separate edits (lines 2
and 15 of the diff).

Matching Ordered Sequences
--------------------------

Graphtage matches ordered sequences like lists using an "online" [#]_, "constructive" [#]_ implementation of the
Levenshtein distance metric [#]_, similar to the Wagner–Fischer algorithm [#]_. The algorithm starts with an
unbounded mapping and iteratively improves it until the bounds converge, at which point the optimal edit sequence is
discovered. This is implemented in the :mod:`graphtage.levenshtein` module.

Matching Unordered Collections
------------------------------

Dicts are matched by solving the minimum weight matching problem [#]_ on the complete bipartite graph from key/value
pairs in the source dict to key/value pairs in the destination dict. This is implemented in the
:mod:`graphtage.matching` module.

That graph has an edge for every pairing of a source key/value pair with a destination key/value pair, and costing all
of those edges dominates the running time on large dicts. The ``--dict-strategy`` command line option chooses how much
of the graph Graphtage builds. It sets :attr:`graphtage.BuildOptions.allow_key_edits` and
:attr:`graphtage.BuildOptions.auto_match_keys`, so the same three strategies are available to library callers:

``auto``
    The default. Key/value pairs whose keys are equal are matched to each other before the graph is built, and only
    the pairs that are left over become vertices of the bipartite graph. Key edits are still found for those remaining
    pairs, so renaming ``"bar"`` to ``"zab"`` is still reported as a key edit rather than as a removal and an
    insertion.

``match``
    Every source key/value pair is costed against every destination key/value pair, including the ones whose keys are
    already equal. This is the most computationally expensive strategy. It can find a cheaper overall matching than
    ``auto`` when two pairs share a key but have very different values.

``none``
    No key edits are considered at all. The dict is built as a :class:`graphtage.FixedKeyDictNode`, which only
    compares two key/value pairs when their keys are equal; every other pair is inserted or removed. This is the least
    computationally expensive strategy, and it is also what ``--no-key-edits``/``-k`` selects.

.. _Unordered Lists:

Unordered Lists
---------------

Lists are ordered by default, so reordering one costs a sequence of Levenshtein edits. The ``--ignore-list-order``
command line option, or :attr:`graphtage.BuildOptions.ignore_list_order`, builds lists as
:class:`graphtage.UnorderedListNode` instead of :class:`graphtage.ListNode`. That node type is a subclass of
:class:`graphtage.MultiSetNode`, so its elements are matched as an unordered collection by the same bipartite matcher
that dicts use, and reordering costs nothing. Duplicate elements still count, so ``[1, 1, 2]`` matches ``[2, 1, 1]``
but not ``[1, 2, 2]``.

The trade-off runs in both directions. Two lists whose elements are all equal match immediately, however long they
are: a shuffle of 2000 integers takes about 0.01 seconds. Two lists that differ require a bipartite matching over
their symmetric difference, which grows much faster than the ordered comparison: 30 dictionaries of which none match
took about 30 seconds in one measurement, against 0.7 seconds by default.

This applies to every format that builds its lists through :func:`graphtage.json.build_tree`, which is all of them
except the rows of a CSV file and the children of an XML element.

Footnotes
---------

.. [#] https://en.wikipedia.org/wiki/Graph_isomorphism_problem
.. [#] https://en.wikipedia.org/wiki/Directed_acyclic_graph
.. [#] https://en.wikipedia.org/wiki/Online_algorithm
.. [#] https://en.wikipedia.org/wiki/Constructive_proof
.. [#] https://en.wikipedia.org/wiki/Levenshtein_distance
.. [#] https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm
.. [#] https://en.wikipedia.org/wiki/Assignment_problem
.. [*] Unless |pvsnp|_.
.. _pvsnp:
    https://en.wikipedia.org/wiki/P_versus_NP_problem
.. |pvsnp| replace:: :math:`P = NP`
