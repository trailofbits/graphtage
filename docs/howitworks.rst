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

Parallel Execution
------------------

Starting with Python 3.14, Graphtage supports parallel execution using free-threading (PEP 703). This can provide
significant speedups when individual tree node comparisons are computationally expensive.

**Parallelizable Operations:**

1. **Edge Matrix Computation**: When matching unordered collections, Graphtage computes an m×n matrix of edit costs.
   Each cell can be computed independently, making this embarrassingly parallel.

2. **Levenshtein Diagonal Tightening**: The Levenshtein algorithm builds its matrix diagonally. Cells within each
   diagonal are independent and can be tightened in parallel.

**Profiling Results:**

With Python 3.14 free-threading enabled:

- Edge matrix computation (expensive comparisons): **1.5-2.8x speedup**
- Edge matrix computation (trivial comparisons): Threading overhead dominates (use sequential)
- Diagonal tightening: Threading overhead dominates (use sequential)
- make_distinct(): Threading overhead dominates (always uses sequential)

**Key Insight**: Parallelization only helps when the work per task is expensive enough to exceed threading overhead.
For simple value comparisons or small structures, sequential execution is faster.

**Configuration:**

Parallel execution is automatically enabled when running on Python 3.14+ with the GIL disabled. It can be configured
programmatically via :mod:`graphtage.parallel`:

.. code-block:: python

    from graphtage.parallel import configure, Backend

    # Disable parallel execution
    configure(enabled=False)

    # Force threaded backend
    configure(preferred_backend=Backend.THREADED)

    # Customize minimum problem size for parallelization
    configure(threaded_min_size=1000)

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
