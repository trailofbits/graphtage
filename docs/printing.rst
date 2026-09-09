.. _Printing Protocol:

Printing Protocol
=================

The protocol for delegating how a :class:`graphtage.TreeNode` or :class:`graphtage.Edit` is printed in
:meth:`graphtage.GraphtageFormatter.print` is as follows:

#. Determine the actual object to be printed:
    * If ``node_or_edit`` is an :class:`graphtage.Edit`:
        * If ``with_edits``, then choose the edit
        * Otherwise, choose :attr:`node_or_edit.from_node <graphtage.Edit.from_node>`
    * If ``node_or_edit`` is a :class:`graphtage.TreeNode`:
        * If ``with_edits`` *and* the node is edited and has a non-zero cost,
            then choose :attr:`node_or_edit.edit <graphtage.EditedTreeNode.edit>`::

                isinstance(node_or_edit, EditedTreeNode) and \
                        node_or_edit.edit is not None and node_or_edit.edit.has_non_zero_cost()

            :meth:`graphtage.Edit.has_non_zero_cost` is not a single comparison: it tightens the edit's bounds in a
            loop until either its lower bound exceeds zero or its bounds are definitive. Deciding whether to print an
            edit can therefore do an arbitrary amount of work.

        * Otherwise choose ``node_or_edit``
#. If the chosen object is an edit:
    * See if there is a specialized formatter for this edit by calling
      :meth:`graphtage.formatter.Formatter.get_formatter`
    * If so, delegate to that formatter and return.
    * If not, try calling the edit's :func:`graphtage.Edit.print` method. If :exc:`NotImplementedError` is
      *not* raised, return.
#. If the chosen object is a node, or if we failed to find a printer for the edit:
    * See if there is a specialized formatter for this node by calling
      :meth:`graphtage.formatter.Formatter.get_formatter`
    * If so, delegate to that formatter and return.
    * If not, print a debug warning and delegate to the node's internal print implementation
      :meth:`graphtage.TreeNode.print`.

This is implemented in :meth:`graphtage.GraphtageFormatter.print`. See the :ref:`Formatting Protocol` for how formatters
are chosen.

Status Output
-------------

A :class:`graphtage.printer.Printer` is also a :class:`graphtage.progress.StatusWriter`, which is what draws the
``tqdm`` progress bars that Graphtage shows while it diffs. Both the diff output and the status output share the same
printer, so the printer buffers whole lines and hands them to :func:`tqdm.write` rather than letting the two
interleave.

Pass ``quiet=True`` to suppress the progress bars::

    >>> from graphtage.printer import Printer
    >>> quiet_printer = Printer(quiet=True)

The command line client passes ``quiet=True`` for ``--no-status`` and ``--quiet``. To suppress the output as well as
the status, use :attr:`graphtage.printer.NULL_PRINTER`, a printer that is both quiet and writes to a stream that
discards everything. It is the default value of :attr:`graphtage.BuildOptions.printer`, which is why building a tree
from the library draws no progress bar while the command line client does.

Enabling ANSI Color
-------------------

Importing :mod:`graphtage` does not call :func:`colorama.init` and does not replace :attr:`sys.stdout`. Call
:func:`graphtage.printer.enable_ansi_support` from your application's entry point if you want that behavior. On a
legacy Windows console it is what makes ANSI escape sequences work, by replacing :attr:`sys.stdout` and
:attr:`sys.stderr` with wrappers that translate the sequences into Win32 console calls.

A :class:`graphtage.printer.Printer` captures its output stream when it is constructed, so call
:func:`graphtage.printer.enable_ansi_support` before constructing any printer. A printer constructed first writes past
the wrapper and loses its color.
