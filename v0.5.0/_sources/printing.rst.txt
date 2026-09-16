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
        * If ``with_edits``, the node's own formatter is not already on the stack, *and* the node is edited and has a
            non-zero cost, then choose :attr:`node_or_edit.edit <graphtage.EditedTreeNode.edit>`::

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
    * Record that this node's formatter is on the stack, for the duration of the next two steps.
    * See if there is a specialized formatter for this node by calling
      :meth:`graphtage.formatter.Formatter.get_formatter`
    * If so, delegate to that formatter and return.
    * If not, print a debug warning and delegate to the node's internal print implementation
      :meth:`graphtage.TreeNode.print`.

This is implemented in :meth:`graphtage.GraphtageFormatter.print`. See the :ref:`Formatting Protocol` for how formatters
are chosen.

Re-entering the protocol with the same node
-------------------------------------------

``with_edits`` is an argument to :meth:`graphtage.GraphtageFormatter.print`, but it is not part of the
``print_<NodeType>(printer, node)`` calling convention that formatters implement, so it stops at the formatter
boundary. That matters because formatters hand nodes back to the protocol: a list formatter that meets a nested dict
has no ``print_MappingNode`` of its own, so it calls ``self.parent.print(printer, node)`` to reach the format's dict
formatter, as :meth:`graphtage.json.JSONListFormatter.print_SequenceNode` does. Such a call starts the protocol over
for a node whose edit the outer call has already printed, or has deliberately declined to print.

Recording the node for the duration of its own formatter is what keeps that second pass from printing the edit again.
It applies to the node being printed only, not to its children, so an edit nested inside the delegated subtree still
prints normally.

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
