Printing Protocol
=================

The protocol for delegating how a :class:`graphtage.TreeNode` or :class:`graphtage.Edit` is printed is as follows:

#. Determine the actual object to be printed:
    * If :obj:`node_or_edit` is an :class:`Edit`:
        * If :obj:`with_edits`, then choose the edit
        * Otherwise, choose :obj:`node_or_edit.from_node`
    * If :obj:`node_or_edit` is a :class:`TreeNode`:
        * If :obj:`with_edits` *and* the node is edited and has a non-zero cost,
            then choose :obj:`node_or_edit.edit`::

                node_or_edit.edit is not None and node_or_edit.edit.bounds().lower_bound > 0

        * Otherwise choose `node_or_edit`
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
