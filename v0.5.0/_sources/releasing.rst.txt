.. _Releasing:

Cutting a Release
=================

This page is for Graphtage maintainers. It describes how a version reaches PyPI, the GitHub releases
page, and the published documentation.

Graphtage uses ``MAJOR.MINOR.PATCH`` version numbers. While the major version is zero, a release that
adds a filetype, adds a command line option, or raises the minimum Python version is a minor bump; a
release that only fixes bugs is a patch bump.

Where the Version Lives
-----------------------

``graphtage/version.py`` is the single source of truth. Three consumers derive their version from it
and need no edit of their own:

* ``pyproject.toml`` declares ``dynamic = ["version"]`` and points ``[tool.hatch.version]`` at the
  module, so the built distributions take their version from it.
* ``docs/conf.py`` executes the module to set the Sphinx ``version`` and ``release`` values, and to
  build the release link in the sidebar.
* ``bindist/Makefile`` runs the module to name the binary archive.

Three files repeat the version and have to be edited by hand:

* ``docs/_templates/layout.html`` lists every published version in the documentation version picker.
  Add the new version directly below the ``latest`` entry.
* ``README.md`` shows the output of ``graphtage --version`` in the command line options section.
* ``CITATION.cff`` carries ``version`` and ``date-released``.

What the Workflows Do
---------------------

Three workflows make up the release. Each has a different trigger, which is what determines the order
of the steps below.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Workflow
     - Trigger
     - Effect
   * - ``artifacts.yml``
     - A pushed tag matching ``v*``
     - Builds the PyInstaller binaries for Linux and macOS and uploads them to the release for that
       tag, which may be a draft.
   * - ``pythonpublish.yml``
     - A published GitHub release
     - Builds the source distribution and wheel and uploads them to PyPI.
   * - ``publish_docs.yml``
     - A push to ``master``, or a pushed tag matching ``v*``
     - Publishes the built documentation to ``gh-pages/master/`` or ``gh-pages/<tag>/``.

The ``latest`` documentation symlink follows ``master``, not the newest release, so it moves as soon
as the version bump lands rather than when the release is published.

Both tag-triggered workflows check that the tag matches ``graphtage/version.py`` and fail if it does
not. ``artifacts.yml`` uploads with ``gh release upload``, which can only add assets to a release
that already exists: pushing a ``v*`` tag with no release behind it fails the upload rather than
creating a release nobody reviewed.

The Release Procedure
---------------------

#. Confirm that ``master`` is green and that the working tree is clean.

#. Confirm the PyPI trusted publisher. Graphtage publishes with `Trusted Publishing`_ and holds no
   PyPI token, so a missing or misconfigured publisher fails the upload with no fallback. In the
   project's PyPI publishing settings, confirm there is a publisher for the ``trailofbits/graphtage``
   repository, the ``pythonpublish.yml`` workflow, and the ``pypi`` environment.

#. Exercise the binary build before it matters. It only runs on tags, so it can break unnoticed
   between releases:

   .. code-block:: console

       $ gh workflow run "Build binary artifacts" --ref master

   On a branch it builds and smoke tests the binaries and uploads nothing.

#. Bump the version. Edit ``graphtage/version.py``, then the three files that repeat the version.
   Commit and push to ``master``, and wait for the checks to pass.

#. Draft the release. Pin it to the commit you just pushed rather than to ``master``, so that a merge
   in the meantime cannot move the tag:

   .. code-block:: console

       $ BUMP_SHA=$(git rev-parse origin/master)
       $ gh release create v1.2.3 --draft --target "$BUMP_SHA" \
             --title "Graphtage v1.2.3" --notes-file notes.md

   A draft release creates no tag, and nothing is public yet.

#. Push the tag, so that the binaries attach to the draft:

   .. code-block:: console

       $ git tag -a v1.2.3 "$BUMP_SHA" -m "Graphtage v1.2.3"
       $ git push origin v1.2.3

   Wait for the binary build, then confirm both archives arrived and the release is still a draft:

   .. code-block:: console

       $ gh release view v1.2.3 --json isDraft,assets -q '.isDraft, [.assets[].name]'

#. Review the draft on GitHub and publish it. Publishing is what uploads to PyPI. Because the tag
   already exists, no other workflow runs.

The tag has to come after the draft and before the publish. Letting the publish create the tag makes
the release public for as long as the binaries take to build, and leaves a published release with no
binaries if that build fails.

Verifying a Release
-------------------

* The new version appears at ``https://pypi.org/project/graphtage/``.
* Both binary archives are attached to the release, and each runs ``graphtage --version``.
* ``https://trailofbits.github.io/graphtage/v1.2.3/`` renders, and the version picker on
  ``https://trailofbits.github.io/graphtage/latest/`` lists the new version.

The version picker is written into each build, so documentation published before this release will
never list it. Only ``latest`` and the new version's own directory are current.

If Something Goes Wrong
-----------------------

While the release is still a draft, the tag is the only public artifact and it can be withdrawn:

.. code-block:: console

    $ git push --delete origin v1.2.3
    $ gh release edit v1.2.3 --target <new-sha>

After the release is published the version is permanent. PyPI does not allow a filename to be
reused, so a failed upload that has to be rebuilt needs a new patch version.

.. _Trusted Publishing: https://docs.pypi.org/trusted-publishers/
