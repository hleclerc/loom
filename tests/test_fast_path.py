"""The FAST PATH of `Build.run`: a kernel that ninja has declared up to date is not asked about again
until a file ninja would have compared has changed.

What this test pins down is the two halves of the contract (see `Build.run`):

  * it SKIPS ninja when nothing moved -- that is the whole point, ~60 ms per kernel;
  * it NEVER skips what ninja would have rebuilt -- a header in the kernel's closure that changes
    is rebuilt, a header it does not include is not even looked at (it is not in the stamp),
    and a changed command line or a missing output is a rebuild too.

Every step also runs with `LOOM_VERIFY_FAST_PATH=1`, which replays ninja behind a fast-path hit
and fails if the two disagree.
"""
from pathlib import Path
import tempfile
import time
import os

import loom
from loom.compilation import make_library
from loom.compilation import build as build_module
from errand import test


class _Kernel:
    """A tiny library with its own directory and its own header overlay: `used.h` is in its
    closure, `unused.h` sits next to it and is not."""

    def __init__( self ):
        self.root = Path( tempfile.mkdtemp( prefix = "loom-fast-path-" ) )
        self.overlay = self.root / "overlay"
        self.overlay.mkdir()
        self.work = self.root / "work"
        self.work.mkdir()
        self.used = self.overlay / "fp_used.h"
        self.unused = self.overlay / "fp_unused.h"
        self.used.write_text( "#define FP_VALUE 1\n" )
        self.unused.write_text( "#define FP_OTHER 1\n" )
        self.src = self.work / f"kernel{ loom.resolved_device().compiler.source_suffix() }"
        self.src.write_text( '#include "fp_used.h"\nextern "C" int fp_value() { return FP_VALUE; }\n' )
        self.name = f"fp_{ self.root.name[ -8: ] }{ '.dylib' if os.uname().sysname == 'Darwin' else '.so' }"

    def build( self, extra_flags = () ) -> Path:
        return make_library( self.name, [ self.src ], loom.resolved_device(), work_dir = self.work,
                             include_overlay = self.overlay, extra_flags = list( extra_flags ) )


def _count_ninja_runs():
    """Replaces `_Graph.run` with a counting wrapper; returns `( counter, restore )`."""
    original = build_module._Graph.run
    counter = { "runs": 0 }

    def counting( self, targets ):
        counter[ "runs" ] += 1
        return original( self, targets )

    build_module._Graph.run = counting
    return counter, lambda: setattr( build_module._Graph, "run", original )


def _touch_later( path: Path, text: str ):
    """Rewrites `path` so that its mtime is certainly newer, whatever the clock's grain."""
    time.sleep( 0.02 )
    path.write_text( text )


if test( "an_up_to_date_kernel_does_not_start_ninja" ):
    k = _Kernel()
    k.build()                                    # builds, then stamps
    counter, restore = _count_ninja_runs()
    try:
        lib = k.build()
        assert counter[ "runs" ] == 0, f"ninja ran { counter[ 'runs' ] } times on an up-to-date kernel"
        mtime = lib.stat().st_mtime_ns
        k.build()
        assert lib.stat().st_mtime_ns == mtime
    finally:
        restore()


if test( "a_header_of_the_closure_that_changes_is_rebuilt" ):
    k = _Kernel()
    lib = k.build()
    before = lib.stat().st_mtime_ns
    _touch_later( k.used, "#define FP_VALUE 2\n" )
    lib = k.build()
    assert lib.stat().st_mtime_ns > before, "the fast path let a changed header through"

    # ... and the new state is stamped: the next call is fast again
    counter, restore = _count_ninja_runs()
    try:
        k.build()
        assert counter[ "runs" ] == 0
    finally:
        restore()


if test( "a_header_outside_the_closure_changes_nothing" ):
    k = _Kernel()
    lib = k.build()
    before = lib.stat().st_mtime_ns
    _touch_later( k.unused, "#define FP_OTHER 2\n" )
    counter, restore = _count_ninja_runs()
    try:
        k.build()
        assert counter[ "runs" ] == 0, "an unrelated file is not part of the stamp"
    finally:
        restore()
    assert lib.stat().st_mtime_ns == before


if test( "a_changed_command_line_is_a_rebuild" ):
    k = _Kernel()
    lib = k.build()
    before = lib.stat().st_mtime_ns
    time.sleep( 0.02 )
    k.build( extra_flags = [ "-DFP_FLAG=1" ] )
    assert lib.stat().st_mtime_ns > before


if test( "a_deleted_output_is_rebuilt" ):
    k = _Kernel()
    lib = k.build()
    lib.unlink()
    assert k.build().exists()


if test( "the_verification_mode_agrees_with_ninja" ):
    k = _Kernel()
    k.build()
    os.environ[ "LOOM_VERIFY_FAST_PATH" ] = "1"
    try:
        k.build()                                # raises if the fast path claimed too much
        _touch_later( k.used, "#define FP_VALUE 3\n" )
        k.build()
        k.build()
    finally:
        del os.environ[ "LOOM_VERIFY_FAST_PATH" ]
