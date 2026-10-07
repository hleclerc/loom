"""WHERE the builds go, and HOW a new kernel gets there.

  * the build directory is the user's cache of the platform, not the checkout nor the directory a
    command is launched from -- so the same kernel built from two directories is one build;
  * a new kernel is built in a private directory and published by a rename, so nobody ever loads a
    half-made one; the directory stays valid after the move (ninja is not asked to redo it), and
    two processes that publish the same kernel agree on the first one.
"""
from pathlib import Path
import tempfile
import time
import os

import loom
from loom.compilation import _resolve_build_dir, make_library
from loom.compilation import build as build_module
from loom.compilation.build import KernelDir
from errand import test


def _env( **values ):
    """Sets environment variables, returns what restores them."""
    saved = { k: os.environ.get( k ) for k in values }
    for k, v in values.items():
        if v is None:
            os.environ.pop( k, None )
        else:
            os.environ[ k ] = v

    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop( k, None )
            else:
                os.environ[ k ] = v
    return restore


if test( "the_default_build_dir_is_in_the_user_cache_whatever_the_current_directory" ):
    cache = tempfile.mkdtemp( prefix = "loom-cache-" )
    elsewhere = tempfile.mkdtemp( prefix = "loom-cwd-" )
    restore = _env( LOOM_CACHE_DIR = cache, LOOM_BUILD_DIR = None, SDOT_BUILD_DIR = None )
    here = os.getcwd()
    try:
        first = _resolve_build_dir( None )
        os.chdir( elsewhere )
        assert _resolve_build_dir( None ) == first
        assert first == Path( cache ) / "build", first
    finally:
        os.chdir( here )
        restore()


if test( "a_relative_override_is_made_absolute_once" ):
    elsewhere = Path( tempfile.mkdtemp( prefix = "loom-cwd-" ) ).resolve()
    here = os.getcwd()
    try:
        os.chdir( elsewhere )
        got = _resolve_build_dir( "my_build" )
        assert got == elsewhere / "my_build" and got.is_dir()
    finally:
        os.chdir( here )


class _Kernel:
    """A tiny kernel built through `KernelDir`, in a build directory of its own."""

    def __init__( self ):
        self.build = Path( tempfile.mkdtemp( prefix = "loom-build-dir-" ) ).resolve()
        self.restore = _env( LOOM_BUILD_DIR = str( self.build ) )
        self.name = f"bd_{ self.build.name[ -8: ] }{ '.dylib' if os.uname().sysname == 'Darwin' else '.so' }"

    def make( self, value = 1 ) -> Path:
        ws = KernelDir( self.name )
        with ws as work:
            src = work / f"kernel{ loom.resolved_device().compiler.source_suffix() }"
            text = f'extern "C" int bd_value() {{ return { value }; }}\n'
            if not ( src.exists() and src.read_text() == text ):      # as the drivers do
                src.write_text( text )
            make_library( self.name, [ src ], loom.resolved_device(), work_dir = work )
        self.ws = ws
        return ws.final / self.name

    def close( self ):
        self.restore()


def _count_ninja_runs():
    original = build_module._Graph.run
    counter = { "runs": 0 }

    def counting( self, targets ):
        counter[ "runs" ] += 1
        return original( self, targets )

    build_module._Graph.run = counting
    return counter, lambda: setattr( build_module._Graph, "run", original )


if test( "a_new_kernel_is_built_in_staging_and_published_by_a_rename" ):
    k = _Kernel()
    try:
        lib = k.make()
        assert lib.is_file() and lib.parent == k.build / "kernels" / k.name
        assert k.ws.staged is not None and not k.ws.staged.exists(), "the staging copy is gone"
        assert not any( ( k.build / "tmp" ).glob( "*" ) ), "nothing is left in tmp"
    finally:
        k.close()


if test( "a_published_kernel_is_up_to_date_where_it_was_moved_to" ):
    k = _Kernel()
    try:
        lib = k.make()
        mtime = lib.stat().st_mtime_ns
        counter, restore = _count_ninja_runs()
        try:
            again = k.make()                    # exists now: worked on in place
            assert counter[ "runs" ] == 0, "the move made ninja redo (or re-ask about) the kernel"
        finally:
            restore()
        assert again == lib and lib.stat().st_mtime_ns == mtime
    finally:
        k.close()


if test( "a_failed_build_publishes_nothing" ):
    k = _Kernel()
    try:
        ws = KernelDir( k.name )
        try:
            with ws as work:
                ( work / "half.o" ).write_text( "x" )
                raise RuntimeError( "boom" )
        except RuntimeError:
            pass
        assert not ws.final.exists() and not ws.staged.exists()
    finally:
        k.close()


if test( "when_someone_published_first_the_other_copy_is_dropped" ):
    import threading
    k = _Kernel()
    original = build_module._BuildLock.acquire
    build_module._BuildLock.acquire = lambda self, published: True    # where `flock` does nothing
    try:
        # the same kernel built twice, in two private directories (one per thread), as two
        # processes would; `b` publishes while `a` is still working
        a, b = KernelDir( k.name ), KernelDir( k.name )
        with a as wa:
            ( wa / "mark" ).write_text( "a" )

            def other():
                with b as wb:
                    assert wb != wa
                    ( wb / "mark" ).write_text( "b" )
            t = threading.Thread( target = other )
            t.start()
            t.join()
        assert ( a.final / "mark" ).read_text() == "b", "the first to publish wins"
        assert not a.staged.exists() and not b.staged.exists()
        assert not any( ( k.build / "tmp" ).glob( "*" ) )
    finally:
        build_module._BuildLock.acquire = original
        k.close()


if test( "two_kernel_dirs_of_one_thread_do_not_disturb_each_other" ):
    k = _Kernel()
    try:
        a, b = KernelDir( "bd_outer" ), KernelDir( "bd_inner" )
        with a as wa:
            ( wa / "mark" ).write_text( "a" )
            with b as wb:
                ( wb / "mark" ).write_text( "b" )
            assert ( wa / "mark" ).read_text() == "a", "the inner one erased the outer one's directory"
        assert ( a.final / "mark" ).read_text() == "a" and ( b.final / "mark" ).read_text() == "b"
    finally:
        k.close()


if test( "a_second_process_waits_for_the_first_instead_of_building_again" ):
    import threading
    import time
    k = _Kernel()
    try:
        order = []
        a, b = KernelDir( k.name ), KernelDir( k.name )
        started = threading.Event()

        def second():
            started.wait()
            with b as wb:
                order.append( ( "b", wb == b.final ) )      # it found the kernel published

        t = threading.Thread( target = second )
        t.start()
        with a as wa:
            assert wa != a.final
            started.set()
            time.sleep( 0.5 )                                # b is waiting on the lock now
            assert not order, "b did not wait"
            ( wa / "mark" ).write_text( "a" )
            order.append( "a built" )
        t.join( 10 )
        assert order == [ "a built", ( "b", True ) ], order
        assert ( b.final / "mark" ).read_text() == "a" and b.staged is None
    finally:
        k.close()


if test( "a_lock_is_free_when_its_holder_dies" ):
    import subprocess
    import sys
    k = _Kernel()
    try:
        code = ( "import os,sys,time\nfrom loom.compilation.build import KernelDir\n"
                 f"ws=KernelDir({ k.name!r }); ws.__enter__(); print('held',flush=True); time.sleep(60)\n" )
        child = subprocess.Popen( [ sys.executable, "-c", code ], stdout = subprocess.PIPE, text = True )
        assert child.stdout.readline().strip() == "held"
        child.kill()                                         # no cleanup of any kind
        child.wait()
        t0 = time.time()
        with KernelDir( k.name ) as work:                    # must not wait for the dead one
            assert work != ( k.build / "kernels" / k.name )
        assert time.time() - t0 < 5
    finally:
        k.close()


if test( "a_build_that_died_half_way_is_redone_not_trusted" ):
    k = _Kernel()
    try:
        lib = k.make()
        # what a link killed with SIGKILL leaves: a truncated file, newer than its inputs, and the
        # marker that the work started and never ended
        time.sleep( 0.05 )
        lib.write_bytes( lib.read_bytes()[ :100 ] )
        ( lib.parent / "ninja" ).mkdir( exist_ok = True )
        ( lib.parent / "ninja" / "building" ).write_text( "pid 1\n" )
        good = k.make()
        assert good == lib and lib.stat().st_size > 100, "the truncated library was taken for a good one"
        assert not ( lib.parent / "ninja" / "building" ).exists()
        # and the next call is the fast path again
        counter, restore = _count_ninja_runs()
        try:
            k.make()
            assert counter[ "runs" ] == 0
        finally:
            restore()
    finally:
        k.close()


if test( "a_normal_end_leaves_no_marker" ):
    k = _Kernel()
    try:
        lib = k.make()
        assert not ( lib.parent / "ninja" / "building" ).exists()
        assert not ( k.build / "ninja" / "building" ).exists()
    finally:
        k.close()
