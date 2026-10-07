"""SPIKE, second half -- a size decided by the DATA, on the card, AND UNDER `jit`.

WHAT WE BELIEVED. `examples/splats/README.md` documents a boundary: the host can read a count
that a kernel has just written, but "eager-only" -- `ShapeArray` refuses a tracer and
`capacity_overflows()` returns `None` under `jit`. The conclusion was that a data-dependent size
does not get through a `jit`.

WHAT THIS TEST ESTABLISHES. That boundary is NOT a property of XLA. It is a property of doing the
read-back IN PYTHON. Moved into the handler's C++, it disappears: the handler runs at EXECUTION
time, so it can read a device count, allocate on it, and none of this has to exist at trace time.

The chain, entirely within a single call:

  1. a kernel counts, on the card;
  2. the host -- the handler, which is host code -- reads that count;
  3. it allocates EXACTLY that in the XLA pool ( `XLA_FFI_DeviceMemory_Allocate` );
  4. a kernel compacts into it, another sums it.

No bound is prescribed, neither by Python nor by XLA. Measured on 2026-09-26 on an sm_75: exact
on three sizes in eager, and on four draws under `jit` ( 1989 / 2023 / 2040 / 2074 elements
allocated for the same function compiled ONCE ).

This test needs a GPU ( on CPU, `scratch` goes through `aligned_alloc` and the device read-back
makes no sense -- see `test_scratch.py` ), and skips itself otherwise. The errand `nsdot`
environment is tagged cuda but empty to date; in the meantime, directly:

    job -- env ERRAND_IN_ENV=1 LOOM_DEVICE=cuda PYTHONPATH=<errand>:<loom>/src \
        /data/venvs/sdot/bin/python -m errand test_scratch_gpu
"""
from pathlib import Path

import loom
from loom.testing import host
import loom
from loom import Axis, ShapeVar, RealTensor, compilation
from loom.compilation.FfiCode import FfiCode
from errand import test, skip

import numpy

compilation.register_include_root( Path( __file__ ).resolve().parent / "include" )


_CODE = """
    void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
        SI n = args.inputs.values.shape( 0 );

        // 1. a kernel counts, on the card. We go through the FREE form of `run_parallel`: `counter`
        //    is a scratch, it does not come from `args`, so the short form does not apply.
        auto counter = args.allocator.template view<int>( 1 );
        counter.fill_with( queue, 0 );
        run_parallel( queue, indices_over( n ), loom_tests::Counter(),
                      OutList(), counter, InpList(), args.inputs.values );

        // 2. the HOST reads what the kernel has just written. `Ptr::value()` would do that, but it is
        //    marked `HD` while its transfer path is HOST only -> the CUDA compiler refuses it
        //    under `--Werror cross-execution-space-call`. Hence the direct `copy`.
        int host_m = 0;
        copy( Ptr<int,CpuHostMemorySpace>( &host_m ), counter.data(), 1 );
        SI m = SI( host_m );

        // 3. ... and we allocate EXACTLY that
        auto compact = args.allocator.template view<double>( m );
        counter.fill_with( queue, 0 );
        run_parallel( queue, indices_over( n ), loom_tests::Compactor(),
                      OutList(), compact, OutList(), counter, InpList(), args.inputs.values );

        // 4. something to check: the sum of the positives, and the size that was allocated
        run_parallel( queue, indices_over( m ), loom_tests::Summer(),
                      OutList(), args.outputs.total, InpList(), compact );
        run_parallel( queue, indices_over( SI( 1 ) ), loom_tests::Setter(),
                      OutList(), args.outputs.total, InpList(), double( m ) );
    }
"""


def _code():
    return FfiCode( code = _CODE, allocator = True,
                            includes = [ "loom_tests/scratch_gpu.h" ] )


def _compute( x ):
    v = RealTensor[ Axis( ShapeVar( len( x ) ), name = "num_point" ) ]( x )
    total = RealTensor[ Axis( ShapeVar( 2 ), name = "num_output" ) ]()
    loom.ffi_call( "test_scratch_gpu", _code(), values = v, total = loom.out( total ) )
    return total


def _expected( x ):
    return float( x[ x > 0 ].sum() ), int( ( x > 0 ).sum() )


def _on_gpu():
    """Is there a GPU here? The question is about loom's DEVICE, not about the presence of a card:
    without CUDA jaxlib, loom falls back to the CPU and this test no longer has a point."""
    try:
        return bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    except Exception:
        return False


_GPU = _on_gpu()
_NO_GPU = "needs a CUDA GPU -- none here ( see the file header for the command )"


if test( "size_decided_by_the_data" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # three different useful sizes, for a single source and with no bound
        for seed, n in ( ( 0, 1000 ), ( 1, 5000 ), ( 2, 37 ) ):
            x = numpy.random.default_rng( seed ).normal( size = n )
            s, m = _compute( x ).raw.tolist()
            expected_s, expected_m = _expected( x )
            assert int( m ) == expected_m, ( n, int( m ), expected_m )
            assert abs( s - expected_s ) < 1e-9, ( n, s, expected_s )
        print( f"scratch: exact size read on the card, on { loom.resolved_device() }" )


if test( "under_jit" ):
    if not _GPU:
        skip( _NO_GPU )
    else:
        # THE point. The function is compiled ONCE; the allocated size changes on every call.
        n = 4096

        def compute( x ):
            return _compute( x ).value

        compile = loom.jit( compute )

        sizes = []
        for seed in range( 4 ):
            x = numpy.random.default_rng( seed ).normal( size = n )
            r = host( compile( loom.array( x ) ) )
            expected_s, expected_m = _expected( x )
            assert int( r[ 1 ] ) == expected_m, ( seed, int( r[ 1 ] ), expected_m )
            assert abs( float( r[ 0 ] ) - expected_s ) < 1e-8, ( seed, float( r[ 0 ] ), expected_s )
            sizes.append( expected_m )

        # and they DIFFER: otherwise the test would pass with a prescribed capacity
        assert len( set( sizes ) ) > 1, sizes
        print( f"scratch UNDER JIT: a single compilation, allocated sizes { sizes }" )
