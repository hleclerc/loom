"""KERNEL-ONLY time on a CUDA card, read from Python ( `LOOM_KERNEL_TIMING=1` ).

The wall clock of a call measures the call: XLA's dispatch, the handler, the allocations of the
scratch, the copy of the outputs. A GPU benchmark wants, next to it, the time the CARD spent in the
kernels -- what `cudaEventElapsedTime` gives around a launch. `CudaQueue.h` does that when the variable
is set ( read once per library, at its first launch: set it before the first call ), and keeps, per
kernel, the accumulated time, the number of launches, and what the compiler made of the kernel
( registers, local memory, occupancy at the launch's block size ).

Each generated library has its own totals; this module walks the libraries loom has loaded
( `register` is called by the FFI loaders ) and asks each one:

    from loom.devices import kernel_timing
    kernel_timing.enable()                  # before the first kernel call
    kernel_timing.reset()
    ...                                     # the calls to time ( their results READ: the work is done )
    for k in kernel_timing.read():          # one dict per kernel that ran
        print( k[ "code_name" ], k[ "ms" ], k[ "count" ], k[ "regs" ], k[ "occupancy" ] )

`read` and `reset` wait for the launches still in flight. A library compiled before the switch
existed ( a catalogue ) has no totals: it is skipped, and `missing()` names it.
"""
import ctypes
import os

#: target name -> ( code name, ctypes library ), in loading order
_libs = {}

_INTS = ( "count", "regs", "local_bytes", "static_shared", "block", "grid", "blocks_per_sm",
          "max_threads_per_sm", "nb_sm", "is_grouped" )


def register( name, code_name, lib ):
    """called by the loaders ( `JaxFfi.compile_and_register` ) for each library they load"""
    _libs[ name ] = ( code_name, lib )


def enable( on = True ):
    """sets `LOOM_KERNEL_TIMING` for this process -- before the first kernel launch of the
    libraries to time ( each reads it once )"""
    os.environ[ "LOOM_KERNEL_TIMING" ] = "1" if on else "0"


def is_enabled():
    from .. import env
    return env.flag( "KERNEL_TIMING" )


def _entry( lib, name ):
    try:
        return getattr( lib, name )
    except AttributeError:
        return None


def missing():
    """the code names of the loaded libraries that cannot be timed ( no timing entry point )"""
    return [ cn for cn, lib in _libs.values() if _entry( lib, "loom_kernel_timing_read" ) is None ]


def reset():
    """zeroes every total ( after waiting for what is in flight )"""
    for _, lib in _libs.values():
        f = _entry( lib, "loom_kernel_timing_reset" )
        if f is not None:
            f.restype = None
            f()


def read():
    """one dict per kernel that has a slot: `code_name`, `target`, `slot`, `ms` ( accumulated since the
    last `reset` ), `count`, `regs`, `local_bytes`, `static_shared`, `block`, `grid`, `blocks_per_sm`,
    `occupancy` ( resident threads / the SM's maximum ), `is_grouped`"""
    res = []
    for target, ( code_name, lib ) in _libs.items():
        f = _entry( lib, "loom_kernel_timing_read" )
        if f is None:
            continue
        f.restype = ctypes.c_int
        f.argtypes = [ ctypes.c_int, ctypes.POINTER( ctypes.c_double ), ctypes.POINTER( ctypes.c_longlong ) ]
        ms = ctypes.c_double( 0 )
        ints = ( ctypes.c_longlong * len( _INTS ) )()
        nb_slots = f( -1, ctypes.byref( ms ), ints )
        for slot in range( nb_slots ):
            f( slot, ctypes.byref( ms ), ints )
            k = dict( code_name = code_name, target = target, slot = slot, ms = ms.value, **dict( zip( _INTS, ints ) ) )
            k[ "occupancy" ] = k[ "blocks_per_sm" ] * k[ "block" ] / k[ "max_threads_per_sm" ] if k[ "max_threads_per_sm" ] else 0.0
            res.append( k )
    return res


def total_ms( code_name = None ):
    """the kernel time since the last `reset`, summed over the kernels ( of `code_name` only, if given )"""
    return sum( k[ "ms" ] for k in read() if code_name is None or k[ "code_name" ] == code_name )
