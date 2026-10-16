"""The hardware, as a LIST to traverse, and the MASK that makes a machine pretend to be a weaker one.

Why: whatever depends on the hardware may be decided in Python that loom does not manage -- in `sdot`, before
any `ffi_call` (`device.nb_threads`, `device.group_size`, ...). The only way to see what the code generates on
another hardware is to run the same Python again with the device behaving as that hardware. A machine can
pretend to be a WEAKER one -- a CPU with AVX-512 can behave as `x86-64-v3` or `v2`, a card of capability 8.9 as an
8.0 or a 7.0 -- and then the program really runs, kernels included. It cannot pretend to be a stronger one. So:

  * `LOOM_TARGET=<target>` is the mask. A target is the name of an architecture (`x86-64-v3`, `armv8-a`,
    `sm_80`): the one of `Device.catalogue_target()`, the one a catalogue build is filed under. Asking for
    more than the machine has is an error (it cannot be honored); a target of the other family (`sm_80` on a
    CPU) concerns nobody here and is ignored.
  * `runs()` is the list of what to traverse to cover the machine: one `( device, target )` per device present
    and per target it can run, richest first. A script that produces a catalogue loops on it, running its program
    inside `behave_as( run )` (see `compilation/catalogue.py`).
  * What no present device can run is reported (`uncovered`): testing AVX-512 code takes a machine that has it.
"""
from contextlib import contextmanager
from collections import namedtuple
import threading
import os
import sys

from .. import env

Run = namedtuple( "Run", "device target" )


_local = threading.local()


def env_mask():
    """The `LOOM_TARGET` mask, or None: what the user (or a script looping on the hardware) said for the whole process."""
    value = ( env.var( "TARGET" ) or "" ).strip()
    return value or None


def mask():
    """The target the machine currently pretends to be, or None: the scope of a kernel (`masked`) if there is one,
    otherwise `LOOM_TARGET`."""
    return getattr( _local, "target", None ) or env_mask()


@contextmanager
def masked( target ):
    """Inside the block the machine behaves as `target` (None: no change). Lighter than `behave_as`: nothing is
    forgotten, this is what a kernel does around its own call and its own decisions (`loom.kernel_parameters_for`)."""
    previous = getattr( _local, "target", None )
    if target is not None:
        _local.target = target
    try:
        yield
    finally:
        _local.target = previous


def known_targets() -> list:
    from ..compilation.Compiler import X86_LEVELS, CUDA_ARCHS
    return [ *( level for level, _ in X86_LEVELS ), "x86-64", "armv8-a", *CUDA_ARCHS ]


def sm_number( target: str ) -> tuple:
    """`sm_86` -> `( 8, 6 )`, `sm_100` -> `( 10, 0 )`."""
    return int( target[ 3:-1 ] ), int( target[ -1 ] )


def is_cuda_target( target: str ) -> bool:
    return target.startswith( "sm_" )


def apply_mask( real: str, family: list ) -> str:
    """The target the device behaves as: `real`, or the mask if it is a weaker target of the same family
    (`family` lists the targets of the device's kind, richest first; for CUDA the order is that of the
    capability). A mask of another family is ignored; an unknown or stronger one is an error. With no mask, the
    architecture the catalogues serve this machine as ( `catalogue.served_target`: `real`, or a weaker one when
    `loom.prefer_catalogue` and they hold no build for `real` )."""
    m = mask()
    if m is None:
        from ..compilation import catalogue
        return catalogue.served_target( real, family )
    if m not in known_targets():
        raise ValueError( f"LOOM_TARGET={ m !r}: unknown target (expected one of { ', '.join( known_targets() ) })" )
    if is_cuda_target( m ) != is_cuda_target( real ):
        return real
    if is_cuda_target( m ):
        if sm_number( m ) > sm_number( real ):
            raise ValueError( f"LOOM_TARGET={ m }: this card is { real }, it cannot behave as something stronger" )
        return m
    if m not in family:
        return real  # the other CPU family (armv8-a on x86): not ours
    if family.index( m ) < family.index( real ):
        raise ValueError( f"LOOM_TARGET={ m }: this machine is { real }, it cannot behave as something stronger" )
    return m


def neighbours( real: str ) -> list:
    """`real` first, then the architectures whose measurements may stand for it, most alike first: when nothing was
    measured on this very architecture, the closest one that was is the best information there is (a table is
    completed by hand, later, with the architectures that matter). CPU: the poorer levels (the parents). CUDA: the
    same generation below, then above (the sisters), then the older generations."""
    known = known_targets()
    if real not in known:
        return [ real ]
    if is_cuda_target( real ):
        cards = sorted( ( t for t in known if is_cuda_target( t ) and t != real ), key = sm_number )
        major = sm_number( real )[ 0 ]
        same = [ t for t in cards if sm_number( t )[ 0 ] == major ]
        below = [ t for t in reversed( same ) if sm_number( t ) < sm_number( real ) ]
        above = [ t for t in same if sm_number( t ) > sm_number( real ) ]
        older = [ t for t in reversed( cards ) if sm_number( t )[ 0 ] < major ]
        return [ real, *below, *above, *older ]
    from ..compilation.Compiler import cpu_targets
    family = cpu_targets()
    return [ real, *family[ family.index( real ) + 1: ] ] if real in family else [ real ]


def clip( target: str, real: str ) -> str:
    """`target`, or `real` if `target` is stronger than the machine (a neighbour's measurement can say more than this
    machine has)."""
    if is_cuda_target( target ) != is_cuda_target( real ):
        return real
    if is_cuda_target( real ):
        return real if sm_number( target ) > sm_number( real ) else target
    from ..compilation.Compiler import cpu_targets
    family = cpu_targets()
    if target not in family or real not in family:
        return real
    return real if family.index( target ) < family.index( real ) else target


_current = None


def current_run():
    """The `Run` of the enclosing `behave_as`, or None."""
    return _current


def runs() -> list:
    """`[ Run( device, target ) ]`: everything to traverse so that the hardware present is covered, the real
    hardware, whatever the mask. The device is a name `Device.factory` understands (`cpu`, `cuda:0`)."""
    from .Device import Device
    from .CudaGpu import cuda_device_count
    names = [ "cpu", *( f"cuda:{ i }" for i in range( cuda_device_count() ) ) ]
    out = []
    for name in names:
        device = Device.factory( name )
        if device.device_is_present:
            out += [ Run( name, t ) for t in device.runnable_targets() ]
    return out


def uncovered( present: list, kind: str, targets: list ) -> list:
    """The targets of `kind` ( "cpu" | "cuda" ) that no run of `present` can execute."""
    done = { r.target for r in present if r.device.startswith( kind ) }
    return [ t for t in targets if t not in done ]


@contextmanager
def behave_as( run ):
    """Inside the block the default device is `run.device`, behaving as `run.target`: Python that asks the device
    (`loom.resolved_device()`, `nb_threads`, ...) sees that architecture, and what is compiled on the spot is compiled
    for it. What the process remembered of an earlier run is forgotten on the way in and on the way out
    (`forget_caches`).

    Asking for a target the device cannot behave as is an error, on entering."""
    import loom
    global _current
    previous_run = _current
    previous_mask, previous_device = os.environ.get( env.PREFIX + "TARGET" ), loom.default_device
    env.set_var( "TARGET", run.target )
    loom.default_device = run.device
    forget_caches()
    _current = run
    try:
        yield run
    finally:
        _current = previous_run
        if previous_mask is None:
            os.environ.pop( env.PREFIX + "TARGET", None )
        else:
            os.environ[ env.PREFIX + "TARGET" ] = previous_mask
        loom.default_device = previous_device
        forget_caches()


def forget_caches():
    """What a change of device must not reuse: the jit caches of Jax (a traced function would not render its kernels
    again), the rendered sources and the primitives loom keeps. A library with caches of its own that depend on
    the device clears them itself."""
    if "jax" in sys.modules:
        sys.modules[ "jax" ].clear_caches()
    if "loom.choices" in sys.modules:
        sys.modules[ "loom.choices" ].forget()
    for name in ( "loom.drivers.render_key", "loom.drivers.JaxMlirPrimitive" ):
        module = sys.modules.get( name )
        if module is not None:
            module._cache.clear()
