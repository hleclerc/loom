"""WHICH ARCHITECTURE a kernel runs as: an explicit memory, never a guess.

A machine has a real architecture (`x86-64-v4`, `sm_89`, `armv8-a`), but for a given kernel a poorer one may do as
well -- AVX-512 that gains nothing over AVX2, a card of capability 8.9 that runs the 8.6 build as fast. What was
measured is kept in a directory, one file per (kernel, architecture), so that machines that measure in parallel never
write the same file and the directories merge by copying:

    <root>/<name>_<hash>/params.json      what the hash is made of (name, extra keys) -- for humans
    <root>/<name>_<hash>/<arch>.json      { "use": "x86-64-v3", "date": ..., "loom": ..., "cpu": ..., ... }

`<arch>` is the REAL architecture of the machine that measured. Not every configuration is measured on every
architecture, so when this machine's own file is missing the closest one stands for it (`hardware.neighbours`: the
poorer CPU levels; the same CUDA generation, below then above, then the older ones), its answer clipped to what this
machine can be. A table is made precise by adding the file of the architecture that matters. No file at all: the kernel
runs as the real architecture. (This is about the TABLE only: which library serves a kernel is decided by the tag.)

The way in is an object, made before the kernel is called, that fixes and keeps what was retained:

    kp = loom.kernel_parameters_for( "my_kernel", dtype = "float32", dim = 3 )
    nt = kp.nb_threads( ... )            # the device, seen as the architecture chosen for THIS kernel
    kp.ffi_call( kernel, x = x, ... )    # the call runs as that architecture, the library loaded is that one's

Extra keys are compile-time facts that change the code or the choice (types, dimension, a size class...). They are
hashed with the name; keep them coarse, each combination is a directory.

`LOOM_TARGET` stays above the table (a script that traverses the hardware, a benchmark); `LOOM_KERNELS=compile`
(development) ignores the table and runs as the real hardware.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os

from .util.encode_base_62 import encode_base_62
from .devices import hardware

_roots = []       # choices directories, in registration order
_cache = {}       # ( name, keys, device id, mask ) -> KernelParameters


def register_choices( root ):
    """Add a choices directory. A missing one is not an error (a checkout has none)."""
    root = Path( root )
    if root.is_dir() and root not in _roots:
        _roots.append( root )
        _cache.clear()


def forget():
    _cache.clear()


def _canonical( extra_keys: dict ) -> str:
    return json.dumps( { str( k ): str( v ) for k, v in extra_keys.items() }, sort_keys = True )


def directory_name( name: str, extra_keys: dict ) -> str:
    h = hashlib.sha256( f"{ name }|{ _canonical( extra_keys ) }".encode() ).hexdigest()
    return f"{ name }_{ encode_base_62( h )[ :10 ] }"


def read( name: str, extra_keys: dict, real_target: str, roots = None ):
    """`( use, measured_on )`: the target recorded for this kernel on this architecture, or on the closest one that
    has an entry (`hardware.neighbours`) -- None if no file at all. `use` is clipped to what this machine can be."""
    d = directory_name( name, extra_keys )
    for t in hardware.neighbours( real_target ):
        for root in ( _roots if roots is None else roots ):
            path = root / d / f"{ t }.json"
            if path.is_file():
                use = json.loads( path.read_text() ).get( "use" )
                return ( use if t == real_target else hardware.clip( use, real_target ) ), t
    return None


def record( root, name: str, extra_keys: dict, real_target: str, use: str, **provenance ):
    """Write the choice of (kernel, keys) for an architecture: atomic, and one file per architecture."""
    d = Path( root ) / directory_name( name, extra_keys )
    d.mkdir( parents = True, exist_ok = True )
    params = d / "params.json"
    if not params.exists():
        params.write_text( json.dumps( { "name": name, "extra_keys": json.loads( _canonical( extra_keys ) ) }, indent = 1 ) )
    target = d / f"{ real_target }.json"
    if provenance.get( "emulated_on" ) and target.exists() and not json.loads( target.read_text() ).get( "emulated_on" ):
        return target                    # a measurement on the real architecture is worth more than a machine behaving as it
    body = { "use": use, "date": datetime.date.today().isoformat(), **provenance }
    tmp = d / f".{ real_target }.{ os.getpid() }.tmp"
    tmp.write_text( json.dumps( body, indent = 1, sort_keys = True ) )
    tmp.replace( target )
    _cache.clear()
    return target


class KernelParameters:
    """What was retained for one kernel: the device and the architecture it behaves as."""

    def __init__( self, name, extra_keys, device, real, target, origin ):
        self.name, self.extra_keys, self.device = name, dict( extra_keys ), device
        self.real, self.target, self.origin = real, target, origin   # origin: "table" | "table:<neighbour>" | "mask" | "real"

    def __repr__( self ):
        return f"KernelParameters({ self.name !r}, { self.extra_keys }, real={ self.real }, target={ self.target }, from { self.origin })"

    def scope( self ):
        """The block in which the machine behaves as the architecture chosen for this kernel."""
        return hardware.masked( self.target if self.origin.startswith( "table" ) else None )

    def ffi_call( self, *kernels, **kwargs ):
        from .calls import ffi_call
        from .compilation import catalogue
        with self.scope(), catalogue.owned_by( { "name": self.name, "extra_keys": json.loads( _canonical( self.extra_keys ) ) } ):
            return ffi_call( self.name, *kernels, **kwargs )

    def __getattr__( self, attr ):
        """`nb_threads`, `group_size`, ...: the device's, answered as the architecture chosen for this kernel."""
        value = getattr( self.__dict__[ "device" ], attr )
        if not callable( value ):
            with self.scope():
                return getattr( self.device, attr )

        def method( *a, **k ):
            with self.scope():
                return getattr( self.device, attr )( *a, **k )
        return method


def kernel_parameters_for( name: str, *, device = None, **extra_keys ) -> KernelParameters:
    """Retain the device and the architecture for kernel `name` (see the module). `device`: a name or a `Device`, the
    default device otherwise."""
    from .devices.Device import Device
    from .compilation import catalogue
    if device is None:
        import loom
        device = loom.resolved_device()
    device = Device.factory( device )
    cache_key = ( name, _canonical( extra_keys ), id( device ), hardware.mask() )
    got = _cache.get( cache_key )
    if got is not None:
        return got
    real = device.real_target()
    target, origin = None, "real"
    if hardware.mask() is not None:
        origin = "mask"
    elif real is not None and catalogue.policy() != "compile":
        got = read( name, extra_keys, real )
        if got is not None:
            use, measured_on = got
            with hardware.masked( use ):
                device.catalogue_target()     # refuses a target the machine cannot be (stronger, unknown)
            target, origin = use, "table" if measured_on == real else f"table:{ measured_on }"
    if target is None:
        target = device.catalogue_target()
    kp = _cache[ cache_key ] = KernelParameters( name, extra_keys, device, real, target, origin )
    return kp
