"""Fill the choices table (`choices.py`) by MEASURING: does a kernel gain from the richest architecture of the machine?

    from loom import benchmark
    m = benchmark.Measurements( "my_package/choices" )
    ...  m.time( "my_kernel", workload, run, dim = 3 )   inside the loop on `hardware.runs()` of the script that builds the catalogue
    m.write()

`workload()` runs the kernel on a representative input and returns what it computed (arrays, anything a framework
can synchronise on). It is run once per architecture the machine can behave as (`hardware.runs`), each time compiled
for it -- in the SAME loop that collects the kernels (`catalogue.collect`), not in another one. What is retained is the POOREST architecture whose time stays within `tolerance` of the best one:
an AVX-512 that gains 3% over AVX2 is recorded as `use: x86-64-v3`, and the kernel then runs (and is shipped) as that.
The file is the one of the REAL architecture of the machine, with the timings and where they come from.

The numbers are only as good as the workload: keep it large enough to be dominated by the kernel (not by the
compilation, which is excluded by a warm-up, nor by Python), and run it on a quiet machine.
"""
from pathlib import Path
import platform
import subprocess
import statistics
import time
import sys

from .devices import hardware
from . import choices


def synchronise( value = None ):
    """Wait for the result: an asynchronous framework (Jax, a card) returns before the work is done. Without a value,
    for everything that is in flight."""
    jax = sys.modules.get( "jax" )
    if jax is not None:
        jax.block_until_ready( value if value is not None else jax.live_arrays() )
    torch = sys.modules.get( "torch" )
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    cupy = sys.modules.get( "cupy" )
    if cupy is not None:
        cupy.cuda.Stream.null.synchronize()
    return value


def time_workload( workload, repeats: int = 5 ) -> float:
    """Median seconds of `repeats` runs, after a warm-up run that absorbs the compilation and the caches."""
    synchronise( workload() )
    times = []
    for _ in range( repeats ):
        t = time.perf_counter()
        synchronise( workload() )
        times.append( time.perf_counter() - t )
    return statistics.median( times )


def retain( timings: dict, tolerance: float ) -> str:
    """`timings`: `{ target: seconds }`, richest first. The poorest target within `tolerance` of the best time."""
    best = min( timings.values() )
    return [ t for t, s in timings.items() if s <= best * ( 1 + tolerance ) ][ -1 ]


def bench_loop( name: str, *, weight: float = 1.0, repeats: int = 3, min_seconds: float = 0.5, max_repeats: int = 200,
                **extra_keys ):
    """Time the body of a loop, for the kernel `loom.kernel_parameters_for( name, **extra_keys )`:

        for _ in loom.bench_loop( "my_kernel", weight = 2 ):
            kp.ffi_call( ... )                      # or whatever calls it, any number of times

    Outside a catalogue build it is a plain loop that runs the body ONCE -- the code is the usage, and stays one. While a
    catalogue is built (`loom.catalogue.build`), the body runs once more to warm up (compilation, caches), then `repeats`
    times (more if that is under `min_seconds` in all, up to `max_repeats`: a short body is mostly noise), each timed to the end of the work (the frameworks are asynchronous: what is in flight is waited for). The
    median is the time of that experiment on the architecture the machine currently behaves as. A warning is printed when the
    timings are too scattered to separate architectures: use a bigger workload.

    A kernel can be run by several experiments (the same kernel, called in different places). To say which architecture
    suits it, each experiment gives the ratio of every architecture's time to the best one, and the ratios are added with
    the experiment's `weight` (1 by default: more for a use that matters more, less for one that is rare)."""
    measurements, run = _active, hardware.current_run()
    if measurements is None or run is None:
        yield None
        return
    synchronise()
    yield None                                # warm-up
    synchronise()
    times, spent = [], 0.0
    # a body that lasts a few milliseconds is mostly noise: keep timing until `min_seconds` have been measured
    while len( times ) < repeats or ( spent < min_seconds and len( times ) < max_repeats ):
        t = time.perf_counter()
        yield None
        synchronise()
        times.append( time.perf_counter() - t )
        spent += times[ -1 ]
    measurements.add( name, extra_keys, run, statistics.median( times ), weight, spread( times ) )


def spread( times: list ) -> float:
    """Relative dispersion of the timings: interquartile range over the median (0 for a perfectly steady measure)."""
    if len( times ) < 4:
        return 0.0
    q = statistics.quantiles( times, n = 4 )
    return ( q[ 2 ] - q[ 0 ] ) / statistics.median( times )


_active = None


def machine_description() -> dict:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run( [ "sysctl", "-n", "machdep.cpu.brand_string" ], capture_output = True, text = True ).stdout.strip() or cpu
        except OSError:
            pass
    elif sys.platform.startswith( "linux" ):
        try:
            cpu = next( l.split( ":", 1 )[ 1 ].strip() for l in open( "/proc/cpuinfo" ) if l.startswith( "model name" ) )
        except ( OSError, StopIteration ):
            pass
    return { "machine": platform.node(), "cpu": cpu, "os": platform.platform() }


class Measurements:
    """The timings of the experiments (`bench_loop`) gathered while a script traverses the hardware, then turned into
    choices. A kernel can be run by several experiments, in different places; which architecture suits it is decided on
    all of them: on each experiment, the time of an architecture is taken relative to the best one (a ratio, so that a
    long experiment does not drown a short one), and the ratios are added with the experiment's weight. The poorest
    architecture whose weighted mean ratio stays within `tolerance` of the best mean is retained.

        with benchmark.Measurements( "my_package/choices" ) as m:
            ...   for _ in bench_loop( "my_kernel" ): ...        # in the usage the script runs on each of `hardware.runs()`
        m.write()                                                  # one file per ( kernel, architecture )
    """

    def __init__( self, out, tolerance: float = 0.05, repeats: int = 5 ):
        self.out, self.tolerance, self.repeats = out, tolerance, repeats
        self.cases = {}        # ( name, canonical keys, device ) -> { ordinal: { "weight", "times": { target: seconds } } }
        self.keys = {}
        self._ordinal = {}     # ( name, canonical keys, device, target ) -> experiments seen so far on this run

    def add( self, name: str, extra_keys: dict, run, seconds: float, weight: float = 1.0, spread: float = 0.0 ):
        """Record one experiment of the kernel on this run's architecture. The n-th experiment of a kernel is the same
        experiment on every run (the usage is the same program), which is how its times are compared across architectures."""
        k = ( name, choices._canonical( extra_keys ), run.device )
        self.keys[ k ] = extra_keys
        count = self._ordinal.get( k + ( run.target, ), 0 )
        self._ordinal[ k + ( run.target, ) ] = count + 1
        case = self.cases.setdefault( k, {} ).setdefault( count, { "weight": weight, "times": {} } )
        case[ "times" ][ run.target ] = seconds
        if spread > self.tolerance:
            import warnings
            warnings.warn( f"loom.bench_loop: noisy timings for '{ name }' on { run.target } (spread { spread:.0%} of the median, "
                           f"{ seconds * 1e3:.2f} ms per run, tolerance { self.tolerance:.0%}): the choice may be wrong, "
                           "use a bigger workload", stacklevel = 3 )

    def time( self, name: str, workload, run, weight: float = 1.0, **extra_keys ):
        self.add( name, extra_keys, run, time_workload( workload, self.repeats ), weight )

    def __enter__( self ):
        global _active
        self._previous, _active = _active, self
        return self

    def __exit__( self, *exc ):
        global _active
        _active = self._previous

    def scores( self, k, targets: list ) -> dict:
        """`{ target: weighted mean of the ratio to the best time }` over the experiments, for these targets only."""
        total, score = 0.0, { t: 0.0 for t in targets }
        for case in self.cases[ k ].values():
            times = { t: case[ "times" ][ t ] for t in targets if t in case[ "times" ] }
            if len( times ) != len( targets ):
                continue                           # an experiment that did not run everywhere says nothing
            best = min( times.values() )
            total += case[ "weight" ]
            for t in targets:
                score[ t ] += case[ "weight" ] * times[ t ] / best
        return { t: v / total for t, v in score.items() } if total else {}

    def decisions( self ) -> list:
        """`[ ( name, extra_keys, device, target, use, scores ) ]`: one decision per architecture the machine can be.
        For the real one it is the measurement; for a weaker one it is what the same timings say once the richer builds
        are out of the picture -- the machine really behaved as that architecture, so a machine that is one gets the answer
        (until someone measures it on the real thing)."""
        out = []
        for k, keys in self.keys.items():
            name, _, device = k
            targets = list( dict.fromkeys( t for c in self.cases[ k ].values() for t in c[ "times" ] ) )     # richest first
            for i, target in enumerate( targets ):
                scores = self.scores( k, targets[ i: ] )
                if scores:
                    out.append( ( name, keys, device, target, retain( scores, self.tolerance ), scores ) )
        return out

    def write( self ) -> list:
        """Record the choices (one file per kernel and real architecture); returns what was decided."""
        from importlib import metadata
        try:
            version = metadata.version( "loom" )
        except metadata.PackageNotFoundError:
            version = None
        done = self.decisions()
        for name, keys, device, target, use, scores in done:
            from .devices.Device import Device
            real = Device.factory( device ).real_target()
            choices.record( self.out, name, keys, target, use, loom = version, tolerance = self.tolerance,
                            scores = { t: round( s, 4 ) for t, s in scores.items() },
                            experiments = [ { "weight": c[ "weight" ], "seconds": { t: round( s, 6 ) for t, s in c[ "times" ].items() } }
                                            for c in self.cases[ ( name, choices._canonical( keys ), device ) ].values() ],
                            **( {} if target == real else { "emulated_on": real } ), **machine_description() )
        return done
