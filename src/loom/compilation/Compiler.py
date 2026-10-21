"""The compilers: what turns a generated source into a loadable library, PER DEVICE.

A device says what it is compiled with (`Device.compiler`): the host compiler for the CPU, `nvcc`
around the host compiler for CUDA. This is the only switch -- `make_library` does not know which
device it serves, it asks the compiler for a command and manages the disk cache, the same for all.

A compiler answers three questions:
  * `commands()` / `rule_for( src )` -- how to compile a source into an object, link a
    library, an executable (see `build.py`, which only knows paths and rules);
  * `build_signature` -- what, OUTSIDE the source, changes the binary (the flags, the machine when
    `-march=native` is part of them); `JaxFfi` puts it in the `.so` name with the source hash,
    so that a change of setting does not fall back on a cache built with another one;
  * `describe()` -- what `sdot-toolchain` displays.

Environment settings, shared:
  * `LOOM_CXX`     : the host compiler (otherwise `CXX`, otherwise `c++` / `clang++` / `g++` on PATH).
  * `SDOT_CXXFLAGS`: extra flags, split into words -- the escape hatch to try a setting
                     without touching the code. They enter the signature.
  * `SDOT_CPU_VARIANT` : compile for a named architecture LEVEL (`x86-64-v2` / `v3` / `v4`,
                     `armv8-a`) instead of `-march=native` -- a binary to take elsewhere, and what
                     the wheel catalogue builds, one variant per level (`cpu_variant()`
                     says which one the present machine can load). `SDOT_NO_MARCH_NATIVE=1`:
                     baseline x86-64.
  * `LOOM_LINEINFO=1`     : `-g`, so that a profiler / sanitizer names the line.
  * `LOOM_BOUNDS_CHECK=1` : arms the bounds check of `TensorView::squeeze` (see
                            `common_macros.h`). One test per access, reserved for diagnosis.
"""
from pathlib import Path
import platform
import shutil
import shlex
import sys
import os
import subprocess
from .. import env


def env_cxxflags() -> list:
    """`SDOT_CXXFLAGS`, split into words."""
    return shlex.split( env.var( "CXXFLAGS", "" ) )


# ── probes, remembered across processes ──────────────────────────────────────────────────────────
# Each probe below compiles ( and runs ) a tiny program: ~70 ms apiece, paid again by EVERY process that
# starts. The answer only moves when the toolchain does, so it is kept on disk next to the other caches,
# keyed on what could change it: the compiler ( resolved path, mtime, size ), the tool directories it
# may live in, the user's own overrides -- and it expires after a week, the net for whatever that key
# does not see ( a library installed next to the compiler ). `LOOM_PROBE_CACHE=0` turns it off.

_PROBE_MAX_AGE = 7 * 24 * 3600


def _probe_key( kind: str, cxx: str, extra = "" ) -> str:
    import hashlib, json
    resolved = shutil.which( cxx ) or cxx
    stats = []
    for f in ( resolved, os.path.realpath( resolved ), "/Library/Developer/CommandLineTools", "/Applications/Xcode.app/Contents/Developer" ):
        try:
            st = os.stat( f )
            stats.append( [ f, st.st_mtime_ns, st.st_size ] )
        except OSError:
            stats.append( [ f, None, None ] )
    ident = json.dumps( [ kind, cxx, stats, os.environ.get( "SDKROOT" ), os.environ.get( "CONDA_PREFIX" ), sys.platform, extra ] )
    return hashlib.sha1( ident.encode() ).hexdigest()


def _probe( kind: str, cxx: str, compute, extra = "" ):
    """`compute()` ( a JSON-able answer ), remembered on disk across processes."""
    import json, time
    if env.var( "PROBE_CACHE", "1" ).strip().lower() in ( "0", "false", "no", "off" ):
        return compute()
    try:
        from . import cache_root
        path = cache_root() / "probes.json"
        key = _probe_key( kind, cxx, extra )
        try:
            known = json.loads( path.read_text() )
        except ( OSError, ValueError ):
            known = {}
        hit = known.get( key )
        if hit is not None and time.time() - hit[ "when" ] < _PROBE_MAX_AGE:
            return hit[ "value" ]
    except Exception:
        return compute()
    value = compute()
    try:
        known = { k: v for k, v in known.items() if time.time() - v[ "when" ] < _PROBE_MAX_AGE }
        known[ key ] = { "when": time.time(), "value": value }
        tmp = path.with_name( f".probes.{ os.getpid() }.tmp" )
        tmp.write_text( json.dumps( known ) )
        os.replace( tmp, path )
    except Exception:
        pass
    return value


# ── OpenMP ───────────────────────────────────────────────────────────────────────────────────────
# Off unless asked ( `LOOM_OPENMP=1`, or `LOOM_OPENMP_FLAG` given ): the kernels run on loom's own thread pool, and an OpenMP
# runtime is one more thing a wheel would depend on ( `libgomp`, or `libomp` which a user's machine does not have ). When asked, it is
# on if the compiler can do it ( g++, a clang with libomp; Apple's clang cannot: `_OPENMP` is then not defined ).

_openmp_cache = {}
_build_signature_cache = {}


def _openmp_flag() -> str:
    """`-fopenmp` unless `LOOM_OPENMP_FLAG` says otherwise ( OpenMP is only used when `LOOM_OPENMP=1` ): `-fopenmp=libgomp` makes clang use the GNU runtime ( `libgomp.so.1`, which
    a `manylinux` wheel may depend on ) instead of LLVM's `libomp.so` ( which a user's machine does not have )."""
    return env.var( "OPENMP_FLAG", "" ).strip() or "-fopenmp"


def link_flags() -> list:
    """`LOOM_LDFLAGS`: what the user adds to every link ( e.g. `-static-libstdc++ -static-libgcc` for a library that must not need a newer
    `libstdc++` than the machine it is installed on has )."""
    import shlex
    return shlex.split( env.var( "LDFLAGS", "" ) )


def _probe_openmp( cxx, sysroot ) -> list:
    import subprocess, tempfile
    src = b"#include <omp.h>\nint main() { int n = 0;\n#pragma omp parallel reduction(+:n)\n n += 1; return n > 0 ? 0 : 1; }\n"
    with tempfile.TemporaryDirectory() as tmp:
        exe = str( Path( tmp ) / "probe" )
        try:
            ok = subprocess.run( [ cxx, _openmp_flag(), *link_flags(), "-x", "c++", "-", "-o", exe, *sysroot ], input = src,
                                 capture_output = True, timeout = 60 ).returncode == 0 \
                 and subprocess.run( [ exe ], capture_output = True, timeout = 60 ).returncode == 0
        except ( OSError, subprocess.SubprocessError ):
            ok = False
    return [ _openmp_flag() ] if ok else []


def shared_suffix() -> str:
    """The extension of a shared library on this system."""
    return ".dylib" if sys.platform == "darwin" else ".dll" if os.name == "nt" else ".so"


_dll_directories = {}


def load_library( path ):
    """`ctypes.CDLL( path )`. A Windows DLL looks for its dependencies in a few system places and beside itself, not where its
    linker found them ( no rpath ): the places loom builds the libraries a kernel needs are added first."""
    import ctypes
    if os.name == "nt":
        from . import cache_root
        for d in ( Path( path ).parent, cache_root() / "build" / "runtime", cache_root() / "build" / "units" ):
            if d.is_dir() and d not in _dll_directories:     # once each: a long list of them makes LoadLibrary fail ( out of memory )
                _dll_directories[ d ] = os.add_dll_directory( str( d ) )
    try:
        return ctypes.CDLL( str( path ) )
    except OSError as e:
        raise OSError( f"loom: cannot load { path }: { e }" ) from e


def is_elf() -> bool:
    return sys.platform != "darwin" and os.name != "nt"


def openmp_flags( cxx: str | None, sysroot: list = () ) -> list:
    """`[ "-fopenmp" ]` when `cxx` compiles, links AND runs a program that uses it, else `[]`. Probed once per
    compiler."""
    asked = env.var( "OPENMP", "" ).strip().lower() in ( "1", "true", "yes", "on" ) or bool( env.var( "OPENMP_FLAG", "" ).strip() )
    if cxx is None or not asked:
        return []
    key = ( cxx, _openmp_flag(), tuple( link_flags() ) )
    if key not in _openmp_cache:
        _openmp_cache[ key ] = _probe( "openmp", cxx, lambda: _probe_openmp( cxx, sysroot ), extra = [ *sysroot, _openmp_flag(), *link_flags() ] )
    return _openmp_cache[ key ]


# ── macOS: an SDK the linker can read ────────────────────────────────────────────────────────────
# The default SDK ( `xcrun --show-sdk-path` ) can be newer than the installed `ld`: its `.tbd`s
# name an architecture this `ld` does not know ( `tapi error: ... unknown architecture` ), and
# nothing links any more -- not even an `int main(){}`. Older SDKs, on the other hand, work. It is not
# loom's job to repair the machine ( `xcode-select`, updating the tools ), but it can take note of
# it: try an empty program ONCE, and failing that the most recent SDK that links. Nothing
# happens when the user chose for themselves ( `SDKROOT`, or an `-isysroot` in `SDOT_CXXFLAGS` ).

_sysroot_flags_cache = {}


def _links( cxx: str, sysroot: str | None ) -> bool:
    """An empty program, compiled AND linked with `cxx` ( on this sysroot, or the default one )."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [ cxx, "-x", "c++", "-", "-o", str( Path( tmp ) / "probe" ), *( [ "-isysroot", sysroot ] if sysroot else [] ) ]
        try:
            return subprocess.run( cmd, input = b"int main() { return 0; }\n", capture_output = True, timeout = 60 ).returncode == 0
        except ( OSError, subprocess.SubprocessError ):
            return False


def _sdk_candidates() -> list:
    """The machine's macOS SDKs, from most recent to oldest ( excluding the versionless alias, which
    the default already is )."""
    import re, subprocess
    roots = [ Path( "/Library/Developer/CommandLineTools/SDKs" ) ]
    try:
        dev = subprocess.run( [ "xcode-select", "-p" ], capture_output = True, text = True, timeout = 10 ).stdout.strip()
        if dev:
            roots.append( Path( dev ) / "Platforms/MacOSX.platform/Developer/SDKs" )
    except ( OSError, subprocess.SubprocessError ):
        pass
    found = {}
    for root in roots:
        for p in root.glob( "MacOSX*.sdk" ):
            m = re.fullmatch( r"MacOSX(\d+(?:\.\d+)*)\.sdk", p.name )
            if m:
                found.setdefault( tuple( int( x ) for x in m.group( 1 ).split( "." ) ), str( p ) )
    return [ found[ v ] for v in sorted( found, reverse = True ) ]


def sysroot_flags( cxx: str | None ) -> list:
    """`[ "-isysroot", sdk ]` when the default SDK does not link and another one does, otherwise `[]`."""
    if sys.platform != "darwin" or cxx is None:
        return []
    if os.environ.get( "SDKROOT" ) or any( f.startswith( ( "-isysroot", "--sysroot" ) ) for f in env_cxxflags() ):
        return []
    if cxx not in _sysroot_flags_cache:
        def compute():
            flags = []
            if not _links( cxx, None ):
                sdk = next( ( s for s in _sdk_candidates() if _links( cxx, s ) ), None )
                if sdk is not None:
                    flags = [ "-isysroot", sdk ]
            return flags
        _sysroot_flags_cache[ cxx ] = _probe( "sysroot", cxx, compute )
    return _sysroot_flags_cache[ cxx ]


def cpu_model() -> str:
    """The processor, as `-march=native` sees it -- what must enter the name of a `.so`
    compiled for it. `/proc/cpuinfo` on Linux, `platform` elsewhere."""
    try:
        with open( "/proc/cpuinfo" ) as f:
            for line in f:
                if line.lower().startswith( "model name" ) or line.lower().startswith( "flags" ):
                    return line.split( ":", 1 )[ 1 ].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


# the architecture levels, from richest to poorest, and what each requires: a wheel's catalogue
# builds one variant per level, the import loads the richest one the machine supports
X86_LEVELS = (
    ( "x86-64-v4", ( "AVX512F", "AVX512BW", "AVX512CD", "AVX512DQ", "AVX512VL" ) ),
    ( "x86-64-v3", ( "AVX2", "FMA3", "BMI2", "F16C", "LZCNT", "MOVBE" ) ),
    ( "x86-64-v2", ( "SSE42", "SSSE3", "POPCNT" ) ),
)


def cpu_targets() -> list:
    """The architecture levels a catalogue is built for on this family of machine, richest first, the
    baseline (which any processor of the family runs) last."""
    machine = platform.machine().lower()
    if machine in ( "aarch64", "arm64" ):
        return [ "armv8-a" ]
    if machine not in ( "x86_64", "amd64" ):
        return [ machine ]
    return [ level for level, _ in X86_LEVELS ] + [ "x86-64" ]


def cpu_variant() -> str:
    """The architecture level the CPU BEHAVES as (`x86-64-v3`, `armv8-a`...): what this machine supports, lowered by
    the `LOOM_TARGET` mask if there is one (see `devices/hardware.py`). It decides which catalogue builds are
    loaded, how a kernel is compiled on the spot (`HostCxx`) and, for the Python that asks the device, what the
    hardware "is"."""
    from ..devices import hardware
    return hardware.apply_mask( real_cpu_variant(), cpu_targets() )


def real_cpu_variant() -> str:
    """The richest architecture level this machine supports, whatever the mask: the key under which a precompiled
    catalogue files its variants. numpy can read the CPU (`__cpu_features__`); failing that, `/proc/cpuinfo`."""
    machine = platform.machine().lower()
    if machine in ( "aarch64", "arm64" ):
        return "armv8-a"
    if machine not in ( "x86_64", "amd64" ):
        return machine
    features = None
    try:
        from numpy.core._multiarray_umath import __cpu_features__ as f
        features = { k for k, v in f.items() if v }
    except Exception:
        try:
            with open( "/proc/cpuinfo" ) as fh:
                flags = { w.upper() for line in fh if line.startswith( "flags" ) for w in line.split() }
            features = { "AVX512F": "AVX512F" in flags, "AVX512BW": "AVX512BW" in flags, "AVX512CD": "AVX512CD" in flags,
                         "AVX512DQ": "AVX512DQ" in flags, "AVX512VL": "AVX512VL" in flags, "AVX2": "AVX2" in flags,
                         "FMA3": "FMA" in flags, "BMI2": "BMI2" in flags, "F16C": "F16C" in flags, "LZCNT": "ABM" in flags,
                         "MOVBE": "MOVBE" in flags, "SSE42": "SSE4_2" in flags, "SSSE3": "SSSE3" in flags, "POPCNT": "POPCNT" in flags }
            features = { k for k, v in features.items() if v }
        except OSError:
            features = set()
    for level, needs in X86_LEVELS:
        if all( n in features for n in needs ):
            return level
    return "x86-64"


# the compute capabilities a catalogue is built for, richest first (`LOOM_CUDA_ARCH` or
# a package's own script chooses another set)
CUDA_ARCHS = ( "sm_90", "sm_89", "sm_86", "sm_80", "sm_75", "sm_70" )


def find_host_cxx() -> str | None:
    """The host C++ compiler: `LOOM_CXX`, then `CXX`, then the usual names on PATH."""
    for var in ( "LOOM_CXX", "SDOT_CXX", "CXX" ):
        cxx = os.getenv( var )
        if cxx and ( shutil.which( cxx ) or Path( cxx ).is_file() ):
            return cxx
    for name in ( "c++", "clang++", "g++" ):
        if shutil.which( name ):
            return name
    return None


class Compiler:
    """The contract. One instance per device, obtained through `Device.compiler`."""

    name = "compiler"

    def is_available( self ) -> bool:
        raise NotImplementedError

    @property
    def build_signature( self ) -> str:
        raise NotImplementedError

    def commands( self ) -> dict:
        """name -> ( argv TEMPLATE, has a depfile, description ). Expected: one entry per kind of
        source (`rule_for`), plus `link_shared` and `link_executable`.

        A template is a LIST of arguments, not a shell line, and its holes are named:
        `{in}`, `{out}`, `{depfile}`, `{includes}`, `{defines}`, `{extra}` for a compilation;
        `{in}`, `{out}`, `{libs}`, `{soname}` for a link.

        This is what separates THE FLAGS ( here, the compiler's knowledge ) from THE SYNTAX ( where
        the command is rendered ). The same template renders as a ninja rule, runs directly, or
        is written into a `compile_commands.json` -- without this layer knowing which."""
        raise NotImplementedError

    def rule_for( self, src: Path ) -> str:
        """The rule that compiles this source (by extension: `.cpp` -> the host compiler,
        `.cu` -> nvcc)."""
        raise NotImplementedError

    def link_libraries( self, libraries: list ) -> str:
        """The flags to link these shared libraries (paths), rpath included."""
        raise NotImplementedError

    def soname_flags( self, out: Path ) -> str:
        return ""

    def source_suffix( self ) -> str:
        return ".cpp"

    def library_file_name( self, name: str ) -> str:
        return f"lib{ name }{ shared_suffix() }"

    def describe( self ) -> list:
        """Lines (`name`, `value`) for `sdot-toolchain`."""
        raise NotImplementedError


class HostCxx( Compiler ):
    """The host compiler, as is: what the CPU uses.

    `-O3 -march=native` by default. This was NOT the case as long as the clip was scalar: measured,
    `-march=native` gained nothing (see `notes/2026-09-02-perf-2d-lmo.md`), and it froze the
    machine's architecture into a `.so` that the cache named after the generated `.cpp` alone.
    The register kernel of `sdot/cell/Engine2Reg.h` changes the picture: written in asimd, it holds
    eight vertices in an AVX2 register and gets SPLIT into two SSE2 `xmm` without this flag -- measured,
    1e6 2D seeds on the same Xeon: 1.00 s on baseline x86-64, 0.33 s with `-march=native`. The
    cache trap is removed another way: the `.so` name carries the flags AND the processor model
    (`build_signature`).

    `-fvisibility=hidden`: a generated library exports only its entry point (declared
    `visibility( "default" )` by the generated source) -- measured, a kernel's `.so` goes from 5.5 MB
    to 0.3 MB, and none of its thousands of template instantiations is visible from another.

    An `SDOT_CXXFLAGS` that already names a `-march=` / `-mcpu=` takes precedence.
    """

    name = "host c++"

    def __init__( self, cxx: str | None = None, variant: str | None = None, cxxflags: list | None = None, ldflags: list | None = None ):
        self.cxx = cxx or find_host_cxx()
        self._cxxflags = cxxflags                    # None: `LOOM_CXXFLAGS`
        self._ldflags = ldflags                      # None: `LOOM_LDFLAGS`
        # `None` = this machine (`-march=native`); a named level = a portable binary
        self.variant = variant or env.var( "CPU_VARIANT" ) or self._masked_variant()

    @staticmethod
    def _masked_variant():
        """Under a `LOOM_TARGET` mask the machine is compiled for the level it pretends to be, not for what it is:
        otherwise the "weaker" run would execute `-march=native` code and test nothing. Likewise when the catalogue
        serves it as a poorer level ( `catalogue.served_target` ): the Python that sized the call's buffers saw that
        level, so the kernel compiled for the call is of that level too."""
        from ..devices import hardware
        mask = hardware.mask()
        if mask in cpu_targets():
            return mask
        served = cpu_variant()
        return served if served != real_cpu_variant() else None

    def is_available( self ) -> bool:
        return self.cxx is not None

    def link_user_flags( self ) -> list:
        return link_flags() if self._ldflags is None else self._ldflags

    def user_flags( self ) -> list:
        return env_cxxflags() if self._cxxflags is None else self._cxxflags

    def march_flags( self ) -> list:
        if env.flag( "NO_MARCH_NATIVE" ):
            return []
        if any( f.startswith( "-march" ) or f.startswith( "-mcpu" ) for f in self.user_flags() ):
            return []
        if self.variant:
            return [ f"-march={ self.variant }" ]
        return [ "-march=native" ]

    def opt_flags( self ) -> list:
        # `-O3` is worth 4 % on the kernel body (Xeon W-2145, 16 threads, FP64, 1e6 seeds in 2D,
        # leaf = 10: 1.018 s with `-O2`, 0.979 s with `-O3`).
        return [ "-O3", "-fno-math-errno" ]

    def diagnostic_flags( self ) -> list:
        flags = []
        if os.environ.get( "LOOM_LINEINFO" ):
            flags.append( "-g" )
        if os.environ.get( "LOOM_BOUNDS_CHECK" ):
            flags.append( "-DLOOM_BOUNDS_CHECK" )
        return flags

    def flags( self ) -> list:
        # `_USE_MATH_DEFINES`: MinGW hides `M_PI` & co. under a strict -std, and kernel bodies use them
        return [ "-std=c++20", *self.opt_flags(), *self.march_flags(), *self.diagnostic_flags(),
                 *( [ "-D_USE_MATH_DEFINES" ] if os.name == "nt" else [ "-fPIC" ] ), "-pthread", "-fvisibility=hidden", "-fvisibility-inlines-hidden", *sysroot_flags( self.cxx ),
                 *openmp_flags( self.cxx, sysroot_flags( self.cxx ) ), *self.user_flags() ]

    @property
    def build_signature( self ) -> str:
        # asked on EVERY kernel call (it is part of the kernel's name), and the flags it summarizes
        # cost ~0.5 ms to rebuild. A device hands out a NEW compiler object each time, so the memo
        # lives at module level, keyed on what the answer depends on: the compiler's own state and
        # the whole environment (everything the flags read -- `LOOM_*`, `CXXFLAGS`, `SDKROOT` --
        # is in it, and a test that changes one gets its new signature).
        # (`os.environ._data` is the raw bytes mapping: `items()` decodes every variable, ~0.4 ms)
        raw = getattr( os.environ, "_data", None )
        key = ( type( self ).__qualname__, repr( sorted( vars( self ).items() ) ),
                tuple( ( raw if raw is not None else os.environ ).items() ) )
        cached = _build_signature_cache.get( key )
        if cached is not None:
            return cached
        flags = self.flags()
        sig = f"{ self.cxx }|" + " ".join( flags ) + "|" + " ".join( [ _openmp_flag(), *self.link_user_flags() ] )
        if "-march=native" in flags:
            sig += "|" + cpu_model()
        _build_signature_cache[ key ] = sig
        return sig

    def _require( self ):
        if self.cxx is None:
            raise RuntimeError( "loom: no C++ compiler found (LOOM_CXX, CXX, or c++/clang++/g++ on PATH)" )

    def commands( self ):
        self._require()
        # ELF: bind each INTERNAL reference to the local definition -- no PLT for the calls
        # of a generated library to its own template instantiations.
        bsymbolic = [ "-Wl,-Bsymbolic" ] if is_elf() else [ "-Wl,--export-all-symbols" ] if os.name == "nt" else []
        return {
            "cxx":             ( [ self.cxx, *self.flags(), "{includes}", "{defines}", "{extra}",
                                   "-MMD", "-MF", "{depfile}", "-c", "{in}", "-o", "{out}" ],
                                 True, "c++ $in $defines" ),
            "link_shared":     ( [ self.cxx, "-pthread", "-shared", *sysroot_flags( self.cxx ), *openmp_flags( self.cxx, sysroot_flags( self.cxx ) ), *self.link_user_flags(), *bsymbolic, "{soname}",
                                   "-o", "{out}", "{in}", "{libs}" ], False, "link $out" ),
            "link_executable": ( [ self.cxx, "-pthread", *sysroot_flags( self.cxx ), *openmp_flags( self.cxx, sysroot_flags( self.cxx ) ), *self.link_user_flags(), "-o", "{out}", "{in}", "{libs}" ],
                                 False, "link $out" ),
        }

    def rule_for( self, src ):
        return "cxx"

    def source_suffix( self ) -> str:
        """The extension of a source generated for this compiler (`.cu` under nvcc)."""
        return ".cpp"

    def link_libraries( self, libraries ):
        flags = []
        for lib in libraries:
            lib = Path( lib )
            name = lib.name
            for prefix, suffix in ( ( "lib", ".so" ), ( "lib", ".dylib" ), ( "lib", ".dll" ) ):
                if name.startswith( prefix ) and name.endswith( suffix ):
                    name = name[ len( prefix ):-len( suffix ) ]
            flags += [ f"-L{ lib.parent }", f"-l{ name }" ]
        if libraries and os.name != "nt":      # Windows has no rpath: `os.add_dll_directory` / preloading, on the Python side
            # next to ITSELF first, then where it was built: a kernel copied into a catalogue loads the runtime that is
            # copied beside it, not the one of the machine that made it; one that stays in the cache has none beside it.
            # (`$$`: this string goes into a ninja file, where `$` introduces a variable; and it is quoted, for the shell
            # that ninja runs it with would otherwise expand `$ORIGIN` to nothing)
            flags.append( "-Wl,-rpath,@loader_path" if sys.platform == "darwin" else "'-Wl,-rpath,$$ORIGIN'" )
            flags += [ f"-Wl,-rpath,{ Path( lib ).parent }" for lib in libraries ]
        return " ".join( flags )

    def soname_flags( self, out ):
        if os.name == "nt":
            # the import library the next link will find as `-l<name>`, even under `-static` ( which only looks for `lib<name>.a` )
            return f"-Wl,--out-implib,{ Path( out ).with_suffix( '.a' ).as_posix() }"
        return f"-Wl,-install_name,@rpath/{ Path( out ).name }" if sys.platform == "darwin" else ""

    def describe( self ):
        return [ ( "host c++", self.cxx or "not found" ), ( "flags", " ".join( self.flags() ) ),
                 ( "variant", f"{ self.variant or 'native' } (the machine supports { cpu_variant() })" ) ]


def cuda_host_variant() -> str:
    """The level the host half of a CUDA kernel is compiled for when it goes into a catalogue: a fixed one (not this CPU's
    `native`), the same baseline for every machine of the family -- `x86-64-v3` (AVX2) or `armv8-a`."""
    targets = cpu_targets()
    return "x86-64-v3" if "x86-64-v3" in targets else targets[ -1 ]


_supported = {}


def nvcc_supported_archs( nvcc: str | None ) -> tuple:
    """The `sm_XX` this nvcc can generate code for (`--list-gpu-code`; CUDA 13 dropped `sm_70`), in `CUDA_ARCHS` order.
    All of them when nvcc cannot be asked."""
    if nvcc not in _supported:
        found = None
        if nvcc is not None:
            try:
                r = subprocess.run( [ nvcc, "--list-gpu-code" ], capture_output = True, text = True, timeout = 60 )
                found = set( r.stdout.split() ) if r.returncode == 0 and r.stdout.strip() else None
            except ( OSError, subprocess.SubprocessError ):
                found = None
        _supported[ nvcc ] = tuple( a for a in CUDA_ARCHS if found is None or a in found )
    return _supported[ nvcc ]


def find_nvcc() -> str | None:
    """`nvcc`: `SDOT_NVCC`, then the one from the pip package `nvidia-cuda-nvcc` (the same toolkit as
    Jax's CUDA plugin, and a compiler with nothing installed on the machine), then
    `/usr/local/cuda/bin`, then PATH."""
    override = env.var( "NVCC" )
    if override and Path( override ).is_file():
        return override
    import site
    for root in [ *site.getsitepackages(), site.getusersitepackages() ]:
        for p in sorted( Path( root ).glob( "nvidia/cu*/bin/nvcc" ), reverse = True ):
            if p.is_file():
                return str( p )
    for p in ( Path( "/usr/local/cuda/bin/nvcc" ), ):
        if p.is_file():
            return str( p )
    return shutil.which( "nvcc" )


def cudart_link_dir( nvcc: str ) -> list:
    """`[ "-L", dir ]` where the SHARED `libcudart.so` can be linked: the pip toolkit ships `libcudart.so.13` alone ( its `lib/` has no
    unversioned name, which `-lcudart` needs ), so a link directory with the symlink is made in the cache. `[]` when the toolkit has it."""
    lib = Path( nvcc ).resolve().parent.parent / "lib"
    if not lib.is_dir() or ( lib / "libcudart.so" ).exists():
        return []
    versions = sorted( lib.glob( "libcudart.so.*" ) )
    if not versions:
        return []
    from . import cache_root
    link = cache_root() / "cudart" / "libcudart.so"
    try:
        link.parent.mkdir( parents = True, exist_ok = True )
        if link.is_symlink() and link.resolve() != versions[ -1 ].resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to( versions[ -1 ] )
    except OSError:
        return []
    return [ "-L", str( link.parent ) ]


class Nvcc( Compiler ):
    """`nvcc` around the host compiler: what CUDA uses. A `.cu` source goes through nvcc
    (the kernel, host and device code in the same unit), a `.cpp` through the host compiler as
    is; nvcc links (it knows where `libcudart` is).

    `arch` is that of the present card (`sm_75`), read by `CudaGpu.sm_arch`: we compile for
    IT. The catalogue holds one library per kernel and per architecture (`compilation/catalogue.py`).

    `--expt-relaxed-constexpr`: the `constexpr` functions of the standard library
    (`std::min`, `std::forward`, `std::tuple`, ...) become callable from the device -- which the
    kernels' code does everywhere. `--extended-lambda`: a hand-written body can remain a
    lambda, provided it is marked `[] HD ( ... )` (what loom generates, for its part, is a functor).
    """

    name = "nvcc"

    def __init__( self, host: HostCxx | None = None, arch: str | None = None ):
        # the host half of a kernel that goes into a catalogue must load on any machine of the family: the baseline, not this CPU
        from . import catalogue
        # `LOOM_CUDA_HOST`: the host compiler behind nvcc when it is not the CPU's ( nvcc's front end does not parse what a clang
        # announces from C++23 in the headers of a recent libstdc++; a g++ at `-std=c++20` does not ), with its own flags
        # ( `LOOM_CUDA_HOST_CXXFLAGS`, `LOOM_CUDA_HOST_LDFLAGS` ) -- the CPU kernels keep theirs
        other = env.var( "CUDA_HOST" )
        self.host_ldflags = shlex.split( env.var( "CUDA_HOST_LDFLAGS", "" ) ) if other else link_flags()
        self.host = host or HostCxx( cxx = other or None, variant = cuda_host_variant() if catalogue.collecting() else None,
                                     cxxflags = shlex.split( env.var( "CUDA_HOST_CXXFLAGS", "" ) ) if other else None,
                                     ldflags = self.host_ldflags if other else None )
        self.nvcc = find_nvcc()
        self.arch = arch or env.var( "CUDA_ARCH" ) or "native"
        # `LOOM_CUDA_PTX=0`: the cubin alone, no PTX for the driver to compile for a newer card. A catalogue
        # build per architecture has no use for it, except in its poorest target: what a card of another
        # generation, newer than every build, is served ( `catalogue.served_target` ).
        poorest = nvcc_supported_archs( self.nvcc )[ -1: ]
        self.ptx = env.flag( "CUDA_PTX", not catalogue.collecting() or [ self.arch ] == list( poorest ) )
        # `LOOM_CUDART=shared`: the kernel needs `libcudart.so` beside the framework's instead of carrying its own copy (static: ~300 kB
        # of host code per kernel)
        self.cudart = env.var( "CUDART", "shared" if catalogue.collecting() else "static" )
        # `LOOM_CUDA_COMPRESS=0`: no compression of the device code (nvcc >= 12.4 can pack it for size)
        self.compress = env.flag( "CUDA_COMPRESS", catalogue.collecting() )

    def is_available( self ) -> bool:
        return self.nvcc is not None and self.host.is_available()

    def source_suffix( self ) -> str:
        return ".cu"

    def flags( self ) -> list:
        # the `-D`s apply to both passes (nvcc takes them directly); the rest of the host flags
        # goes to the host compiler via `-Xcompiler`
        host_flags = [ f for f in self.host.flags() if f not in ( "-std=c++20", "-pthread" ) and not f.startswith( "-D" ) ]
        defines    = [ f for f in self.host.flags() if f.startswith( "-D" ) ]
        # `--Werror cross-execution-space-call`: calling a host function from device code is
        # a compilation ERROR, not a warning -- otherwise nvcc makes it a trap that
        # shows up at run time as an "illegal memory access", far from the faulty line
        # one architecture: `-arch=sm_75`; several (`SDOT_CUDA_ARCH=sm_70,sm_80,sm_90`, the
        # catalogue): one `-gencode` per architecture, plus the PTX of the highest for whatever
        # comes after
        archs = [ a.strip() for a in self.arch.split( "," ) if a.strip() ]
        if len( archs ) == 1 and self.ptx:
            arch_flags = [ f"-arch={ archs[ 0 ] }" ]
        elif len( archs ) == 1:
            arch_flags = [ f"-gencode=arch=compute_{ archs[ 0 ][ 3: ] },code=sm_{ archs[ 0 ][ 3: ] }" ]
        else:
            arch_flags = [ f"-gencode=arch=compute_{ a[ 3: ] },code=sm_{ a[ 3: ] }" for a in archs ]
            arch_flags.append( f"-gencode=arch=compute_{ archs[ -1 ][ 3: ] },code=compute_{ archs[ -1 ][ 3: ] }" )
        return [ "-std=c++20", f"-ccbin={ self.host.cxx }", *arch_flags, "-O3", *defines,
                 "--expt-relaxed-constexpr", "--extended-lambda", "--Werror", "cross-execution-space-call",
                 f"--cudart={ self.cudart }", *( [ "--compress-mode=size" ] if self.compress else [] ),
                 "-Xcompiler", ",".join( host_flags ),
                 *( [ "-lineinfo" ] if os.environ.get( "LOOM_LINEINFO" ) else [] ) ]

    @property
    def build_signature( self ) -> str:
        return f"nvcc:{ self.nvcc }|" + " ".join( self.flags() ) + "|" + self.host.build_signature

    def commands( self ):
        if not self.is_available():
            raise RuntimeError( "loom: nvcc not found (pip install nvidia-cuda-nvcc-cu13, or LOOM_NVCC=/path/nvcc)" )
        # `.cpp` files keep the host command: a CUDA device compiles both kinds of source
        res = dict( self.host.commands() )
        bsymbolic = [ "-Xlinker", "-Bsymbolic" ] if is_elf() else []
        ccbin = f"-ccbin={ self.host.cxx }"
        # the `.cpp` units are compiled by the host compiler with its OpenMP flag ( `openmp_flags` ): the link must
        # bring the runtime too, which nvcc does not know about -- through the host compiler, as its own link does
        omp = [ a for f in openmp_flags( self.host.cxx, sysroot_flags( self.host.cxx ) ) for a in ( "-Xcompiler", f ) ]
        omp += [ a for f in self.host_ldflags for a in ( "-Xcompiler", f ) ]
        if self.cudart == "shared":
            omp += cudart_link_dir( self.nvcc )            # `LOOM_LDFLAGS` reach the host linker too
        res[ "nvcc" ]            = ( [ self.nvcc, *self.flags(), "{includes}", "{defines}", "{extra}",
                                       "-MD", "-MF", "{depfile}", "-c", "{in}", "-o", "{out}" ],
                                     True, "nvcc $in $defines" )
        res[ "link_shared" ]     = ( [ self.nvcc, ccbin, f"--cudart={ self.cudart }", "-shared", *omp, *bsymbolic, "{soname}",
                                       "-o", "{out}", "{in}", "{libs}" ], False, "link $out" )
        res[ "link_executable" ] = ( [ self.nvcc, ccbin, f"--cudart={ self.cudart }", *omp, "-o", "{out}", "{in}", "{libs}" ],
                                     False, "link $out" )
        return res

    def rule_for( self, src ):
        return "nvcc" if Path( src ).suffix == ".cu" else "cxx"

    def link_libraries( self, libraries ):
        # nvcc does not know `-Wl,`: what goes to the linker goes through `-Xlinker`
        return self.host.link_libraries( libraries ).replace( "'-Wl,-rpath,$$ORIGIN'", "-Xlinker '-rpath,$$ORIGIN'" ) \
                                                    .replace( "-Wl,-rpath,", "-Xlinker -rpath -Xlinker " )

    def soname_flags( self, out ):
        return self.host.soname_flags( out ).replace( "-Wl,-install_name,", "-Xlinker -install_name -Xlinker " )

    def describe( self ):
        return [ ( "nvcc", self.nvcc or "not found" ), ( "arch", self.arch ), *self.host.describe() ]
