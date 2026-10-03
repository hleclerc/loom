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
from .. import env


def env_cxxflags() -> list:
    """`SDOT_CXXFLAGS`, split into words."""
    return shlex.split( env.var( "CXXFLAGS", "" ) )


# ── OpenMP ───────────────────────────────────────────────────────────────────────────────────────
# Some of the code a kernel pulls in is parallel on its own ( the AMGCL linear solvers of the transport ):
# without `-fopenmp` it silently runs on ONE thread -- measured: 4x slower on 2D/3D Newton solves, 2.2x on
# the whole 2D uniform solve. So OpenMP is on whenever the compiler can do it ( g++, a clang with libomp );
# Apple's clang cannot, and the code then falls back to its sequential choices ( `_OPENMP` is not defined ).
# `LOOM_OPENMP=0` turns it off ( e.g. to keep the OpenMP runtime out of a process that has another one ).

_openmp_cache = {}


def openmp_flags( cxx: str | None, sysroot: list = () ) -> list:
    """`[ "-fopenmp" ]` when `cxx` compiles, links AND runs a program that uses it, else `[]`. Probed once per
    compiler."""
    if cxx is None or env.var( "OPENMP", "" ).strip().lower() in ( "0", "false", "no", "off" ):
        return []
    if cxx not in _openmp_cache:
        import subprocess, tempfile
        src = b"#include <omp.h>\nint main() { int n = 0;\n#pragma omp parallel reduction(+:n)\n n += 1; return n > 0 ? 0 : 1; }\n"
        with tempfile.TemporaryDirectory() as tmp:
            exe = str( Path( tmp ) / "probe" )
            try:
                ok = subprocess.run( [ cxx, "-fopenmp", "-x", "c++", "-", "-o", exe, *sysroot ], input = src,
                                     capture_output = True, timeout = 60 ).returncode == 0 \
                     and subprocess.run( [ exe ], capture_output = True, timeout = 60 ).returncode == 0
            except ( OSError, subprocess.SubprocessError ):
                ok = False
        _openmp_cache[ cxx ] = [ "-fopenmp" ] if ok else []
    return _openmp_cache[ cxx ]


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
        flags = []
        if not _links( cxx, None ):
            sdk = next( ( s for s in _sdk_candidates() if _links( cxx, s ) ), None )
            if sdk is not None:
                flags = [ "-isysroot", sdk ]
        _sysroot_flags_cache[ cxx ] = flags
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


def cpu_variant() -> str:
    """The richest architecture level this machine supports (`x86-64-v3`, `armv8-a`...):
    the key under which a precompiled catalogue files its variants. numpy can read the CPU
    (`__cpu_features__`); failing that, `/proc/cpuinfo`."""
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
        return f"lib{ name }.dylib" if sys.platform == "darwin" else f"lib{ name }.so"

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

    def __init__( self, cxx: str | None = None, variant: str | None = None ):
        self.cxx = cxx or find_host_cxx()
        # `None` = this machine (`-march=native`); a named level = a portable binary
        self.variant = variant or env.var( "CPU_VARIANT" ) or None

    def is_available( self ) -> bool:
        return self.cxx is not None

    def march_flags( self ) -> list:
        if env.flag( "NO_MARCH_NATIVE" ):
            return []
        if any( f.startswith( "-march" ) or f.startswith( "-mcpu" ) for f in env_cxxflags() ):
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
        return [ "-std=c++20", *self.opt_flags(), *self.march_flags(), *self.diagnostic_flags(),
                 "-fPIC", "-pthread", "-fvisibility=hidden", "-fvisibility-inlines-hidden", *sysroot_flags( self.cxx ),
                 *openmp_flags( self.cxx, sysroot_flags( self.cxx ) ), *env_cxxflags() ]

    @property
    def build_signature( self ) -> str:
        flags = self.flags()
        sig = f"{ self.cxx }|" + " ".join( flags )
        if "-march=native" in flags:
            sig += "|" + cpu_model()
        return sig

    def _require( self ):
        if self.cxx is None:
            raise RuntimeError( "loom: no C++ compiler found (LOOM_CXX, CXX, or c++/clang++/g++ on PATH)" )

    def commands( self ):
        self._require()
        # ELF: bind each INTERNAL reference to the local definition -- no PLT for the calls
        # of a generated library to its own template instantiations.
        bsymbolic = [] if sys.platform == "darwin" else [ "-Wl,-Bsymbolic" ]
        return {
            "cxx":             ( [ self.cxx, *self.flags(), "{includes}", "{defines}", "{extra}",
                                   "-MMD", "-MF", "{depfile}", "-c", "{in}", "-o", "{out}" ],
                                 True, "c++ $in $defines" ),
            "link_shared":     ( [ self.cxx, "-pthread", "-shared", *sysroot_flags( self.cxx ), *openmp_flags( self.cxx, sysroot_flags( self.cxx ) ), *bsymbolic, "{soname}",
                                   "-o", "{out}", "{in}", "{libs}" ], False, "link $out" ),
            "link_executable": ( [ self.cxx, "-pthread", *sysroot_flags( self.cxx ), *openmp_flags( self.cxx, sysroot_flags( self.cxx ) ), "-o", "{out}", "{in}", "{libs}" ],
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
            for prefix, suffix in ( ( "lib", ".so" ), ( "lib", ".dylib" ) ):
                if name.startswith( prefix ) and name.endswith( suffix ):
                    name = name[ len( prefix ):-len( suffix ) ]
            flags += [ f"-L{ lib.parent }", f"-l{ name }", f"-Wl,-rpath,{ lib.parent }" ]
        return " ".join( flags )

    def soname_flags( self, out ):
        return f"-Wl,-install_name,@rpath/{ Path( out ).name }" if sys.platform == "darwin" else ""

    def describe( self ):
        return [ ( "host c++", self.cxx or "not found" ), ( "flags", " ".join( self.flags() ) ),
                 ( "variant", f"{ self.variant or 'native' } (the machine supports { cpu_variant() })" ) ]


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


class Nvcc( Compiler ):
    """`nvcc` around the host compiler: what CUDA uses. A `.cu` source goes through nvcc
    (the kernel, host and device code in the same unit), a `.cpp` through the host compiler as
    is; nvcc links (it knows where `libcudart` is).

    `arch` is that of the present card (`sm_75`), read by `CudaGpu.sm_arch`: we compile for
    IT. A binary for several architectures is the wheels' business (step 4).

    `--expt-relaxed-constexpr`: the `constexpr` functions of the standard library
    (`std::min`, `std::forward`, `std::tuple`, ...) become callable from the device -- which the
    kernels' code does everywhere. `--extended-lambda`: a hand-written body can remain a
    lambda, provided it is marked `[] HD ( ... )` (what loom generates, for its part, is a functor).
    """

    name = "nvcc"

    def __init__( self, host: HostCxx | None = None, arch: str | None = None ):
        self.host = host or HostCxx()
        self.nvcc = find_nvcc()
        self.arch = arch or env.var( "CUDA_ARCH" ) or "native"

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
        if len( archs ) == 1:
            arch_flags = [ f"-arch={ archs[ 0 ] }" ]
        else:
            arch_flags = [ f"-gencode=arch=compute_{ a[ 3: ] },code=sm_{ a[ 3: ] }" for a in archs ]
            arch_flags.append( f"-gencode=arch=compute_{ archs[ -1 ][ 3: ] },code=compute_{ archs[ -1 ][ 3: ] }" )
        return [ "-std=c++20", f"-ccbin={ self.host.cxx }", *arch_flags, "-O3", *defines,
                 "--expt-relaxed-constexpr", "--extended-lambda", "--Werror", "cross-execution-space-call",
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
        bsymbolic = [] if sys.platform == "darwin" else [ "-Xlinker", "-Bsymbolic" ]
        ccbin = f"-ccbin={ self.host.cxx }"
        res[ "nvcc" ]            = ( [ self.nvcc, *self.flags(), "{includes}", "{defines}", "{extra}",
                                       "-MD", "-MF", "{depfile}", "-c", "{in}", "-o", "{out}" ],
                                     True, "nvcc $in $defines" )
        res[ "link_shared" ]     = ( [ self.nvcc, ccbin, "-shared", *bsymbolic, "{soname}",
                                       "-o", "{out}", "{in}", "{libs}" ], False, "link $out" )
        res[ "link_executable" ] = ( [ self.nvcc, ccbin, "-o", "{out}", "{in}", "{libs}" ],
                                     False, "link $out" )
        return res

    def rule_for( self, src ):
        return "nvcc" if Path( src ).suffix == ".cu" else "cxx"

    def link_libraries( self, libraries ):
        # nvcc does not know `-Wl,`: what goes to the linker goes through `-Xlinker`
        return self.host.link_libraries( libraries ).replace( "-Wl,-rpath,", "-Xlinker -rpath -Xlinker " )

    def soname_flags( self, out ):
        return self.host.soname_flags( out ).replace( "-Wl,-install_name,", "-Xlinker -install_name -Xlinker " )

    def describe( self ):
        return [ ( "nvcc", self.nvcc or "not found" ), ( "arch", self.arch ), *self.host.describe() ]
