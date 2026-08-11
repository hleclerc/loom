"""`sdot-toolchain` — say what this machine can compile for, and get it there.

    sdot-toolchain            # or: python -m sdot.toolchain    -- survey, then advise
    sdot-toolchain install    # acquire what the present devices need (builds acpp if needed)
    sdot-toolchain install --device cuda
    sdot-toolchain path --device cuda

The distribution aims at "nothing to install": a prebuilt AdaptiveCpp travels with the
package (`sdot/_toolchain`, see `adaptive_cpp._cache_candidates`) and kernels compile to the
device-independent `generic` target, so no vendor toolchain is ever needed at run time. This
command is the escape hatch for the machines that fall outside that: it tells you exactly what
is missing, and builds it when it can.

It is deliberately read-only until asked otherwise (`install`), and it never fails the process
just because a device is absent — an absent GPU is a fact about the machine, not an error.
"""
from pathlib import Path
import argparse
import platform
import shutil
import sys
import os

from .compilation import adaptive_cpp as ac
from .devices.Device import Device

# The devices we know how to ask about. Metal is intentionally absent: it has its own
# (non-acpp) path, so `sdot-toolchain` has nothing to say about it.
DEVICE_NAMES = ( "cpu", "cuda" )


def _mark( ok ) -> str:
    return "\033[32mok\033[0m" if ok else "\033[31mmissing\033[0m"


def _devices( only = None ):
    names = ( only, ) if only else DEVICE_NAMES
    for name in names:
        try:
            yield name, Device.factory( name )
        except Exception as e:
            print( f"  { name }: cannot be instantiated ({ e })" )


def survey( only = None ) -> int:
    """Print what is available and what each present device would compile with."""
    print( f"machine   : { platform.system() } { platform.machine() }, python { platform.python_version() }" )
    print( f"acpp      : pinned { ac.ACPP_VERSION }, boost { ac.BOOST_VERSION }" )

    print( "\ncache roots (first writable one is used for new builds):" )
    for root in ac._cache_candidates():
        state = []
        if root.is_dir():
            state.append( "exists" )
            if ( root / "adaptivecpp" ).is_dir():
                state.append( "has acpp" )
        state.append( "writable" if ac._is_writable_dir( root ) else "read-only" )
        print( f"  { root }  [{ ', '.join( state ) }]" )

    packaged = Path( ac.__file__ ).resolve().parents[ 1 ] / "_toolchain"
    print( f"\nbundled toolchain: { packaged } [{ 'present' if packaged.is_dir() else 'absent' }]" )

    print( "\nAdaptiveCpp installs found:" )
    found = False
    for profile in ac.VALID_PROFILES:
        for root in ac._cache_candidates():
            base = root / "adaptivecpp"
            if not base.is_dir():
                continue
            for d in sorted( base.iterdir() ):
                if d.name.startswith( f"{ ac.ACPP_VERSION }-{ profile }-" ) and ( d / "bin" / "acpp" ).is_file():
                    print( f"  { d }" )
                    found = True
    if not found:
        print( "  (none — one will be built on first use, or run `sdot-toolchain install`)" )

    print( "\ndevices:" )
    needs_build = []
    for name, device in _devices( only ):
        present = device.device_is_present
        line = f"  { name:5s} present={ present }"
        if not device.acpp_reachable:
            print( line + "  (not reachable through AdaptiveCpp — separate backend)" )
            continue
        targets, profile, backends = ac.resolve_targets( device )
        ready = ac.is_available( profile, backends )
        print( line + f"  target={ targets } profile={ profile } backends={ backends or '()' } "
                      f"acpp={ _mark( ready ) }" )
        # only nag about hardware that is actually here: a laptop has no CUDA card and does not
        # need to hear about the CUDA toolkit.
        if not ready and present:
            needs_build.append( ( name, profile, backends ) )

    print( "\nbuild prerequisites (only needed to BUILD AdaptiveCpp, not to use it):" )
    cxx = ac._cxx_compiler()
    llvm = ac._find_llvm_config()
    cuda = ac._cuda_toolkit_root()
    print( f"  cmake        : { _mark( shutil.which( 'cmake' ) ) }" )
    print( f"  c++ compiler : { _mark( cxx ) } { cxx or '' }" )
    print( f"  llvm-config  : { _mark( llvm ) } { llvm or '' }"
           + ( "  (clang dev headers ok)" if llvm and ac._llvm_has_clang_dev( llvm ) else
               "  \033[31m(clang dev headers missing)\033[0m" if llvm else "" ) )
    print( f"  cuda toolkit : { _mark( cuda ) } { cuda or '' }" )

    if not needs_build:
        print( "\n→ nothing to do: every present device already has the toolchain it needs." )
        return 0

    print( "\n→ these still need an AdaptiveCpp build:" )
    for name, profile, backends in needs_build:
        print( f"     { name } (profile { profile }, backends { backends or '()' })" )
    print( "   run `sdot-toolchain install` to build them here." )

    # Report what such a build would be missing, without raising.
    for _, profile, backends in needs_build:
        try:
            ac._check_build_tools( profile, backends )
        except RuntimeError as e:
            print( f"\n{ e }" )
            break
    return 0


def install( only = None, force = False ) -> int:
    """Acquire (build if needed) the AdaptiveCpp every present, acpp-reachable device needs."""
    rc = 0
    done = set()
    for name, device in _devices( only ):
        if not device.acpp_reachable:
            continue
        if not device.device_is_present and not only:
            print( f"{ name }: absent from this machine, skipped (ask for it explicitly to force)" )
            continue
        targets, profile, backends = ac.resolve_targets( device )
        if ( profile, backends ) in done:
            continue
        done.add( ( profile, backends ) )
        print( f"\n=== { name }: target { targets }, acpp profile { profile }, backends { backends or '()' } ===" )
        try:
            print( "acpp:", ac.ensure_acpp( profile, backends, force = force ) )
        except Exception as e:
            print( f"FAILED: { e }" )
            rc = 1
    return rc


def show_path( only = None ) -> int:
    for name, device in _devices( only ):
        if not device.acpp_reachable:
            continue
        _, profile, backends = ac.resolve_targets( device )
        print( ac.acpp_path( profile, backends ) )
    return 0


def main( argv = None ) -> int:
    parser = argparse.ArgumentParser( prog = "sdot-toolchain", description = __doc__.splitlines()[ 0 ] )
    parser.add_argument( "command", nargs = "?", default = "check", choices = ( "check", "install", "path" ) )
    parser.add_argument( "--device", choices = DEVICE_NAMES, help = "restrict to one device" )
    parser.add_argument( "--force", action = "store_true", help = "(install) rebuild from scratch" )
    args = parser.parse_args( argv )

    if args.command == "install":
        return install( args.device, args.force )
    if args.command == "path":
        return show_path( args.device )
    return survey( args.device )


if __name__ == "__main__":
    sys.exit( main() )
