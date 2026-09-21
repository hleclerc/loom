"""`sdot-toolchain` — say what this machine can compile for.

    sdot-toolchain            # or: python -m loom.toolchain    -- survey
    sdot-toolchain --device cpu

Each device compiles with its own compiler (`Device.compiler`, see `compilation/Compiler.py`):
the host C++ compiler for the CPU, nvcc around it for CUDA. This command only reports what each
present device would use, and what is missing. It is read-only, and it never fails the process
just because a device is absent -- an absent GPU is a fact about the machine, not an error.
"""
import argparse
import platform
import sys

from .compilation import build_dir, include_dirs
from .compilation.build import ninja_path
from .devices.Device import Device

# The devices we know how to ask about. Metal is intentionally absent: it has its own
# (non-C++) path, so `sdot-toolchain` has nothing to say about it.
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
    print( f"build dir : { build_dir() }" )
    try:
        print( f"ninja     : { ninja_path() }" )
    except RuntimeError as e:
        print( f"ninja     : { _mark( False ) } ({ e })" )
    print( "includes  : " + ", ".join( map( str, include_dirs() ) ) )

    print( "\ndevices:" )
    rc = 0
    for name, device in _devices( only ):
        try:
            compiler = device.compiler
        except NotImplementedError as e:
            print( f"  { name:5s} { e }" )
            continue
        present = device.device_is_present
        print( f"  { name:5s} present={ present }  compiler={ compiler.name } [{ _mark( compiler.is_available() ) }]" )
        for k, v in compiler.describe():
            print( f"        { k:12s}: { v }" )
        if present and not compiler.is_available():
            rc = 1
    return rc


def main( argv = None ) -> int:
    parser = argparse.ArgumentParser( prog = "sdot-toolchain", description = __doc__.splitlines()[ 0 ] )
    parser.add_argument( "--device", choices = DEVICE_NAMES, help = "restrict to one device" )
    args = parser.parse_args( argv )
    return survey( args.device )


if __name__ == "__main__":
    sys.exit( main() )
