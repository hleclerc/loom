"""The catalogue is a directory `<tag>/<key>.so`: the architecture is in the tag (`cpu-x86-64-v3`, `cuda-sm_80`),
not in the key; a machine takes the first tag it can load that holds the kernel; a kernel that is not
a machine looks in ONE tag (the architecture it behaves as), never in a poorer one; a machine can behave as a weaker one (`LOOM_TARGET`),
which is how a script covers every architecture on the hardware it has."""
from pathlib import Path
import json
import os
import sys
import tempfile

from loom.compilation import catalogue, Compiler as C
from loom.compilation.FfiCode import FfiCode, Kernels
from loom.devices.Cpu import Cpu
from loom.devices.CudaGpu import CudaGpu
from errand import test


def _cuda( major, minor ):
    return CudaGpu( 0, _attrs = ( 1, 1, 1, 1, 1, major, minor, 1 ) )


def _with_variant( level, f ):
    saved = C.cpu_variant
    C.cpu_variant = lambda: level
    try:
        return f()
    finally:
        C.cpu_variant = saved


if test( "the_key_does_not_depend_on_the_architecture" ):
    k = catalogue.key( "src", [ ( "a.cpp", [ ( "X", "1" ) ] ) ] )
    assert k == catalogue.key( "src", [ ( "a.cpp", [ ( "X", "1" ) ] ) ] )
    assert k != catalogue.key( "src2", [] ) and k != catalogue.key( "src", [] )


if test( "a_device_is_served_from_the_tag_of_the_architecture_it_behaves_as_and_no_other" ):
    cpu = Cpu()
    assert cpu.current_catalogue_tag() == cpu.catalogue_tag( cpu.catalogue_target() )
    assert not hasattr( cpu, "catalogue_tags" )
    assert _cuda( 8, 6 ).current_catalogue_tag() == "cuda-sm_86"
    assert _cuda( 12, 0 ).current_catalogue_tag() == "cuda-sm_120"      # not in the catalogue: compiled, not served by an older tag


if test( "nvcc_builds_a_cubin_alone_or_with_its_ptx" ):
    from loom.compilation.Compiler import Nvcc
    with_ptx = Nvcc( arch = "sm_80" ).flags()
    assert "-arch=sm_80" in with_ptx
    os.environ[ "LOOM_CUDA_PTX" ] = "0"
    try:
        cubin = Nvcc( arch = "sm_80" ).flags()
    finally:
        del os.environ[ "LOOM_CUDA_PTX" ]
    assert "-gencode=arch=compute_80,code=sm_80" in cubin and "-arch=sm_80" not in cubin


# ── the mask: a machine pretends to be a weaker one ──────────────────────────────────────────────────
X86 = [ "x86-64-v4", "x86-64-v3", "x86-64-v2", "x86-64" ]


class _Patch:
    """Replace attributes for the duration of a block: ( obj, name, value ) ..."""
    def __init__( self, *changes, mask = None ):
        self.changes, self.mask = changes, mask
    def __enter__( self ):
        self.saved = [ ( o, n, getattr( o, n ) ) for o, n, _ in self.changes ]
        for o, n, v in self.changes:
            setattr( o, n, v )
        self.old_mask = os.environ.get( "LOOM_TARGET" )
        if self.mask:
            os.environ[ "LOOM_TARGET" ] = self.mask
        else:
            os.environ.pop( "LOOM_TARGET", None )
    def __exit__( self, *exc ):
        for o, n, v in self.saved:
            setattr( o, n, v )
        if self.old_mask is None:
            os.environ.pop( "LOOM_TARGET", None )
        else:
            os.environ[ "LOOM_TARGET" ] = self.old_mask


def _x86( real, mask = None ):
    return _Patch( ( C, "cpu_targets", lambda: list( X86 ) ), ( C, "real_cpu_variant", lambda: real ), mask = mask )


if test( "a_cpu_behaves_as_the_weaker_level_it_is_asked_to" ):
    with _x86( "x86-64-v4", mask = "x86-64-v3" ):
        cpu = Cpu()
        assert cpu.real_target() == "x86-64-v4" and cpu.catalogue_target() == "x86-64-v3"
        assert cpu.current_catalogue_tag() == "cpu-x86-64-v3"
        # and it really compiles as that level, instead of running `-march=native` under another name
        assert "-march=x86-64-v3" in C.HostCxx( "c++" ).march_flags()
    with _x86( "x86-64-v4" ):
        assert Cpu().catalogue_target() == "x86-64-v4"
        assert "-march=native" in C.HostCxx( "c++" ).march_flags()
        assert Cpu().runnable_targets() == X86


if test( "a_mask_cannot_ask_for_more_than_the_machine_has_nor_for_what_does_not_exist" ):
    for mask, ok in ( ( "x86-64-v4", False ), ( "x86-64-v5", False ), ( "x86-64-v2", True ), ( "sm_80", True ), ( "armv8-a", True ) ):
        with _x86( "x86-64-v3", mask = mask ):
            try:
                got = Cpu().catalogue_target()
                assert ok, f"{ mask } should have been refused"
                # a target of another family concerns nobody here
                assert got == ( mask if mask in X86 else "x86-64-v3" )
            except ValueError as e:
                assert not ok and mask in str( e )


if test( "a_card_behaves_as_a_weaker_one_with_that_generations_limits" ):
    real = ( 82, 1536, 65536, 102400, 24 << 30, 8, 9, 49152 )  # an sm_89
    gpu = CudaGpu( 0, _attrs = real )
    with _Patch( mask = "sm_80" ):
        assert gpu.real_target() == "sm_89" and gpu.sm_arch == "sm_80" and gpu.catalogue_target() == "sm_80"
        nb_sm, thr, regs, shm_sm, mem, major, minor, shm_blk = gpu._get_attrs()
        assert ( major, minor ) == ( 8, 0 )
        assert thr == 1536 and shm_sm == 102400            # never above what the card has
        assert nb_sm == 82 and mem == 24 << 30             # what is not a generation's matter is left alone
        assert gpu.compiler.arch == "sm_80"
    with _Patch( mask = "sm_75" ):
        assert gpu._get_attrs()[ 1 ] == 1024 and gpu._get_attrs()[ 3 ] == 65536   # lowered to Turing's
    with _Patch( mask = "sm_90" ):
        try:
            gpu.sm_arch
            raise AssertionError( "a card cannot behave as a stronger one" )
        except ValueError:
            pass
    with _Patch():
        assert gpu.sm_arch == "sm_89" and gpu._get_attrs() == real
        assert gpu.runnable_targets() == [ "sm_89", "sm_86", "sm_80", "sm_75", "sm_70" ]   # not sm_90


if test( "the_runs_are_every_device_present_and_every_target_it_can_run_whatever_the_mask" ):
    from loom.devices import hardware
    import importlib
    cuda_module = importlib.import_module( "loom.devices.CudaGpu" )   # the module, not the class of the same name
    from loom.compilation.Compiler import Nvcc
    real = ( 82, 1536, 65536, 102400, 24 << 30, 8, 6, 49152 )   # an sm_86
    with _x86( "x86-64-v3", mask = "x86-64-v2" ), _Patch( ( CudaGpu, "_real_attrs", lambda self: real ),
                                                          ( cuda_module, "cuda_device_count", lambda: 1 ),
                                                          ( Nvcc, "is_available", lambda self: True ),
                                                          ( C.HostCxx, "is_available", lambda self: True ),
                                                          mask = "x86-64-v2" ):
        runs = hardware.runs()
        assert [ ( r.device, r.target ) for r in runs ] == [
            ( "cpu", "x86-64-v3" ), ( "cpu", "x86-64-v2" ), ( "cpu", "x86-64" ),
            ( "cuda:0", "sm_86" ), ( "cuda:0", "sm_80" ), ( "cuda:0", "sm_75" ), ( "cuda:0", "sm_70" ) ]
        assert hardware.uncovered( runs, "cpu", X86 ) == [ "x86-64-v4" ]
        assert hardware.uncovered( runs, "cuda", list( C.CUDA_ARCHS ) ) == [ "sm_90", "sm_89" ]




# ── the catalogue is a directory ────────────────────────────────────────────────────────────────────
import importlib
import ctypes
from loom.compilation.build import Build


class _FakeLib:
    """What `ctypes.CDLL` returns for a file of these tests: it has the handler symbol, and says where it came from."""
    def __init__( self, path ):
        self._name = path
        self.sdot_ffi_entry = f"handler of { Path( path ).parent.name }/{ Path( path ).name }"


def _publish_env( d ):
    """Publishing without compiling: the runtime library of the build graph is a file of ours."""
    runtime = Path( d ) / "libloom_runtime_test.so"
    runtime.write_bytes( b"runtime" )
    return _Patch( ( Build, "runtime_library", lambda self: runtime ), ( ctypes, "CDLL", _FakeLib ) )


if test( "a_collected_kernel_lands_under_the_tag_of_the_architecture_the_device_behaves_as" ):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        lib = Path( d ) / "kernel.so"
        lib.write_bytes( b"binary" )
        k = catalogue.key( "int a;", [] )
        with _publish_env( d ), catalogue.collect( out ):
            for target in ( "x86-64-v4", "x86-64-v3" ):
                with _x86( "x86-64-v4", mask = target ):
                    where = catalogue.publish( lib, "int a;", [], Cpu(), "kernel_a", "sig" )
                    assert where == Path( out ).resolve() / f"cpu-{ target }" / ( k + catalogue.library_suffix() )
                    assert where.read_bytes() == b"binary"
                    assert ( where.parent / "libloom_runtime_test.so" ).read_bytes() == b"runtime"   # beside it
        assert sorted( catalogue.index( out ) ) == [ "cpu-x86-64-v3", "cpu-x86-64-v4" ]
        assert catalogue.index( out )[ "cpu-x86-64-v3" ][ k ][ "name" ] == "kernel_a"
    assert catalogue.publish( lib, "int a;", [], Cpu() ) is None      # outside `collect`: nothing


if test( "the_lookup_looks_in_one_tag_only_and_never_falls_back_on_a_poorer_architecture" ):
    with tempfile.TemporaryDirectory() as root, _publish_env( root ):
        saved = list( catalogue._roots ), dict( catalogue._loaded_libs )
        suffix = catalogue.library_suffix()
        k = catalogue.key( "s", [] )
        for tag in ( "cuda-sm_80", "cuda-sm_70" ):
            ( Path( root ) / tag ).mkdir()
            ( Path( root ) / tag / ( k + suffix ) ).write_bytes( b"" )
        try:
            catalogue._roots[ : ] = [ Path( root ) ]
            assert catalogue.lookup( "s", [], _cuda( 8, 0 ) )[ 1 ] == "handler of cuda-sm_80/" + k + suffix
            assert catalogue.lookup( "s", [], _cuda( 7, 0 ) )[ 1 ] == "handler of cuda-sm_70/" + k + suffix
            assert catalogue.lookup( "s", [], _cuda( 8, 6 ) ) is None       # no sm_86 build: sm_80 is another code
            assert catalogue.lookup( "s", [], _cuda( 7, 5 ) ) is None
            assert catalogue.lookup( "t", [], _cuda( 8, 0 ) ) is None
            os.environ[ "LOOM_KERNELS" ] = "compile"
            assert catalogue.lookup( "s", [], _cuda( 8, 0 ) ) is None
        finally:
            os.environ.pop( "LOOM_KERNELS", None )
            catalogue._roots[ : ] = saved[ 0 ]
            catalogue._loaded_libs.clear(); catalogue._loaded_libs.update( saved[ 1 ] )


if test( "the_catalogue_is_not_consulted_while_it_is_collected" ):
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out, _publish_env( root ):
        saved = list( catalogue._roots )
        k = catalogue.key( "s", [] )
        ( Path( root ) / "cuda-sm_70" ).mkdir()
        ( Path( root ) / "cuda-sm_70" / ( k + catalogue.library_suffix() ) ).write_bytes( b"" )
        try:
            catalogue._roots[ : ] = [ Path( root ) ]
            assert catalogue.lookup( "s", [], _cuda( 7, 0 ) ) is not None
            with catalogue.collect( out ):
                assert catalogue.lookup( "s", [], _cuda( 7, 0 ) ) is None     # it is compiled, then published
            assert catalogue.lookup( "s", [], _cuda( 7, 0 ) ) is not None
        finally:
            catalogue._roots[ : ] = saved


if test( "a_kernel_that_not_every_run_produced_depends_on_the_hardware" ):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        lib = Path( d ) / "kernel.so"
        lib.write_bytes( b"binary" )
        with _publish_env( d ), catalogue.collect( out ):
            for target in ( "x86-64-v4", "x86-64-v3", "x86-64-v2" ):
                with _x86( "x86-64-v4", mask = target ):
                    catalogue.publish( lib, "int a;", [], Cpu(), "same_everywhere", "sig" )
                    catalogue.publish( lib, "int b;", [], Cpu(), "same_everywhere", "sig" )   # several sources, on every run
            with _x86( "x86-64-v4", mask = "x86-64-v4" ):
                catalogue.publish( lib, "int c;", [], Cpu(), "wide_only", "sig" )              # produced by the richest run only
        dependent = catalogue.hardware_dependent( out )
        assert list( dependent ) == [ "wide_only" ]
        assert list( dependent[ "wide_only" ].values() ) == [ [ "cpu-x86-64-v4" ] ]
        text = catalogue.report( out )
        assert "cpu-x86-64-v4" in text and "`wide_only` depends on the hardware" in text


if test( "a_catalogue_made_by_one_run_has_nothing_that_depends_on_the_hardware" ):
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        lib = Path( d ) / "kernel.so"
        lib.write_bytes( b"binary" )
        with _publish_env( d ), catalogue.collect( out ), _x86( "x86-64-v4" ):
            catalogue.publish( lib, "int a;", [], Cpu(), "k", "s" )
            catalogue.publish( lib, "int b;", [], Cpu(), "k", "s" )
        assert catalogue.hardware_dependent( out ) == {}


# ── running as another hardware, in the same process ─────────────────────────────────────────────────
if test( "behave_as_sets_the_device_and_the_mask_for_the_block_and_restores_them" ):
    import loom
    from loom.devices import hardware
    before = ( os.environ.get( "LOOM_TARGET" ), loom.default_device )
    with _x86( "x86-64-v4" ):
        with hardware.behave_as( hardware.Run( "cpu", "x86-64-v3" ) ):
            assert os.environ[ "LOOM_TARGET" ] == "x86-64-v3" and loom.default_device == "cpu"
            assert Cpu().catalogue_target() == "x86-64-v3"
        assert Cpu().catalogue_target() == "x86-64-v4"
    assert ( os.environ.get( "LOOM_TARGET" ), loom.default_device ) == before


if test( "a_kernel_links_the_runtime_found_next_to_it" ):
    flags = C.HostCxx( "c++" ).link_libraries( [ Path( "/somewhere/libloom_runtime_x.so" ) ] )
    assert ( "@loader_path" if sys.platform == "darwin" else "$$ORIGIN" ) in flags      # `$$`: ninja's `$`
    assert "-rpath,/somewhere" in flags                                                  # and still where it was built
    assert flags.index( "$$ORIGIN" if sys.platform != "darwin" else "@loader_path" ) < flags.index( "-rpath,/somewhere" )   # first
    assert C.HostCxx( "c++" ).link_libraries( [] ) == ""


if test( "the_origin_rpath_survives_the_shell_ninja_runs_the_command_with" ):
    import subprocess
    libs = [ Path( "/somewhere/libloom_runtime_x.so" ) ]
    for compiler in ( C.HostCxx( "c++" ), C.Nvcc( arch = "sm_80" ) ):
        flags = compiler.link_libraries( libs ).replace( "$$", "$" )         # ninja turns `$$` into `$`
        seen = subprocess.run( f"printf '%s\\n' {flags}", shell = True, capture_output = True, text = True ).stdout.split()
        if sys.platform != "darwin":
            assert any( a.endswith( "rpath,$ORIGIN" ) for a in seen ), seen   # not expanded to nothing
