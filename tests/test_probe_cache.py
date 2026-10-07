"""The toolchain probes ( OpenMP, macOS SDK ) are remembered on disk across processes -- each one
compiles a program, ~70 ms that every starting process used to pay again."""
from pathlib import Path
import tempfile
import os

from loom.compilation import Compiler as C
from errand import test


class _Cache:
    """A private cache directory for the duration of a test."""
    def __enter__( self ):
        self.saved = os.environ.get( "LOOM_CACHE_DIR" ), os.environ.get( "LOOM_PROBE_CACHE" )
        os.environ[ "LOOM_CACHE_DIR" ] = tempfile.mkdtemp( prefix = "loom-probes-" )
        os.environ.pop( "LOOM_PROBE_CACHE", None )
        return Path( os.environ[ "LOOM_CACHE_DIR" ] )

    def __exit__( self, *exc ):
        for name, value in zip( ( "LOOM_CACHE_DIR", "LOOM_PROBE_CACHE" ), self.saved ):
            if value is None:
                os.environ.pop( name, None )
            else:
                os.environ[ name ] = value


if test( "a_probe_is_computed_once_then_read_from_disk" ):
    with _Cache() as cache:
        calls = []
        def compute():
            calls.append( 1 )
            return [ "-fanswer" ]
        assert C._probe( "t", "c++", compute ) == [ "-fanswer" ]
        assert C._probe( "t", "c++", compute ) == [ "-fanswer" ]
        assert len( calls ) == 1 and ( cache / "probes.json" ).is_file()


if test( "another_question_is_another_probe" ):
    with _Cache():
        assert C._probe( "a", "c++", lambda: 1 ) == 1
        assert C._probe( "b", "c++", lambda: 2 ) == 2
        assert C._probe( "a", "c++", lambda: 3 ) == 1
        assert C._probe( "a", "c++", lambda: 3, extra = "other" ) == 3


if test( "the_cache_can_be_switched_off" ):
    with _Cache():
        C._probe( "t", "c++", lambda: 1 )
        os.environ[ "LOOM_PROBE_CACHE" ] = "0"
        assert C._probe( "t", "c++", lambda: 2 ) == 2


if test( "a_stale_answer_is_not_trusted" ):
    with _Cache():
        C._probe( "t", "c++", lambda: 1 )
        old = C._PROBE_MAX_AGE
        C._PROBE_MAX_AGE = -1
        try:
            assert C._probe( "t", "c++", lambda: 2 ) == 2
        finally:
            C._PROBE_MAX_AGE = old


if test( "the_signature_follows_the_environment" ):
    saved = os.environ.get( "LOOM_CXXFLAGS" )
    try:
        os.environ.pop( "LOOM_CXXFLAGS", None )
        a = C.HostCxx().build_signature
        assert C.HostCxx().build_signature == a
        os.environ[ "LOOM_CXXFLAGS" ] = "-DPROBE_TEST=1"
        assert C.HostCxx().build_signature != a
    finally:
        if saved is None:
            os.environ.pop( "LOOM_CXXFLAGS", None )
        else:
            os.environ[ "LOOM_CXXFLAGS" ] = saved
