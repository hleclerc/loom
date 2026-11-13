"""EXTERNAL libraries ( `compilation/externals.py` ): declared by a package, fetched only for the kernels that ask.

  * declaring one fetches nothing, and a kernel that does not ask never triggers the download nor sees the headers;
  * a kernel that asks ( `FfiCode( externals = [ ... ] )` ) gets them on its `-I` and on that of the units it links;
  * the check is made once per process: a library that cannot be had is not fetched again at every kernel;
  * a name nobody declared is an error, with the list of what is declared.

The library is a local archive served by a `file://` URL: no network.
"""
from pathlib import Path
import hashlib
import io
import os
import tarfile
import tempfile

import loom
from loom.testing import host
from loom.compilation import register_external
from loom.compilation import externals
from loom.compilation.FfiCode import FfiCode
from loom.tensor import Axis, IntTensor, ShapeVar
from errand import test

HERE = Path( __file__ ).resolve().parent / "cpp_sources"

# the archive: `fake-1/fake/value.h`, its top level stripped at unpacking
_tmp = Path( tempfile.mkdtemp( prefix = "loom-ext-test-" ) )
_archive = _tmp / "fake-1.tar.gz"
with tarfile.open( _archive, "w:gz" ) as t:
    data = b"#pragma once\n#define FAKE_VALUE 7\n"
    info = tarfile.TarInfo( "fake-1/fake/value.h" )
    info.size = len( data )
    t.addfile( info, io.BytesIO( data ) )
os.environ[ "LOOM_EXT_DIR" ] = str( _tmp / "ext" )
register_external( "fake", "1", _archive.as_uri(), hashlib.sha256( _archive.read_bytes() ).hexdigest() )
_fetched = _tmp / "ext" / "fake-1"


_CODE = """
    namespace {
        struct Setter {
            SI v;
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const { args.outputs.res( coords ) = v; }
        };
        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( Setter{ loom_test::external_value() }, args.outputs.res.domain(), args, batch_axes );
        }
    }
"""


def _value( name, asked ):
    res = IntTensor[ Axis( ShapeVar( 1 ), name = "num" ) ]()
    loom.ffi_call(
        name,
        FfiCode( code = _CODE, includes = [ str( HERE / "external_value.h" ) ],
                 sources = [ str( HERE / "external_value.cpp" ) ], externals = asked ),
        res = loom.out( res ),
    )
    return host( res.value ).reshape( -1 ).tolist()[ 0 ]


if test( "a_kernel_that_does_not_ask_never_fetches_nor_sees_it" ):
    assert _value( "test_externals_not_asked", [] ) == 0
    assert not _fetched.exists()


if test( "a_kernel_that_asks_gets_it_in_the_units_it_links" ):
    assert _value( "test_externals_asked", [ "fake" ] ) == 7
    assert ( _fetched / "fake" / "value.h" ).is_file()
    assert _value( "test_externals_not_asked_after", [] ) == 0          # the unit without it stays another unit


if test( "a_library_that_cannot_be_had_is_checked_once" ):
    register_external( "broken", "1", ( _tmp / "missing.tar.gz" ).as_uri(), "0" * 64 )
    calls = []
    real = externals._fetch
    externals._fetch = lambda ext: ( calls.append( ext.name ), real( ext ) )
    try:
        assert externals.include_dir_of( "broken" ) is None
        assert externals.include_dir_of( "broken" ) is None
    finally:
        externals._fetch = real
    assert calls == [ "broken" ], calls


if test( "an_undeclared_name_is_an_error" ):
    try:
        externals.externals_marker( [ "nobody_declared_this" ] )
    except ValueError as e:
        assert "fake" in str( e ), e
    else:
        raise AssertionError( "an undeclared external was accepted" )
