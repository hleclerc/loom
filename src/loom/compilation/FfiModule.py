"""A unit of C++ written by the program itself: declarations and definitions as text (f-strings, if need be), that
loom turns into a header and a source with stable names, compiles ONCE into a shared library, and lets the kernels
that name it include and link.

    engine = loom.FfiModule(
        "engine_2d_f32",
        header = f"LOOM_EXPORT void run( const Seeds &, float *out );",
        source = f"void run( const Seeds &seeds, float *out ) {{ ... { dim} ... }}",
        includes = [ "sdot/engine/Seeds.h" ],
    )
    loom.ffi_call( name, loom.FfiCode.inline( "engine_2d_f32::run( ... );", sources = [ engine ] ), ... )

The files are named after the module AND its content (`loom_modules/<name>_<hash>.h` / `.cpp`): the same text is the same
unit however many kernels (or processes) ask for it, and a different text never overwrites another's. The source
includes the header; the header starts with `loom/support/export.h` (`LOOM_EXPORT`), which an entry point wears since
kernels are compiled with hidden visibility. A kernel that names the module includes its header by itself.

`suffix`: `".cu"` for a unit with device code ( compiled by nvcc ); `.cpp` by default.

`needs`: units the module calls (see `FfiCode.sources`), linked into it.

`files` (`{ rel_path: text }`): headers the unit includes by those names, written in a directory of its own that comes first on
its include path -- the generated headers of a call, which a body compiled apart must see exactly as the call rendered them.
"""
import hashlib
import json

from . import build_dir
from .generated_headers import include_root, write_if_changed

_written = set()


class FfiModule:
    def __init__( self, name: str, header: str = "", source: str = "", includes = (), defines = None, files = None, needs = (), suffix: str = ".cpp" ):
        if not name or not all( c.isalnum() or c == "_" for c in name ):
            raise ValueError( f"FfiModule: `{ name }` is not usable as a file name (letters, digits, '_')" )
        self.name = name
        self.includes = tuple( includes )
        self.defines = tuple( sorted( ( defines or {} ).items() ) )
        self.files = dict( files or {} )
        # other units this one calls ( `( path, defines )`, as `FfiCode.sources` has them ): linked into it, so whoever links the module gets them
        self.needs = tuple( ( str( p ), tuple( sorted( dict( d ).items() ) ) ) for p, d in needs )
        self.header_text = ( "#pragma once\n#include <loom/support/export.h>\n"
                             + "".join( f'#include "{ i }"\n' for i in self.includes ) + header + "\n" )
        self.source_text = f'#include "{{header}}"\n{ source }\n'
        digest = hashlib.sha256( f"{ self.header_text }\0{ self.source_text }\0{ self.defines }\0{ sorted( self.files.items() ) }\0{ self.needs }\0{ suffix }".encode() ).hexdigest()[ :12 ]
        self.stem = f"{ name }_{ digest }"   # the digest includes `suffix`: the same text is another unit for another compiler
        self.header_path = f"loom_modules/{ self.stem }.h"
        self.source_path = f"loom_modules/{ self.stem }{ suffix }"

    def unit( self ):
        """`( path, defines )` as `FfiCode.sources` has them; the files exist when this returns."""
        self.write()
        return self.source_path, self.defines

    def write( self ):
        if self.stem in _written:
            return
        root = include_root()
        write_if_changed( root / self.header_path, self.header_text )
        write_if_changed( root / self.source_path, self.source_text.replace( "{header}", self.header_path ) )
        # the headers this unit compiles against, PRIVATE to it (first on its `-I`, see `Build.unit_library`): what a kernel was
        # rendered with, never the shared tree another call may have rewritten meanwhile
        for rel_path, content in self.files.items():
            write_if_changed( root / f"loom_modules/{ self.stem }.d" / rel_path, content )
        if self.needs:
            write_if_changed( root / f"loom_modules/{ self.stem }.needs.json", json.dumps( self.needs ) )
        _written.add( self.stem )
