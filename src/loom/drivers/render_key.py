"""The KEY of a rendered kernel source: when two calls are certain to render the same text.

Rendering a call as C++ (`FfiSource._render_source`) costs ~0.4 ms, a third of the time of a small
call, and most calls repeat an earlier one: same code, same shapes of arguments, only the numbers in
the buffers differ. The text depends on the lowering nodes (`CallArg_*`), which SNAPSHOT what they
read off the Python objects when they are built -- so a function of those snapshots, and of the few
global things the rendering reads, is a key.

THE RULE THAT MAKES IT SAFE is that the key FAILS CLOSED. It is built by walking the fields of the
nodes and of the code, and a field whose value is not a plain, comparable fact (a number, a string, a
list or dict of such, an enum, a dtype, a layout, another node) makes the call UNCACHEABLE: it is
rendered as before. Nothing is ever guessed. The two ways to be wrong are therefore narrow, and
both are about what the rendering reads that is NOT a field of a node:

  * a new read of the Python object (`node.inst`) or of some global, in a `cpp_*` / `jax_*` method.
    Declare it in `_render_key_extra` of the node (or in `global_key` here). Every such read that
    exists today is listed there, and `LOOM_VERIFY_RENDER_CACHE=1` -- which renders anyway and
    compares -- is the check to run over the test suite after touching the lowering.
  * a field that the rendering ignores but that VARIES (the value of an attribute, the count of a
    ShapeVar: they travel as FFI attributes, not in the source). It does not make the key wrong, only
    useless -- one entry per value, a miss each time. Such a field is skipped by its node
    (`_render_key_skip`), with the reason written next to it.

`LOOM_RENDER_CACHE=0` turns the whole thing off.
"""
from pathlib import Path
import enum
import os

import numpy

from .. import env


class Uncacheable( Exception ):
    """Raised while building a key: something in the call is not a plain fact."""


SCALAR_TYPES = frozenset( ( type( None ), bool, int, str ) )


def plain( v ):
    """`v` as something hashable and comparable that says everything the rendering could read from it,
    or `Uncacheable`."""
    t = type( v )
    if v is None or t is bool or t is int or t is str:
        return v
    if t is float:
        return ( "f", repr( v ) )
    if t is tuple or t is list:
        return ( t.__name__, tuple( [ plain( x ) for x in v ] ) )
    if t is dict:
        # insertion order is kept: it decides the order of what is generated
        return ( "d", tuple( [ ( plain( k ), plain( x ) ) for k, x in v.items() ] ) )
    if isinstance( v, enum.Enum ):
        return ( "e", t.__qualname__, v.name )
    if isinstance( v, Path ):
        return ( "p", str( v ) )
    if isinstance( v, numpy.generic ):
        return plain( v.item() )
    if isinstance( v, numpy.ndarray ):
        return ( "a", v.dtype.str, v.shape, v.tobytes() )
    if isinstance( v, type ):
        return ( "t", v.__module__, v.__qualname__ )
    key = getattr( v, "render_key", None )
    if key is not None:
        return key()
    raise Uncacheable( f"{ t.__module__ }.{ t.__qualname__ }" )


def plain_object( obj, skip = () ):
    """An object that is nothing but its fields: its class, then each field."""
    return ( type( obj ).__qualname__,
             tuple( [ ( k, plain( v ) ) for k, v in vars( obj ).items() if k not in skip ] ) )


# what the rendering asks of a device: every one is spelled C++ (see `Device`). The key holds the
# ANSWERS, not the device -- which carries opaque things (a jax device, its attributes) that the
# source never reads.
_DEVICE_QUESTIONS = ( "cpp_queue_type", "cpp_memory_space", "cpp_queue_include", "cpp_stream_param", "cpp_stream_bind",
                      "cpp_queue_decl", "cpp_scratch_param", "cpp_scratch_bind", "cpp_scratch_decl" )


def device_key( device ) -> tuple:
    answers = []
    for question in _DEVICE_QUESTIONS:
        value = getattr( device, question )
        answers.append( plain( value() if callable( value ) else value ) )
    return ( type( device ).__qualname__, tuple( answers ) )


def global_key( device ) -> tuple:
    """What the rendering reads that belongs to no node: the environment (`LOOM_ZERO_OUTPUTS`, ...), the
    device, the real type of the call (`loom.resolved_dtype()`: the `TF` of a body), and the C++ roots, which
    decide whether an aggregate has a hand-written header (`manual_header`)."""
    from ..compilation import include_roots
    from . import framework_defaults
    raw = getattr( os.environ, "_data", None )
    return ( tuple( ( raw if raw is not None else os.environ ).items() ),
             device_key( device ),
             framework_defaults.ftype().cpp_name,
             tuple( str( r ) for r in include_roots() ) )


def render_key( code, ca, device ):
    """The key of `( code, ca, device )`, or `None` when the call cannot be keyed (see the module doc)."""
    try:
        return ( global_key( device ),
                 plain_object( code ),
                 tuple( ca.batch_axes ),
                 tuple( ca.declared_batch_size ),               # the NAMES: the sizes are FFI attributes
                 plain( ca.groups ),
                 plain( ca.args ),
                 ca.errors.render_key() )
    except Uncacheable as e:
        stats[ "uncacheable" ][ str( e ) ] = stats[ "uncacheable" ].get( str( e ), 0 ) + 1
        return None


# ── the cache ────────────────────────────────────────────────────────────────────────────────────────

_MAX_ENTRIES = 1024
_cache: dict = {}
stats = { "hit": 0, "miss": 0, "uncacheable": {} }


def enabled() -> bool:
    return env.var( "RENDER_CACHE", "1" ).strip().lower() not in ( "0", "false", "no", "off" )


def verifying() -> bool:
    """`LOOM_VERIFY_RENDER_CACHE`: on a hit, render anyway and fail if the two differ."""
    return env.flag( "VERIFY_RENDER_CACHE" )


def lookup( key ):
    return _cache.get( key )


def store( key, value ):
    if len( _cache ) >= _MAX_ENTRIES:
        _cache.pop( next( iter( _cache ) ) )       # the oldest: dicts keep insertion order
    _cache[ key ] = value


def _print_stats():
    import sys
    print( f"[render cache] pid { os.getpid() }: { stats[ 'hit' ] } hits, { stats[ 'miss' ] } misses, "
           f"uncacheable: { stats[ 'uncacheable' ] or 'none' }", file = sys.stderr, flush = True )


if env.flag( "RENDER_CACHE_STATS" ):
    import atexit
    atexit.register( _print_stats )
