"""`loom.jit` under Jax: a `jax.jit` that WATCHES the capacities of the `loom.ffi_call`s it traces, and runs again when one overflows.

Under a trace, a call cannot grow a capacity from Python: the count that did not fit only exists once the kernel has run. Outside
`loom.jit`, the check is a `jax.debug.callback` that raises at run time -- a host round trip in the middle of the graph, per call,
and an error instead of a rerun. Inside `loom.jit`, each call adds a `checkify.check` on its error buffer instead: `checkify`
carries it out of the compiled program as an output ( through `while_loop`, `scan`, `cond`, `vmap`, the user's own included ),
and the host reads it ONCE, after the run. On an overflow, the capacity that call site asked for is grown -- `max( wanted,
2 * capacity )`, as in eager -- and kept for the next calls, the function is traced again and run again, whole.

Running again is safe because a jitted Jax function is pure: same inputs, same program. Donated arguments are not ( a donated
buffer is gone after the first run ), hence refused.

A call site is `( call name, rank among the calls of that name in the trace )`: the same kernel called twice in one function
learns two capacities. Its message carries everything the host needs ( the site, the capacities it was traced with, the counts
asked for ), so nothing of the trace has to be kept.
"""
import functools
import re

import jax
from jax.experimental import checkify

from .CallArg_Errors import CAPACITY_OVERFLOW, FAILURE, KernelFailure, failure_message

_PREFIX = "loom-capacity"
_active = []    # the watches whose function is being traced, innermost last


class Watch:
    """what one `loom.jit` function learned: `( call name, rank, path ) -> capacity`"""

    def __init__( self ):
        self.learned = {}
        self.ranks = {}      # per trace: how many calls of each name were met so far
        self.version = 0     # bumped when a capacity grows: the program has to be traced again

    def site( self, name ):
        """the rank of this call among the calls named `name` in the trace being made"""
        rank = self.ranks.get( name, 0 )
        self.ranks[ name ] = rank + 1
        return rank


def active():
    """the watch of the `loom.jit` being traced, or None"""
    return _active[ -1 ] if _active else None


def apply_learned( watch, name, rank, output_capacities ):
    """the capacities of this call site, grown by what earlier runs learned ( `output_capacities` is updated in place )"""
    for ( n, r, path ), cap in watch.learned.items():
        if n == name and r == rank:
            output_capacities[ path ] = max( cap, output_capacities.get( path, 0 ) )


def add_checks( ca, name, rank, failures ):
    """the `checkify.check`s of one call: its overflows ( one value per capacity, the largest count asked for ) and its failures"""
    import jax.numpy as jnp
    raw = ca.errors.raw
    m = ca.errors.max_records
    rec = jnp.reshape( raw[ 1 : 1 + 3 * m ], ( m, 3 ) )
    valid = jnp.arange( m ) < raw[ 0 ]
    kind, ids, value = rec[ :, 0 ], rec[ :, 1 ], rec[ :, 2 ]

    evs = ca.error_vars
    if evs:
        over = valid & ( kind == CAPACITY_OVERFLOW )
        wanted = [ jnp.max( jnp.where( over & ( ids == i ), value, 0 ) ) for i in range( len( evs ) ) ]
        caps = ",".join( f"{ ev.path }={ int( ev.max_bound ) }" for ev in evs )
        fmt = " ".join( "{}" for _ in range( len( evs ) + 1 ) )
        # more records than the buffer holds: some were dropped, so the counts read may miss a capacity -- the host then
        # doubles every capacity of the site ( the first value of the message )
        dropped = ( raw[ 0 ] > m ).astype( jnp.int32 )
        checkify.check( ~jnp.any( over ), f"{ _PREFIX }|{ name }|{ rank }|{ caps }|{ fmt }", dropped, *wanted )

    # anything else the kernel recorded is not fixed by running again: raised with the call's message
    other = valid & ( kind != CAPACITY_OVERFLOW )
    first = jnp.argmax( other )
    checkify.check( ~jnp.any( other ), f"loom-failure|{ name }|{{}}|{{}}|{{}}", kind[ first ], ids[ first ], value[ first ] )
    _failures[ name ] = dict( failures or {} )


_failures = {}   # call name -> its `failures` messages, for the host side of a failure check


def _grow( watch, message ):
    """the capacities a `loom-capacity` message asks for, learned; False if the message is not one of ours"""
    parts = message.split( "|" )
    if len( parts ) != 5 or parts[ 0 ] != _PREFIX:
        return False
    _, name, rank, caps, wanted = parts
    dropped, *wanted = [ int( w ) for w in re.findall( r"-?\d+", wanted ) ]
    for ( path, cap ), w in zip( ( c.split( "=" ) for c in caps.split( "," ) ), wanted ):
        cap = int( cap )
        key = ( name, int( rank ), path )
        # the capacities that overflowed grow as in eager; the others keep theirs, unless records were dropped
        if w > cap or dropped:
            watch.learned[ key ] = max( watch.learned.get( key, 0 ), w, 2 * cap )
    return True


def _raise_failure( message ):
    parts = message.split( "|" )
    if len( parts ) != 5 or parts[ 0 ] != "loom-failure":
        return False
    _, name, kind, id, value = parts
    text = failure_message( name, [ ( int( kind ), int( id ), int( value ) ) ], _failures.get( name, {} ) )
    raise KernelFailure( text )


def _message_of( err ):
    """the text of the first failed check ( checkify appends "(`check` failed)" and where, after the message )"""
    text = err.get()
    if text is None:
        return None
    return re.split( r"\s*\(`check` failed", text, maxsplit = 1 )[ 0 ].strip()


def _has_tracer( tree ):
    return any( isinstance( x, jax.core.Tracer ) for x in jax.tree_util.tree_leaves( tree ) )


def jit( func, **kwargs ):
    """`func`, compiled by `jax.jit`, the capacities of its loom calls watched ( see the module )"""
    if kwargs.get( "donate_argnums" ) or kwargs.get( "donate_argnames" ):
        raise ValueError( "loom.jit cannot run again a function whose arguments are donated: they are gone after the first run" )

    watch = Watch()

    def traced( *args, **kw ):
        watch.ranks = {}
        _active.append( watch )
        try:
            return func( *args, **kw )
        finally:
            _active.pop()

    compiled = {}

    def program():
        if watch.version not in compiled:
            compiled.clear()
            compiled[ watch.version ] = jax.jit( checkify.checkify( traced ), **kwargs )
        return compiled[ watch.version ]

    plain = jax.jit( func, **kwargs )

    @functools.wraps( func )
    def run( *args, **kw ):
        # under an outer transform ( a `jax.grad` of the `loom.jit` function... ), the error is a tracer the host cannot read:
        # the calls fall back to their run-time callback
        if _has_tracer( ( args, kw ) ):
            return plain( *args, **kw )
        while True:
            err, out = program()( *args, **kw )
            message = _message_of( err )
            if message is None:
                return out
            if _grow( watch, message ):
                watch.version += 1
                continue
            if not _raise_failure( message ):
                err.throw()

    run.watch = watch
    return run
