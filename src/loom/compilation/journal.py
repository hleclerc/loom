"""HOW MANY kernels this process compiled, and WHY each one was new.

A kernel body does not yield one binary: it yields one PER GENERATED SOURCE, and the source
depends on the call -- the extents frozen in the type, the batch axes, and above all, in the backward pass,
which arguments are perturbed (a `NoneTensor` rather than a buffer) and which cotangents are
symbolic zeros (a `ZeroTensor`). This is intended: the body's `if constexpr` then drops
whole terms, and a buffer that does not exist is not allocated.

But it costs seconds of compilation, and NOTHING SAID SO. The only way to notice
that a two-derivative test had produced ten kernels was to count `ninja` lines in
a log. Hence this journal: the instrument comes before the knob -- you do not tune a specialization
you do not measure, and the right setting is surely not the same for geometry and for a
training loop.

    LOOM_JOURNAL=1        prints the report at the end of the process

    from loom.compilation.journal import report, stats
    print( report() )     by hand, whenever you want

The report groups by code NAME, and for a name that has several variants, it says what changes
from one variant to the first -- not "10 kernels", but "`grad_for_cell.data`: none -> out".
"""
import atexit
import os


# one entry per DISTINCT kernel (per target), in the order they appeared
_kernels = []
# how many times an already known target was served again -- the denominator that tells whether the cache works
_reuses = 0


def record( target, code_name, signature, outcome, seconds = 0.0 ):
    """One more distinct kernel. `outcome`: `"compiled"`, `"catalogue"`.

    `signature` is what describes the CALL (not the source): a readable dict, whose difference
    from another variant's is the answer to "why is this one new?"."""
    _kernels.append( dict( target = target, code_name = code_name or "?",
                           signature = dict( signature or {} ), outcome = outcome,
                           seconds = float( seconds ) ) )


def record_reuse():
    """A target already loaded, served again as is."""
    global _reuses
    _reuses += 1


def stats():
    """`( distinct kernels, compiled, taken from the catalogue, reuses, seconds )`."""
    return dict(
        kernels    = len( _kernels ),
        compiled   = sum( 1 for k in _kernels if k[ "outcome" ] == "compiled" ),
        catalogue  = sum( 1 for k in _kernels if k[ "outcome" ] == "catalogue" ),
        reuses     = _reuses,
        seconds    = sum( k[ "seconds" ] for k in _kernels ),
    )


def _differences( reference, other ):
    """The keys where two signatures differ, as `key: before -> after`."""
    res = []
    for key in sorted( set( reference ) | set( other ) ):
        before, after = reference.get( key, "-" ), other.get( key, "-" )
        if before != after:
            res.append( f"{ key } : { before } -> { after }" )
    return res


def report():
    """The report, as text. Empty if there was nothing to compile."""
    if not _kernels:
        return "loom: no kernel compiled."

    st = stats()
    lines = [ f"loom: { st[ 'kernels' ] } distinct kernel(s) "
              f"({ st[ 'compiled' ] } compiled in { st[ 'seconds' ]:.1f} s, "
              f"{ st[ 'catalogue' ] } from the catalogue), { st[ 'reuses' ] } reuse(s)." ]

    by_name = {}
    for k in _kernels:
        by_name.setdefault( k[ "code_name" ], [] ).append( k )

    for name, variants in sorted( by_name.items(), key = lambda kv: -len( kv[ 1 ] ) ):
        total = sum( v[ "seconds" ] for v in variants )
        lines.append( f"\n  { name } : { len( variants ) } variant(s), { total:.1f} s" )
        if len( variants ) == 1:
            continue
        # what separates each variant from the FIRST: the answer to "why new?"
        reference = variants[ 0 ][ "signature" ]
        for index, v in enumerate( variants[ 1 : ], start = 1 ):
            diffs = _differences( reference, v[ "signature" ] )
            lines.append( f"    [{ index }] " + ( "; ".join( diffs ) if diffs
                                                  else "nothing visible here (a source that differs "
                                                       "otherwise: body, includes, compiler)" ) )
    return "\n".join( lines )


def _print_at_exit():
    value = os.environ.get( "LOOM_JOURNAL", "" ).strip().lower()
    if value in ( "", "0", "false", "no", "off" ):
        return
    if _kernels:
        print( report() )


atexit.register( _print_at_exit )
