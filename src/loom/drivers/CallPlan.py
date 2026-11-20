"""A call REPLAYED: what the analysis of a call derived, kept and run again on the buffers of the next call of the same form.

Analysing a call -- lowering its arguments into `CallArg`s, rendering the source, finding the compiled handler -- costs one to
two milliseconds of Python, whatever the size of the data. A kernel called in a loop on buffers of the same shapes pays it at
every call for the same answer, and on a card that is often more than the kernel itself. So the pointer drivers ( `PointerFfi`:
Torch today ) key a call on its FORM -- the code, the shape, dtype, strides and device of each buffer, the axes of each output,
the settings of the call -- and the first call of a form keeps what the next ones need:

    the handler          the compiled entry, already loaded
    the inputs           which argument goes in which FFI slot, and whether the kernel reads its strides
    the outputs          the shape and dtype of each buffer to allocate, which argument receives it, and its layout
    the attributes       the integers of the frame ( extents: a function of the shapes, which are in the key )

and nothing of the call's values: the plan holds no buffer, no tensor of the caller's.

A call is only kept when its replay is EXACTLY what the analysis would do, which is checked on the analysis itself and not
assumed ( `_build` ): every argument is a buffer of the framework read in place, or an unbound output tensor written back in
place; no count can overflow ( no regrow ), nothing is scratch, nothing is batched or traced, and -- under Torch -- the code has
no backward ( an autograd node is the analysis's business; numpy and cupy never run one ). Anything else -- an aggregate, a numpy value converted, a mutable -- is analysed every
time, as before. `LOOM_CALL_PLAN=0` turns the replay off; `LOOM_VERIFY_CALL_PLAN=1` analyses every replayed call again and
compares what it renders with the plan ( a test of the key ).
"""
import os

from .. import env

# key -> `_Plan`, or `None` for a form that cannot be replayed ( so it is not examined again ). Bounded: a form that changes
# at every call ( a size that grows ) must not fill memory; the oldest entries go first.
_plans: dict = {}
_MAX_PLANS = 512

# what makes a raw scalar a value we can key on as it is
_SCALARS = ( bool, int, float )


def enabled():
    return env.flag( "CALL_PLAN", True )


def key_of( name, kernels, args, device, settings, adapter ):
    """The FORM of a call, hashable -- or `None` when an argument is not one a plan can describe."""
    from ..calls import Arg, _OUT
    from ..compilation.FfiCode import FfiCode
    from ..devices import hardware
    parts = [ name, type( device ).__name__, getattr( device, "device_id", None ), hardware.mask(),
              _env_snapshot() ]
    for k in kernels:
        if type( k ) is not FfiCode:
            return None
        parts.append( _code_key( k ) )
    parts.append( settings )
    ids = {}
    for key, value in args.items():
        if isinstance( value, Arg ):
            if value.kind != _OUT or value.members or value.excluded:
                return None
            sig = _output_key( value.value, ids )
            if sig is None:
                return None
            parts.append( ( key, "out", tuple( sorted( value.capacities.items() ) ), sig ) )
            continue
        sig = adapter.signature( value )
        if sig is None:
            return None
        parts.append( ( key, sig ) )
    try:
        res = tuple( parts )
        hash( res )
        return res
    except TypeError:
        return None


def find( name, kernels, args, device, nb_items, batch_alignment, has_dynamic_capacity, failures, adapter ):
    """`( key, plan )` of a call about to be made: `plan` a `_Plan` to replay, `None` ( a form that is not replayed, or no key:
    the call is analysed ) or `False` ( a form never seen: the call is analysed, then `record`ed under `key` )."""
    if not enabled():
        return None, None
    from . import framework_defaults
    settings = ( nb_items, batch_alignment, bool( has_dynamic_capacity ), tuple( sorted( ( failures or {} ).items() ) ),
                 str( framework_defaults.ftype() ), str( framework_defaults.itype() ) )
    key = key_of( name, kernels, args, device, settings, adapter )
    return key, ( None if key is None else lookup( key ) )


def replay( plan, data, device, adapter, code, analyse ):
    """Run `plan` on `data` ( `lower_args`'s ): `True` when the call is done, `False` when the error buffer says something only
    the analysis can act on ( the caller then takes the long way ). `analyse()`: the analysis of this call, made only to check
    the plan ( `LOOM_VERIFY_CALL_PLAN=1` )."""
    if not plan.run( data, device, adapter ):
        return False
    if plan.source is not None:
        verify( plan, code, analyse(), device )
    return True


def lookup( key ):
    """The plan of `key`: a `_Plan`, `None` ( a form known not to be replayable ), or `False` ( never seen )."""
    return _plans.get( key, False )


def record( key, code, ca, data, device, prefix, adapter, failures, forward_only = False ):
    """Keep what the call that has just run under `key` derived, if it can be replayed. `forward_only`: the driver never runs
    a backward ( numpy, cupy ), so a code that has one is replayed like any other."""
    plan = _build( code, ca, data, device, prefix, adapter, failures, forward_only )
    if len( _plans ) >= _MAX_PLANS:
        del _plans[ next( iter( _plans ) ) ]
    _plans[ key ] = plan


class _Plan:
    __slots__ = ( "entry", "in_paths", "strided", "out_specs", "out_targets", "attrs", "errors_ret", "checked", "call_name",
                  "failures", "source" )

    def run( self, data, device, adapter ):
        """The call, replayed on `data` ( path -> value, as `lower_args` gives it ): writes the outputs back, and returns
        `False` when the error buffer says something the analysis has to handle ( the caller then runs the call again, the
        long way ), `True` otherwise."""
        outs = launch( self.entry, [ data[ p ] for p in self.in_paths ], self.strided, self.out_specs, self.attrs, device, adapter )
        if self.checked:
            from ..tensor.host import to_host
            from .CallArg_Errors import KernelFailure, failure_message
            raw = to_host( outs[ self.errors_ret ] )
            nb = int( raw[ 0 ] )
            if nb:
                records = [ tuple( int( v ) for v in raw[ 1 + 3 * i : 4 + 3 * i ] ) for i in range( min( nb, ( len( raw ) - 1 ) // 3 ) ) ]
                message = failure_message( self.call_name, records, self.failures )
                if message is not None:
                    raise KernelFailure( message )
                return False
        for ( path, layout ), out in zip( self.out_targets, outs ):
            if path is not None:
                data[ path ].set_raw( out, layout = layout )
        return True


def launch( *args ):
    from .PointerFfi import launch
    return launch( *args )


def _build( code, ca, data, device, prefix, adapter, failures, forward_only ):
    """The plan of the call `ca` has just described, or `None` when replaying it would not be exactly what the analysis does."""
    from .CallArg_Errors import CallArg_Errors
    from .CallArg_Tensor import CallArg_Tensor
    from .PointerFfi import prepare
    if ( code.has_backward and not forward_only ) or ca.error_vars or ca.scratch_buffers or ca.batch_axes:
        return None
    entry, inputs, outputs, attrs = prepare( code, ca, device, prefix )
    plan = _Plan()
    plan.entry, plan.attrs = entry, attrs
    plan.in_paths, plan.strided = [], []
    for b in inputs:
        # read in place, as given: the analysis handed the kernel the very value the caller passed ( no conversion )
        if type( b ) is not CallArg_Tensor or b.path not in data or b.jax_input_array() is not data[ b.path ] or b.inst.is_fill:
            return None
        plan.in_paths.append( b.path )
        plan.strided.append( bool( getattr( b, "runtime_strides", False ) ) )
    plan.out_specs, plan.out_targets, plan.errors_ret = [], [], -1
    for i, b in enumerate( outputs ):
        if type( b ) is CallArg_Errors:
            plan.errors_ret = i
            plan.out_targets.append( ( None, None ) )
        elif type( b ) is CallArg_Tensor and data.get( b.path ) is b.inst:
            plan.out_targets.append( ( b.path, None if b.layout.is_identity else b.layout ) )
        else:
            return None
        plan.out_specs.append( b.out_shape_dtype() )
    plan.checked = plan.errors_ret >= 0 and ca.errors.checked
    plan.call_name, plan.failures = ca.errors.call_name, dict( failures or {} )
    plan.source = _render_source( code, ca, device ) if env.flag( "VERIFY_CALL_PLAN" ) else None
    return plan


def _render_source( code, ca, device ):
    from .FfiSource import _render_call
    source, _, _, attrs, _, _ = _render_call( code, ca, device )
    return source, tuple( int( v ) for _, _, v in attrs )


def verify( plan, code, ca, device ):
    """`LOOM_VERIFY_CALL_PLAN=1`: the call replayed under `plan` renders, analysed again, what the plan holds."""
    if plan.source is not None and _render_source( code, ca, device ) != plan.source:
        raise AssertionError( f"loom: the call plan of `{ code.name }` differs from the analysis of the call ( its key misses something )" )


# -- the key ---------------------------------------------------------------------------------------------------------------

def _env_snapshot():
    """The variables that steer the rendering and the compilation: a change of one of them is another form. Read off the
    encoded mapping when there is one: decoding every variable of the process was most of the cost of a key."""
    data = getattr( os.environ, "_data", None )
    if isinstance( data, dict ):
        prefixes = ( b"LOOM_", b"SDOT_" ) if data and isinstance( next( iter( data ) ), bytes ) else ( "LOOM_", "SDOT_" )
        return tuple( sorted( ( k, v ) for k, v in data.items() if k.startswith( prefixes ) ) )
    return tuple( sorted( ( k, v ) for k, v in os.environ.items() if k.startswith( ( "LOOM_", "SDOT_" ) ) ) )


_code_keys: dict = {}


def _code_key( k ):
    """An `FfiCode`, by its fields ( all of them: what it renders can read any )."""
    def frozen( v ):
        if isinstance( v, dict ):
            return tuple( sorted( ( a, frozen( b ) ) for a, b in v.items() ) )
        if isinstance( v, ( list, tuple ) ):
            return tuple( frozen( x ) for x in v )
        return v if isinstance( v, ( str, int, float, bool, type( None ) ) ) else str( v )
    return ( type( k ).__name__, frozen( vars( k ) ) )


def _output_key( t, ids ):
    """An unbound output tensor, by its class, dtype, device and axes -- identities ( an axis or a count two outputs share )
    numbered in order of appearance, and the extents' values."""
    from ..tensor.Axis import Axis
    from ..tensor.Tensor import Tensor
    from ..tensor.storage import Unbound
    if not isinstance( t, Tensor ) or type( t.storage ) is not Unbound or t.storage.buffer is not None:
        return None
    axes = []
    for axis in t.axes:
        if type( axis ) is not Axis or axis.step != 1 or axis.is_batch or axis.lo.coeffs:
            return None
        terms = []
        for sym, coeff in axis.hi.coeffs.items():
            terms.append( ( ids.setdefault( id( sym ), len( ids ) ), type( sym ).__name__, coeff ) )
        extent = axis.max
        if extent is None:
            return None
        axes.append( ( ids.setdefault( id( axis.axis_id ), len( ids ) ), axis.axis_id.name, bool( getattr( axis, "inferred", False ) ),
                       axis.lo.offset, axis.hi.offset, tuple( terms ), extent ) )
    return ( type( t ).__name__, str( t.dtype ), t._dtype_pinned, str( t.device ), getattr( t, "item_alignment_bytes", 0 ), tuple( axes ) )
