from __future__ import annotations

import jax._src.core as jax_core
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import numpy

_cache: dict = {}

# module_name -> batch_rule callable registered by JaxDriver._register_vmap_rule()
_vmap_rules: dict = {}

def _make_ir_attr( v ):
    # Attribute widths must match the handler's Attr<...> decode type, i.e. the driver's
    # normalized int / float widths (e.g. SI32 / FP32 on Metal, SI64 / FP64 on CPU).
    from .driver import driver
    if isinstance( v, ( int, numpy.integer ) ):
        return ir.IntegerAttr.get( ir.IntegerType.get_signless( driver.itype.size ), int( v ) )
    if isinstance( v, ( float, numpy.floating ) ):
        float_type = { 16: ir.F16Type, 32: ir.F32Type, 64: ir.F64Type }[ driver.ftype.size ]
        return ir.FloatAttr.get( float_type.get(), float( v ) )
    raise NotImplementedError( f"Unsupported FFI attribute type: { type( v ) }" )

def get_or_create( module_name: str, output_specs, attributes: dict = {} ) -> jax_core.Primitive:
    """
    Returns (creating if needed) a JAX Primitive whose MLIR lowering emits
    a stablehlo.custom_call targeting the XLA FFI handler `module_name`.
    output_specs : list[ jax.ShapeDtypeStruct ]
    The cache key includes output shapes and attribute values so different
    capacities or attribute combos get distinct primitives.
    """
    cache_key = ( module_name, tuple( ( tuple( s.shape ), s.dtype ) for s in output_specs ), tuple( sorted( attributes.items() ) ) )
    if cache_key in _cache:
        return _cache[ cache_key ]

    prim = jax_core.Primitive( module_name )
    prim.multiple_results = True

    def abstract_eval( *avals, **_ ):
        return [ jax_core.ShapedArray( s.shape, s.dtype ) for s in output_specs ]

    prim.def_abstract_eval( abstract_eval )

    def lower( ctx, *mlir_args, **_ ):
        out_types = [ mlir.aval_to_ir_type( a ) for a in ctx.avals_out ]
        backend_config_dict = { k: _make_ir_attr( v ) for k, v in attributes.items() }
        op = hlo.CustomCallOp(
            out_types,
            list( mlir_args ),
            call_target_name = ir.StringAttr.get( module_name ),
            api_version      = ir.IntegerAttr.get( ir.IntegerType.get_signless( 32 ), 4 ),
            backend_config   = ir.DictAttr.get( backend_config_dict ),
        )
        return op.results

    mlir.register_lowering( prim, lower )
    _cache[ cache_key ] = prim

    if module_name in _vmap_rules:
        from jax.interpreters import batching
        batching.primitive_batchers[ prim ] = _vmap_rules[ module_name ]

    return prim
