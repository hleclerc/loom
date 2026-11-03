"""PROGRAMS PER RANK: one function, written once, that each rank of a group runs on its own block of the data, talking to the others
through a few collectives -- whatever the framework.

    ranks = loom.JaxRanks( mesh )                 # Jax: the devices of a mesh, one process ( `shard_map` )
    ranks = loom.TorchRanks()                     # Torch: the processes of `torch.distributed`, one rank each
    ranks = loom.MpiRanks()                       # numpy / CuPy: the processes of an MPI communicator, one rank each
    ranks = loom.LocalRanks()                     # one rank, any framework ( what the others reduce to with one device )

    x = loom.spread( positions, ranks )           # a tensor split along its first axis, one block per rank
    y = loom.spmd( program, ranks )( x, ... )     # `program( ranks, block, ... )` on every rank; `y` spread the same way

Inside a program the arrays are the rank's blocks, of the caller's framework, and the ranks talk through `ranks.rank()`,
`ranks.all_sum( x )`, `ranks.exchange( buf )` ( `buf[ j ]` goes to rank `j` ). `loom.route` packs rows by destination for an exchange
and `loom.scatter_add` adds values at indices: loom kernels, the same code for every framework and under a `jit`. A `loom.Capacity`
sizes what a program cannot know in advance ( rows per destination... ): when one overflows, the program runs again, whole, with that
capacity doubled -- and kept for the next calls -- as loom does with the capacities of its kernels.

Jax is ONE process driving every device: `spmd` traces the program in a `shard_map` under `jit`, the spread arguments being split
along the ranks' axis and the others replicated, and types what the kernels compute as varying along that axis. Torch is ONE process
PER RANK, and so is MPI: the program is called as it is, on the block this process holds.
"""
import numpy as np


# ---- the groups of ranks ------------------------------------------------------------------------------------------------------------

class Ranks:
    """a group of ranks: how many, and what a program running on them can ask"""
    size = 1
    # one process per rank ( Torch, MPI ): a spread value holds this rank's block only, so a reduction over it has to ask the others
    one_process_per_rank = False

    def rank( self ):
        """the rank running the program, as an int32 array `[ 1 ]` of its framework"""
        raise NotImplementedError

    def all_sum( self, x ):
        """`x` summed over the ranks ( every rank gets the sum )"""
        raise NotImplementedError

    def all_max( self, x ):
        raise NotImplementedError

    def all_min( self, x ):
        raise NotImplementedError

    def exchange( self, buf ):
        """`buf` ( `[ size, ... ]` ): `buf[ j ]` goes to rank `j`; returns what the ranks sent to this one, `[ size, ... ]` by sender"""
        raise NotImplementedError

    # ---- outside a program ( the host ) ----

    def run( self, program, args, spread ):
        """`program( *blocks )` on every rank: `args` the native values, `spread[ i ]` whether `args[ i ]` is split along the ranks.
        Returns the native outputs, each one the blocks of the ranks along its first axis"""
        raise NotImplementedError

    def from_host( self, array ):
        """a host array split along its first axis between the ranks ( its length a multiple of `size` ): the native value a
        spread tensor holds ( the whole array for Jax, this rank's block for Torch )"""
        raise NotImplementedError

    def to_host( self, value ):
        """the whole of a spread native value, on the host ( a collective for Torch: every rank must call it )"""
        raise NotImplementedError


class LocalRanks( Ranks ):
    """a single rank: the program is called as it is, collectives are the identity"""
    size = 1

    def rank( self ):
        return np.zeros( 1, dtype = np.int32 )

    def all_sum( self, x ):
        return x

    def exchange( self, buf ):
        return buf

    def all_max( self, x ):
        return x

    def all_min( self, x ):
        return x

    def run( self, program, args, spread ):
        return program( *args )

    def from_host( self, array ):
        return array

    def to_host( self, value ):
        from .tensor.host import to_host
        return np.asarray( to_host( value ) )


class JaxRanks( Ranks ):
    """the devices along axis `axis` of a Jax `mesh` ( its only one by default )"""

    def __init__( self, mesh, axis = None ):
        self.mesh = mesh
        self.axis = axis or mesh.axis_names[ 0 ]
        self.size = int( mesh.shape[ self.axis ] )

    def __eq__( self, other ):
        return isinstance( other, JaxRanks ) and other.mesh == self.mesh and other.axis == self.axis

    def __hash__( self ):
        return hash( ( id( self.mesh ), self.axis ) )

    def rank( self ):
        import jax
        import jax.numpy as jnp
        return jnp.reshape( jax.lax.axis_index( self.axis ).astype( jnp.int32 ), ( 1, ) )

    def all_sum( self, x ):
        import jax
        return jax.lax.psum( x, self.axis )

    def all_max( self, x ):
        import jax
        return jax.lax.pmax( x, self.axis )

    def all_min( self, x ):
        import jax
        return jax.lax.pmin( x, self.axis )

    def exchange( self, buf ):
        import jax
        return jax.lax.all_to_all( buf, self.axis, 0, 0, tiled = True )

    def _spec( self, ndim, spread ):
        from jax.sharding import PartitionSpec as P
        return P( self.axis, *( [ None ] * ( ndim - 1 ) ) ) if spread else P()

    def run( self, program, args, spread ):
        import jax
        from jax.sharding import PartitionSpec as P
        try:
            from jax import shard_map
        except ImportError:
            from jax.experimental.shard_map import shard_map
        from .drivers.JaxOperations import varying
        in_specs = tuple( self._spec( a.ndim, s ) for a, s in zip( args, spread ) )

        def body( *blocks ):
            # every output is the rank's block: varying along the axis, even a value the ranks agree on ( a sum )
            return tuple( varying( o ) for o in program( *blocks ) )

        # ( one spec for every output: split along its first axis )
        return jax.jit( shard_map( body, mesh = self.mesh, in_specs = in_specs, out_specs = P( self.axis ) ) )( *args )

    def from_host( self, array ):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding
        a = jnp.asarray( array )
        return jax.device_put( a, NamedSharding( self.mesh, self._spec( a.ndim, True ) ) )

    def to_host( self, value ):
        return np.asarray( value )


class TorchRanks( Ranks ):
    """the processes of a `torch.distributed` group ( the world by default ), one rank each"""
    one_process_per_rank = True

    def __init__( self, group = None ):
        import torch.distributed as dist
        self.group = group
        self.size = dist.get_world_size( group )
        self._rank = dist.get_rank( group )

    def rank( self ):
        import torch
        return torch.tensor( [ self._rank ], dtype = torch.int32 )

    def all_sum( self, x, op = "SUM" ):
        import torch.distributed as dist
        y = x.clone()
        dist.all_reduce( y, op = getattr( dist.ReduceOp, op ), group = self.group )
        return y

    def all_max( self, x ):
        return self.all_sum( x, "MAX" )

    def all_min( self, x ):
        return self.all_sum( x, "MIN" )

    def exchange( self, buf ):
        import torch
        import torch.distributed as dist
        buf = buf.contiguous()
        out = torch.empty_like( buf )
        try:
            dist.all_to_all_single( out, buf, group = self.group )
        except RuntimeError:
            # a backend without all-to-all ( gloo, on some builds ): every block to every rank, each keeps its own
            got = [ torch.empty_like( buf ) for _ in range( self.size ) ]
            dist.all_gather( got, buf, group = self.group )
            out = torch.stack( [ g[ self._rank ] for g in got ] )
        return out

    def run( self, program, args, spread ):
        return program( *args )

    def from_host( self, array ):
        import torch
        a = np.asarray( array )
        m = len( a ) // self.size
        return torch.as_tensor( a[ self._rank * m:( self._rank + 1 ) * m ].copy() )

    def to_host( self, value ):
        import torch
        import torch.distributed as dist
        got = [ torch.empty_like( value ) for _ in range( self.size ) ]
        dist.all_gather( got, value.contiguous(), group = self.group )
        return torch.cat( got ).cpu().numpy()


class MpiRanks( Ranks ):
    """the processes of an MPI communicator ( `mpi4py`, `COMM_WORLD` by default ), one rank each: numpy blocks, or CuPy ones with a
    CUDA-aware MPI ( the buffers go to MPI as they are, through their array interface )"""
    one_process_per_rank = True

    def __init__( self, comm = None ):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.size = self.comm.Get_size()
        self._rank = self.comm.Get_rank()

    @staticmethod
    def _xp( x ):
        """the array module of `x`: CuPy for a CuPy array, numpy otherwise"""
        if hasattr( x, "__cuda_array_interface__" ):
            import cupy
            return cupy
        return np

    def rank( self ):
        return np.array( [ self._rank ], dtype = np.int32 )

    def all_sum( self, x, op = "SUM" ):
        from mpi4py import MPI
        xp = self._xp( x )
        shape = xp.asarray( x ).shape                # ( `ascontiguousarray` makes a scalar a vector )
        x = xp.ascontiguousarray( xp.asarray( x ) )
        y = xp.empty_like( x )
        self.comm.Allreduce( x, y, op = getattr( MPI, op ) )
        return y.reshape( shape )

    def all_max( self, x ):
        return self.all_sum( x, "MAX" )

    def all_min( self, x ):
        return self.all_sum( x, "MIN" )

    def exchange( self, buf ):
        xp = self._xp( buf )
        buf = xp.ascontiguousarray( buf )
        out = xp.empty_like( buf )
        self.comm.Alltoall( buf, out )        # ( `buf[ j ]`, the same size for every `j`, to rank `j` )
        return out

    def run( self, program, args, spread ):
        return program( *args )

    def from_host( self, array ):
        a = np.asarray( array )
        m = len( a ) // self.size
        return a[ self._rank * m:( self._rank + 1 ) * m ].copy()

    def to_host( self, value ):
        v = value.get() if hasattr( value, "__cuda_array_interface__" ) else np.asarray( value )
        v = np.ascontiguousarray( v )
        out = np.empty( ( self.size, ) + v.shape, dtype = v.dtype )
        self.comm.Allgather( v, out )
        return out.reshape( ( self.size * v.shape[ 0 ], ) + v.shape[ 1: ] )


# ---- spread tensors --------------------------------------------------------------------------------------------------------------

class Spread:
    """what a tensor carries when it is split along its first axis between `ranks`"""

    def __init__( self, ranks, axis = 0 ):
        if axis != 0:
            raise NotImplementedError( "spread along the first axis only for now" )
        self.ranks = ranks
        self.axis = axis


class _OverRanks:
    """what a tensor spread between PROCESSES ( Torch, MPI: `Ranks.one_process_per_rank` ) does differently: a reduction along the
    spread axis gives the result over all the ranks -- every process gets it -- instead of over its block. Mixed in front of the
    tensor's class ( `_over_ranks_class` ), so that the other tensors do not even test for it. A Jax spread array needs nothing:
    it is the whole array already."""

    def _over_ranks( self, res, axis, collective ):
        if axis is not None:
            keys = axis if isinstance( axis, ( tuple, list ) ) else ( axis, )
            if 0 not in { self._axis_pos( k ) for k in keys }:
                return _set_spread( res, self.spread )      # along the other axes: still one block per rank
        return res._wrap_axes( getattr( self.spread.ranks, collective )( res.value ), res._dim_axes() )

    def sum( self, axis = None ):
        return self._over_ranks( super().sum( axis ), axis, "all_sum" )

    def max( self, axis = None ):
        return self._over_ranks( super().max( axis ), axis, "all_max" )

    def min( self, axis = None ):
        return self._over_ranks( super().min( axis ), axis, "all_min" )

    def any( self, axis = None ):
        return self._over_ranks( super().any( axis ), axis, "all_max" )

    def all( self, axis = None ):
        return self._over_ranks( super().all( axis ), axis, "all_min" )

    def prod( self, axis = None ):
        raise NotImplementedError( "the product of a tensor spread between processes" )

    def mean( self, axis = None ):
        return self.sum( axis ) / self._over_ranks( self._valid_counts( axis ), axis, "all_sum" )


_over_ranks_classes = {}


def _over_ranks_class( cls ):
    """`cls` with the reductions over the ranks in front ( made once per class )"""
    res = _over_ranks_classes.get( cls )
    if res is None:
        # ( `dtype_kinds = None`: not registered as THE class of its dtype, see `Tensor.__init_subclass__` )
        res = type( "Spread" + cls.__name__, ( _OverRanks, cls ), { "dtype_kinds": None, "__module__": __name__ } )
        _over_ranks_classes[ cls ] = res
    return res


def _set_spread( t, spread ):
    """`t` marked as split along its first axis between `spread.ranks`"""
    t.spread = spread
    if spread.ranks.one_process_per_rank and not isinstance( t, _OverRanks ):
        t.__class__ = _over_ranks_class( type( t ) )
    return t


def _as_tensor( value ):
    from .tensor.IntTensor import IntTensor
    from .tensor.RealTensor import RealTensor
    kind = str( getattr( value, "dtype", "" ) )             # ( "int32", "torch.int64"... )
    return IntTensor( value ) if "int" in kind else RealTensor( value )


def spread( value, ranks ):
    """`value` as a loom tensor split along its first axis between `ranks`: what THIS RANK holds -- its block for Torch and MPI ( one
    process per rank ), the whole array for Jax ( one process for all: a host array is then placed on the mesh, split )"""
    if isinstance( ranks, JaxRanks ) and isinstance( getattr( value, "value", value ), np.ndarray ):
        value = ranks.from_host( value )
    return _set_spread( _as_tensor( getattr( value, "value", value ) ), Spread( ranks ) )


def spread_host( array, ranks ):
    """a WHOLE host array split along its first axis between `ranks` ( every rank passes the same array, its length a multiple of
    `ranks.size` ): each one keeps its block"""
    return spread( ranks.from_host( array ), ranks )


def spread_of( x ):
    """the `Spread` of `x`, or `None`: a loom tensor made by `spread`, or a Jax array split along its first axis by a
    `NamedSharding` over several devices"""
    s = getattr( x, "spread", None )
    if isinstance( s, Spread ):
        return s
    import sys
    if "jax" in sys.modules:
        import jax
        from jax.sharding import NamedSharding
        from .drivers.framework_defaults import is_traced
        if isinstance( x, jax.Array ) and not is_traced( x ):
            sh = x.sharding
            if isinstance( sh, NamedSharding ) and len( sh.device_set ) > 1 and len( sh.spec ) and isinstance( sh.spec[ 0 ], str ):
                return Spread( JaxRanks( sh.mesh, sh.spec[ 0 ] ) )
    return None


def to_host( x ):
    """the whole of `x` on the host: a spread tensor gathered from its ranks ( a collective for Torch ), anything else as it is"""
    s = spread_of( x )
    v = getattr( x, "value", x )
    if s is not None:
        return s.ranks.to_host( v )
    from .tensor.host import to_host as th
    return np.asarray( th( v ) )


# ---- capacities ----------------------------------------------------------------------------------------------------------------

class Capacity:
    """a size a program cannot know in advance ( rows per destination of an exchange, vertices per row... ): `value` is what the
    program is traced with; an overflow reported during a run ( `report` ) makes `spmd` run the program again with it doubled"""

    def __init__( self, value ):
        self.value = int( value )

    def __int__( self ):
        return self.value

    def report( self, overflow ):
        """`overflow` ( a bool / int array of the program's framework ): this capacity was too small on this rank"""
        if not _reports:
            raise RuntimeError( "`Capacity.report` outside of a `loom.spmd` program" )
        _reports[ -1 ].append( ( self, overflow ) )


_reports = []


# ---- running a program -------------------------------------------------------------------------------------------------------

def spmd( program, ranks = None ):
    """`program( ranks, *blocks )` run on every rank of `ranks` ( by default, the ranks of the spread arguments ). The spread
    arguments ( `spread`, or a sharded Jax array ) arrive as the rank's block, the others whole; the outputs, the blocks of the
    ranks along their first axis, come back as spread tensors. Run again, whole, while a `Capacity` reports an overflow."""

    def call( *args ):
        from .drivers.framework_defaults import ops
        from .tensor.Dtype import Dtype
        from .tensor.host import to_host as th
        sp = [ spread_of( a ) for a in args ]
        rk = ranks or next( ( s.ranks for s in sp if s is not None ), LocalRanks() )
        values = [ getattr( a, "value", a ) for a in args ]
        flags = [ s is not None for s in sp ]
        while True:
            caps, single = [], []

            def body( *blocks ):
                _reports.append( [] )
                try:
                    outs = program( rk, *blocks )
                    reported = _reports[ -1 ]
                finally:
                    _reports.pop()
                single[ : ] = [ not isinstance( outs, ( tuple, list ) ) ]
                outs = ( outs, ) if single[ 0 ] else tuple( outs )
                caps[ : ] = [ c for c, _ in reported ]
                # the overflows, one per report, summed over the ranks ( all of them see the same )
                o = ops( outs[ 0 ] )
                i32 = Dtype.si( 32 )
                first = lambda x: o.reshape( x, ( -1, ) )[ 0:1 ]
                if reported:
                    f = rk.all_sum( o.concatenate( [ o.astype( first( r ), i32 ) for _, r in reported ] ) )
                else:
                    f = o.astype( first( outs[ 0 ] ) * 0, i32 )
                return outs + ( f, )

            # under the framework of the blocks: a numpy block is numpy's, even where Jax is the default ( MPI )
            from .drivers.framework_defaults import framework_name_of, using
            name = next( ( framework_name_of( v ) for v, s in zip( values, flags ) if s ), None )
            with using( name ) if name is not None else _nothing():
                *outs, f = rk.run( body, values, flags )
            over = np.asarray( th( f ) ).reshape( -1 )[ :len( caps ) ]
            if not over.any():
                break
            for c, v in zip( caps, over ):
                if v:
                    c.value *= 2
        res = tuple( _wrap( o, rk ) for o in outs )
        return res[ 0 ] if single[ 0 ] else res

    return call


class _nothing:
    def __enter__( self ):
        return self

    def __exit__( self, *exc ):
        return False


def _wrap( value, ranks ):
    t = _as_tensor( value )
    return _set_spread( t, Spread( ranks ) ) if ranks.size > 1 else t


# ---- the kernels of a program ------------------------------------------------------------------------------------------------

def route( dest, rows, nb_ranks, capacity, fill = -1 ):
    """`rows` ( `[ m, W ]` ) put in `[ nb_ranks, capacity, W ]` by destination `dest` ( `[ m ]`; a negative one keeps the row nowhere ),
    in the order of the rows, the empty slots set to `fill`. A destination given more rows than `capacity` holds reports an overflow
    to it ( a `Capacity` ). For `Ranks.exchange`."""
    from .calls import ffi_call, out
    from .compilation.FfiCode import FfiCode
    from .tensor.Axis import Axis
    from .tensor.IntTensor import IntTensor
    from .tensor.RealTensor import RealTensor
    from .tensor.ShapeVar import ShapeVar
    C = int( capacity )
    m, W = int( rows.shape[ 0 ] ), int( rows.shape[ 1 ] )
    num_row, num_w = Axis( ShapeVar( m ), name = "num_row" ), Axis( ShapeVar( W ), name = "row_width" )
    num_rank, num_slot = Axis( ShapeVar( int( nb_ranks ) ), name = "num_rank" ), Axis( ShapeVar( C ), name = "num_slot" )
    one = Axis( ShapeVar( 1 ), name = "one" )
    res = RealTensor[ num_rank, num_slot, num_w ]()
    ovf = IntTensor[ one ]()
    ffi_call( "loom_route", FfiCode.per_item(
        "const SI m = SI( inputs.rows.shape( 0 ) ), W = SI( inputs.rows.shape( 1 ) ), K = SI( outputs.res.shape( 0 ) ), "
        "C = SI( outputs.res.shape( 1 ) );\n"
        "for ( SI j = 0; j < K; ++j ) for ( SI s = 0; s < C; ++s ) for ( SI w = 0; w < W; ++w ) outputs.res( j, s, w ) = inputs.fill( 0 );\n"
        "SI counts[ 4096 ] = {}; int over = 0;\n"            # ( K <= 4096 ranks )
        "for ( SI i = 0; i < m; ++i ) {\n"
        "    const SI j = SI( inputs.dest( i ) );\n"
        "    if ( j < 0 || j >= K ) continue;\n"
        "    const SI s = counts[ j ]++;\n"
        "    if ( s >= C ) { over = 1; continue; }\n"
        "    for ( SI w = 0; w < W; ++w ) outputs.res( j, s, w ) = inputs.rows( i, w );\n"
        "}\n"
        "outputs.overflow( 0 ) = over;" ),
        dest = IntTensor[ num_row ]( dest ), rows = RealTensor[ num_row, num_w ]( rows ),
        fill = RealTensor[ one ]( _like_rows( rows, fill ) ), res = out( res ), overflow = out( ovf ) )
    if isinstance( capacity, Capacity ):
        capacity.report( ovf.value )
    return res.value


def _like_rows( rows, value ):
    from .drivers.framework_defaults import ops
    o = ops( rows )
    return o.reshape( rows, ( -1, ) )[ 0:1 ] * 0 + value


def scatter_add( target, idx, vals ):
    """`target` ( `[ n ]` ) plus `vals[ i ]` at `idx[ i ]` for each `i` ( an index outside `[ 0, n )` adds nothing )"""
    from .calls import ffi_call, out
    from .compilation.FfiCode import FfiCode
    from .tensor.Axis import Axis
    from .tensor.IntTensor import IntTensor
    from .tensor.RealTensor import RealTensor
    from .tensor.ShapeVar import ShapeVar
    n, m = int( target.shape[ 0 ] ), int( idx.shape[ 0 ] )
    num_t, num_v = Axis( ShapeVar( n ), name = "num_target" ), Axis( ShapeVar( m ), name = "num_val" )
    res = RealTensor[ num_t ]()
    ffi_call( "loom_scatter_add", FfiCode.per_item(
        "const SI n = SI( inputs.target.shape( 0 ) ), m = SI( inputs.idx.shape( 0 ) );\n"
        "for ( SI k = 0; k < n; ++k ) outputs.res( k ) = inputs.target( k );\n"
        "for ( SI i = 0; i < m; ++i ) { const SI k = SI( inputs.idx( i ) ); if ( k >= 0 && k < n ) outputs.res( k ) += inputs.vals( i ); }" ),
        target = RealTensor[ num_t ]( target ), idx = IntTensor[ num_v ]( idx ), vals = RealTensor[ num_v ]( vals ), res = out( res ) )
    return res.value
