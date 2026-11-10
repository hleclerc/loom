"""PROGRAMS PER BOX: an algorithm written for ONE box of the data, which receives rows from the other boxes, sends rows to them and
adds values to outputs held by any box -- in rounds, until no row travels. Which process holds which box, and how the rows go from
one to the other, is the runtime's business, not the program's ( the Pregel model: box = vertex, round = superstep ).

    prog = loom.BoxProgram( start, step, row_width = 5, outputs = { "res": L } )
    out = prog( a, b, shared = ( w, ) )          # `a`, `b`: `[ K, ... ]`, one block per box; `out[ "res" ]`: `[ K, L ]`

    def start( box, a, b, w ):                   # round 0, on each box: its blocks `a[ k ]`, `b[ k ]` and the whole `w`
        return loom.Outbox( send = ( dest, rows ), add = { "res": ( owner_box, index, values ) } )

    def step( box, inbox, a, b, w ):             # the next rounds: `inbox.rows` `[ m, row_width ]`, `inbox.valid` `[ m ]`
        return loom.Outbox( ... )

`box.index` is the box running ( an int32 array `[ 1 ]` of the framework ), `box.nb_boxes` their count. A row sent to a negative box,
or an add aimed at one, goes nowhere. The adds of an output are summed in place ( a combiner: `scatter_add` at the owner ).

The runtime comes from the operands, as the framework does: the ranks of a spread argument ( `loom.spread` along the boxes, a sharded
Jax array ), else `ranks =`, else `loom.default_ranks`, else what the process was started in ( `detect_ranks`: `torch.distributed`, an
MPI launcher ), else one rank. With P ranks, each holds K / P consecutive boxes; one rank runs all the boxes in turn. A round is a
`loom.spmd` program: the boxes of the rank, one after the other, then one exchange of rows ( `loom.route` + `Ranks.exchange` ) and
one per output -- under a `jit` for Jax ( `shard_map` + `lax.while_loop` ).

What a program cannot know in advance is a `loom.Capacity`: `send_capacity` ( rows a rank sends to a box in a round ), `add_capacity`
( adds of an output a rank sends to a box in a round ), and any the program reports itself. An overflow runs the whole program again
with that capacity doubled, kept on the `BoxProgram` for the next calls.

An argument split along the boxes may also be a list, one value per box ( boxes of different sizes; not on Jax devices ). OUT OF
CORE: a `BoxStore` ( or a zarr / dask array, or `max_resident =` ) makes the rounds run from Python on one rank, the boxes with mail
loaded one by one, at most `max_resident` kept, the rows waiting on the host ( `_out_of_core` ).

THE ADJOINT ( Jax: a `custom_vjp`; Torch: an `autograd.Function` ): the forward keeps the inbox of every round ( a tape of
`tape_rounds` rounds, a `Capacity` ), the backward replays the rounds from the last one, each box again under the framework's `vjp`.
The cotangent of a row goes back to the box that sent it by the exchange in reverse; that of an add is asked for at the box holding
the output. Over Torch processes, each rank gets what its boxes contribute; over Jax devices, the gradient of the whole.

Limits for now: K a multiple of P, rows of reals ( a row index travels as a real: exact below 2^24 in fp32 ), adds of scalars,
synchronous rounds, out of core on one rank and forward only, no adjoint with numpy ( no autodiff ).
"""
import numpy as np

from .programs import Capacity, LocalRanks, detect_ranks, route, scatter_add, spmd, spread_host, spread_of, to_host


class Box:
    """the box a `start` / `step` runs on: `index` ( int32 `[ 1 ]`, of the program's framework ) and `nb_boxes`"""

    def __init__( self, index, nb_boxes ):
        self.index = index
        self.nb_boxes = nb_boxes


class Inbox:
    """what a box received in a round: `rows` `[ m, row_width ]`, of which only those where `valid` ( `[ m ]` bool ) holds were sent;
    the others are zeros"""

    def __init__( self, rows, valid ):
        self.rows = rows
        self.valid = valid


class Outbox:
    """what a box gives in a round: `send = ( dest, rows )` ( `[ m ]` box indices, `[ m, row_width ]` ), `add = { name: ( box, index,
    values ) }` ( `[ m ]` each: `values[ i ]` added to `outputs[ name ][ box[ i ], index[ i ] ]` ). A negative box drops the entry"""

    def __init__( self, send = None, add = None ):
        self.send = send
        self.add = dict( add or {} )


class BoxProgram:
    """`start( box, *blocks, *shared )` then `step( box, inbox, *blocks, *shared )` on every box, in rounds, until no row travels
    ( or `max_rounds` rounds ): see the module's doc. `outputs`: `{ name: L }`, each output `[ K, L ]` reals, zero then added to"""

    def __init__( self, start, step, *, row_width, outputs, send_capacity = 16, add_capacity = 64, max_rounds = None ):
        self.start = start
        self.step = step
        self.row_width = row_width
        self.outputs = dict( outputs )
        self.send_capacity = send_capacity if isinstance( send_capacity, Capacity ) else Capacity( send_capacity )
        self.add_capacity = add_capacity if isinstance( add_capacity, Capacity ) else Capacity( add_capacity )
        self.max_rounds = max_rounds
        self.tape_rounds = Capacity( 16 )   # rounds the tape of the adjoint holds
        self.last_rounds = None         # rounds of the last call ( the start included )

    def __call__( self, *blocks, shared = (), ranks = None, max_resident = None ):
        if not blocks:
            raise ValueError( "a BoxProgram needs at least one argument split along the boxes" )
        if max_resident is not None or any( _is_store( b ) for b in blocks ):
            return self._out_of_core( blocks, shared, ranks, max_resident )
        ranks = _ranks_for( blocks, ranks )
        args = _Args( blocks, shared )
        K = args.nb_boxes( ranks )
        if K % ranks.size:
            raise NotImplementedError( f"{ K } boxes on { ranks.size } ranks: the count of boxes must be a multiple of the count of ranks for now" )
        from .drivers.framework_defaults import framework_name_of
        floats = [ l for l in args.leaves if _is_float( l ) ]
        fw = next( ( framework_name_of( l ) for l in floats ), None )
        if fw == "jax":
            outs = self._jax_call( args, ranks, K )
        elif fw == "torch" and _torch_wants_grad( floats ):
            outs = _torch_call( self, args, ranks, K )
        else:
            outs = self._forward( args, ranks, K, args.leaves, False )[ 0 ]
        from .programs import _wrap
        return { n: _wrap( v, ranks ) for n, v in zip( self.outputs, outs ) }

    # ---- forward, backward: what the frameworks' autodiff calls ----

    def _forward( self, args, ranks, K, leaves, tape ):
        """the rounds, on the native `leaves` of the arguments: `( outputs, rounds, tape )` -- the outputs native, `rounds` and `tape`
        ( the inboxes of every round, kept for the adjoint when `tape` ) as `spmd` gives them back ( spread )"""
        blocks, shared = args.rebuild( leaves )
        local = args.local( blocks, ranks, K )
        names = list( self.outputs )
        nb = len( local )

        def program( rk, *a ):
            return self._per_rank( rk, K, names, a[ :nb ], a[ nb: ], tape )

        outs = spmd( program, ranks )( *local, *shared )
        if tape:
            *res, tp, rounds = outs
        else:
            ( *res, rounds ), tp = outs, None
        try:
            self.last_rounds = int( np.asarray( to_host( rounds ) ).reshape( -1 )[ 0 ] )
        except Exception:           # ( traced: known once run )
            self.last_rounds = None
        return [ getattr( r, "value", r ) for r in res ], rounds, tp

    def _backward( self, args, ranks, K, leaves, tape, rounds, cots ):
        """the gradients of the native `leaves` ( `None` where there is none ), from the cotangents `cots` of the outputs: the rounds
        replayed in reverse ( `_per_rank_back` )"""
        from .drivers.framework_defaults import ops
        from .programs import spread
        blocks, shared = args.rebuild( leaves )
        local = args.local( blocks, ranks, K )
        names = list( self.outputs )
        P = ranks.size
        o = ops( next( l for l in leaves if _is_float( l ) ) )
        mark = ( lambda v: spread( v, ranks ) ) if P > 1 else ( lambda v: v )
        rows_out = K if not ranks.one_process_per_rank else K // P
        cs = []
        for n, c in zip( names, cots ):
            if c is None:           # ( an output nothing depends on, for Torch )
                c = o.zeros( ( rows_out, self.outputs[ n ] ), _reals( o, local ) )
            cs.append( mark( getattr( c, "value", c ) ) )
        nb = len( local )
        nn = len( names )

        def program( rk, tp, rd, *a ):
            return self._per_rank_back( rk, K, names, tp, rd, a[ :nn ], a[ nn:nn + nb ], a[ nn + nb: ] )

        outs = spmd( program, ranks )( tape, rounds, *cs, *local, *shared )
        outs = outs if isinstance( outs, tuple ) else ( outs, )
        return args.gradients( [ getattr( g, "value", g ) for g in outs ], ranks, K, o )

    def _jax_call( self, args, ranks, K ):
        """the call as a `jax.custom_vjp`: the plain rounds when nothing is differentiated, the rounds and their inboxes otherwise"""
        import jax

        @jax.custom_vjp
        def f( *leaves ):
            return tuple( self._forward( args, ranks, K, list( leaves ), False )[ 0 ] )

        def fwd( *leaves ):
            outs, rounds, tape = self._forward( args, ranks, K, list( leaves ), True )
            return tuple( outs ), ( leaves, rounds.value, tape.value )

        def bwd( res, cots ):
            leaves, rounds, tape = res
            from .programs import spread
            mark = ( lambda v: spread( v, ranks ) ) if ranks.size > 1 else ( lambda v: v )
            return tuple( self._backward( args, ranks, K, list( leaves ), mark( tape ), mark( rounds ), list( cots ) ) )

        f.defvjp( fwd, bwd )
        return list( f( *args.leaves ) )

    # ---- out of core: the boxes loaded when they have mail, driven from Python ----

    def _out_of_core( self, blocks, shared, ranks, max_resident ):
        """the rounds driven from Python, outside any `jit`: a box is loaded ( `BoxStore.load`, then moved to the framework ) only
        when it has something to do, at most `max_resident` of them are kept loaded ( 2 by default: the one running, the next one,
        read ahead by a thread ), and what travels waits on the host. Rows and adds are kept exactly, so only the capacities the
        program reports itself can run it again"""
        import threading
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        from .drivers.framework_defaults import ops
        from .programs import _reports
        from .tensor.Dtype import Dtype
        from .tensor.host import to_host as th
        from .tensor.RealTensor import RealTensor
        if ranks is not None and ranks.size > 1:
            raise NotImplementedError( "out of core on several ranks: not yet ( one rank, the boxes in turn )" )
        from .drivers.framework_defaults import is_traced
        sh = _flatten( list( shared ) )[ 0 ]
        if any( is_traced( l ) or getattr( l, "requires_grad", False ) for l in sh ):
            raise NotImplementedError( "the adjoint out of core: not yet ( the tape would have to live out of core too )" )
        stores = [ b if isinstance( b, BoxStore ) else BoxStore.of( b ) for b in blocks ]
        K = stores[ 0 ].nb_boxes
        if any( st.nb_boxes != K for st in stores ):
            raise ValueError( "the arguments split along the boxes do not have the same count of boxes" )
        M = max( 1, int( max_resident or 2 ) )
        o = ops( None )
        host = lambda x: np.asarray( th( getattr( x, "value", x ) ) )
        native = lambda v: _map_arrays( v, lambda a: o.array( a, Dtype.from_numpy( a.dtype ) ) )
        shared = tuple( _map_arrays( v, lambda a: o.array( a, Dtype.from_numpy( a.dtype ) ) ) if isinstance( v, np.ndarray ) else v
                        for v in shared )
        names = list( self.outputs )
        i32 = Dtype.si( 32 )
        lock = threading.Lock()
        self.last_loads = 0

        def read( b ):
            """box `b` from its stores, on the host ( the part a thread can do ahead )"""
            return [ _map_arrays( st.load( b ), np.asarray ) for st in stores ]

        while True:
            caps = []
            resident = OrderedDict()
            pending = {}
            pool = ThreadPoolExecutor( 1 )

            def get( b ):
                if b in resident:
                    resident.move_to_end( b )
                    return resident[ b ]
                fut = pending.pop( b, None )
                vals = fut.result() if fut is not None else read( b )
                with lock:
                    self.last_loads += 1
                vals = [ native( v ) for v in vals ]
                resident[ b ] = vals
                while len( resident ) > M:
                    resident.popitem( last = False )
                return vals

            def ahead( b ):
                if b is not None and b not in resident and b not in pending and len( resident ) + len( pending ) < M + 1:
                    pending[ b ] = pool.submit( read, b )

            outs = { n: np.zeros( ( K, self.outputs[ n ] ) ) for n in names }
            W = int( self.row_width() if callable( self.row_width ) else self.row_width )
            mail = { b: None for b in range( K ) }          # round 0: every box, with no inbox
            rounds = 0
            try:
                while mail and ( self.max_rounds is None or rounds < self.max_rounds ):
                    rounds += 1
                    nxt = {}
                    order = sorted( mail )
                    for j, b in enumerate( order ):
                        ahead( order[ j + 1 ] if j + 1 < len( order ) else None )
                        data = get( b )
                        box = Box( o.array( np.array( [ b ], dtype = np.int32 ), i32 ), K )
                        _reports.append( [] )
                        try:
                            if mail[ b ] is None:
                                ob = self.start( box, *data, *shared )
                            else:
                                rows = mail[ b ]
                                ob = self.step( box, Inbox( o.array( rows, Dtype.from_numpy( rows.dtype ) ),
                                                            o.array( np.ones( len( rows ), dtype = bool ), Dtype( "bool" ) ) ), *data, *shared )
                            reported = _reports[ -1 ]
                        finally:
                            _reports.pop()
                        for c, r in reported:
                            if host( r ).reshape( -1 )[ :1 ].any():
                                caps.append( c )
                        if ob.send is not None:
                            d = host( ob.send[ 0 ] ).reshape( -1 ).astype( np.int64 )
                            r = host( ob.send[ 1 ] ).reshape( len( d ), W )
                            for k in np.unique( d[ ( d >= 0 ) & ( d < K ) ] ):
                                nxt.setdefault( int( k ), [] ).append( r[ d == k ] )
                        for n, ( bx, idx, val ) in ob.add.items():
                            bx, idx = host( bx ).reshape( -1 ).astype( np.int64 ), host( idx ).reshape( -1 ).astype( np.int64 )
                            keep = ( bx >= 0 ) & ( bx < K ) & ( idx >= 0 ) & ( idx < self.outputs[ n ] )
                            np.add.at( outs[ n ], ( bx[ keep ], idx[ keep ] ), host( val ).reshape( -1 )[ keep ] )
                        if caps:
                            break
                    if caps:
                        break
                    mail = { b: np.concatenate( v ) for b, v in nxt.items() }
            finally:
                pool.shutdown( wait = True )
            if not caps:
                break
            for c in { id( c ): c for c in caps }.values():
                c.value *= 2
        self.last_rounds = rounds
        return { n: RealTensor( o.array( outs[ n ], Dtype.fp( 64 ) ) ) for n in names }

    # ---- what each rank runs ( a `loom.spmd` program ) ----

    def _per_rank( self, ranks, K, names, blocks, shared, tape = False ):
        """the rounds on the boxes of this rank: the blocks of the outputs `[ B, L ]`, ( with `tape`, the inbox of every round,
        `[ B, R, P * S, 1 + W ]` ), and the count of rounds"""
        from .drivers.framework_defaults import ops, while_loop
        from .tensor.Dtype import Dtype
        k = _Kit( self, ranks, K, names, blocks )
        o, i32, B, W = k.o, k.i32, k.B, k.W

        def run_boxes( inbox ):
            if inbox is None:
                outboxes = [ self.start( k.boxes[ b ], *k.mine( b ), *shared ) for b in range( B ) ]
            else:
                outboxes = [ self.step( k.boxes[ b ], Inbox( inbox[ b, :, 1: ], inbox[ b, :, 0 ] > 0.5 ), *k.mine( b ), *shared )
                             for b in range( B ) ]
            return k.gather( [ _normalized( ob, o, k.ftype, W, i32 ) for ob in outboxes ] )

        dest, rows, adds = run_boxes( None )
        res = k.settle( tuple( o.zeros( ( B * self.outputs[ n ], ), k.ftype ) for n in names ), adds )
        rounds = o.zeros( ( 1, ), i32 ) + 1
        R = self.tape_rounds.value
        T = o.zeros( ( R, B, k.P * k.S, 1 + W ), k.ftype ) if tape else None

        def loop( c ):
            res, inbox, _, rounds, T = c
            if T is not None:
                # the inbox of this round, for the adjoint ( `rounds` is its index )
                self.tape_rounds.report( rounds >= R )
                T = o.put_row( T, o.where( rounds < R, rounds, rounds * 0 + R - 1 ), inbox )
            dest, rows, adds = run_boxes( inbox )
            return k.settle( res, adds ), k.send( dest, rows ), k.moving( dest ), rounds + 1, T

        def going_on( c ):
            more = c[ 2 ] > 0
            if self.max_rounds is not None:
                more = more & ( c[ 3 ][ 0 ] < self.max_rounds )
            return more

        res, _, _, rounds, T = while_loop( going_on, loop, ( res, k.send( dest, rows ), k.moving( dest ), rounds, T ) )
        outs = tuple( o.reshape( r, ( B, self.outputs[ n ] ) ) for r, n in zip( res, names ) )
        if tape:
            outs += ( o.transpose( T, ( 1, 0, 2, 3 ) ), )
        return outs + ( rounds, )

    # ---- the adjoint: the rounds in reverse --------------------------------------------------------------------------------------

    def _per_rank_back( self, ranks, K, names, tape, rounds, cots, blocks, shared ):
        """the gradients of this rank's float leaves ( see `_Args.gradients` for their order ), from the cotangents `cots` of its
        blocks of the outputs. The rounds are replayed from the last one: each box is run again from the inbox the tape kept, under
        the framework's `vjp`; the cotangent of a row it sent comes back from the box that received it ( the exchange in reverse ),
        that of an add from the box that holds the output ( asked for there, like a row ). What a box gets for its inbox goes back the
        same way to the boxes that sent it, one round earlier"""
        from .drivers.framework_defaults import framework_name_of, while_loop
        from .programs import _reports
        k = _Kit( self, ranks, K, names, blocks )
        o, i32, B, W, P, S, A, ft = k.o, k.i32, k.B, k.W, k.P, k.S, k.A, k.ftype
        fw = framework_name_of( tape )
        Ls = [ self.outputs[ n ] for n in names ]
        G = [ o.reshape( c, ( -1, ) ) for c in cots ]
        T = o.transpose( tape, ( 1, 0, 2, 3 ) )           # [ R, B, P * S, 1 + W ]

        # the leaves of each box, of the shared values: the float ones are differentiated
        per_box = [ _flatten( k.mine( b ) ) for b in range( B ) ]
        sh_leaves, sh_rebuild = _flatten( list( shared ) )
        d_box = [ [ i for i, l in enumerate( lv ) if _is_float( l ) ] for lv, _ in per_box ]
        d_sh = [ i for i, l in enumerate( sh_leaves ) if _is_float( l ) ]

        def back_by_box( g, dest, C ):
            """the reverse of `by_box`: `g` `[ B, P * C, w ]`, the cotangents of what the boxes of this rank received, back to
            the rows this rank sent to `dest` ( `[ m ]` ): `[ m, w ]`"""
            w, m = int( g.shape[ 2 ] ), int( dest.shape[ 0 ] )
            back = o.reshape( o.transpose( o.reshape( g, ( B, P, C, w ) ), ( 1, 0, 2, 3 ) ), ( P, B * C, w ) )
            got = o.reshape( ranks.exchange( back ), ( K * C, w ) )
            # where `route` put each row: its index travels as `by_box` moved it
            src = route( dest, o.reshape( o.astype( o.arange( m, i32 ), ft ), ( m, 1 ) ), K, C, fill = -1 )
            src = o.astype( o.reshape( src, ( K * C, 1 ) ), i32 )
            at = o.where( src < 0, src * 0 - 1, src * w + o.reshape( o.arange( w, i32 ), ( 1, w ) ) )
            flat = scatter_add( o.zeros( ( m * w, ), ft ), o.reshape( at, ( -1, ) ), o.reshape( got, ( -1, ) ) )
            return o.reshape( flat, ( m, w ) )

        def fetch( i, bx, idx ):
            """`G[ i ][ bx, idx ]` ( `[ m ]` ), asked for at the rank holding box `bx`; 0 for an add that went nowhere"""
            m = int( bx.shape[ 0 ] )
            req = k.by_box( bx, o.reshape( o.astype( idx, ft ), ( m, 1 ) ), A, -1 )       # [ B, P * A, 1 ], at the owners
            at = o.astype( req[ :, :, 0 ], i32 )
            ok = ( at >= 0 ) & ( at < Ls[ i ] )
            flat = o.where( ok, at + o.reshape( o.arange( B, i32 ) * Ls[ i ], ( B, 1 ) ), at * 0 )
            v = G[ i ][ flat ]
            v = o.where( ok, v, v * 0 )
            return o.reshape( back_by_box( o.reshape( v, ( B, P * A, 1 ) ), bx, A ), ( -1, ) )

        def box_fn( b, valid ):
            """box `b` as a function of its float leaves ( after the rows of its inbox, unless `valid` is `None`: the start ):
            the reals it gives ( rows sent, values added ), and what says where they go"""
            leaves, rebuild = per_box[ b ]

            def f( *prim ):
                it = iter( prim )
                rows_in = next( it ) if valid is not None else None
                lb = list( leaves )
                for i in d_box[ b ]:
                    lb[ i ] = next( it )
                ls = list( sh_leaves )
                for i in d_sh:
                    ls[ i ] = next( it )
                _reports.append( [] )           # ( the capacities were settled by the forward )
                try:
                    if rows_in is None:
                        ob = self.start( k.boxes[ b ], *rebuild( lb ), *sh_rebuild( ls ) )
                    else:
                        ob = self.step( k.boxes[ b ], Inbox( rows_in, valid ), *rebuild( lb ), *sh_rebuild( ls ) )
                finally:
                    _reports.pop()
                ob = _normalized( ob, o, ft, W, i32 )
                reals = ( ( ob.send[ 1 ], ) if ob.send is not None else () ) + tuple( ob.add[ n ][ 2 ] for n in names if n in ob.add )
                where = ( ob.send[ 0 ] if ob.send is not None else None, { n: ob.add[ n ][ :2 ] for n in names if n in ob.add } )
                return reals, where

            return f

        def round_back( inbox, g_in, acc ):
            """one round in reverse: `inbox` the one the tape kept ( `None`: the start ), `g_in` the cotangents of what its boxes
            sent, as their receivers hold them. Returns the cotangents of `inbox`, and `acc` plus this round's gradients"""
            runs = []
            for b in range( B ):
                prim = [ inbox[ b, :, 1: ] ] if inbox is not None else []
                prim += [ per_box[ b ][ 0 ][ i ] for i in d_box[ b ] ] + [ sh_leaves[ i ] for i in d_sh ]
                runs.append( _vjp_aux( fw, box_fn( b, inbox[ b, :, 0 ] > 0.5 if inbox is not None else None ), prim ) )
            # the cotangents of the rows sent and of the values added, from where they went ( every rank exchanges, as forward )
            dests = [ w[ 0 ] for _, w, _ in runs if w[ 0 ] is not None ] or [ o.full( ( 1, ), -1, i32 ) ]
            g_send = back_by_box( g_in, o.concatenate( dests ), S )
            g_add = []
            for i, n in enumerate( names ):
                parts = [ w[ 1 ][ n ] for _, w, _ in runs if n in w[ 1 ] ] or [ ( o.full( ( 1, ), -1, i32 ), o.zeros( ( 1, ), i32 ) ) ]
                g_add.append( fetch( i, o.concatenate( [ p[ 0 ] for p in parts ] ), o.concatenate( [ p[ 1 ] for p in parts ] ) ) )
            acc = list( acc )
            g_rows, at_s, at_a, j = [], 0, [ 0 ] * len( names ), 0
            for b, ( _, w, pull ) in enumerate( runs ):
                ct = []
                if w[ 0 ] is not None:
                    m = int( w[ 0 ].shape[ 0 ] )
                    ct.append( g_send[ at_s:at_s + m ] )
                    at_s += m
                for i, n in enumerate( names ):
                    if n in w[ 1 ]:
                        m = int( w[ 1 ][ n ][ 0 ].shape[ 0 ] )
                        ct.append( g_add[ i ][ at_a[ i ]:at_a[ i ] + m ] )
                        at_a[ i ] += m
                g = list( pull( tuple( ct ) ) )
                if inbox is not None:
                    g_rows.append( g.pop( 0 ) )
                for x in g[ :len( d_box[ b ] ) ]:
                    acc[ j ] = acc[ j ] + x
                    j += 1
                for t, x in enumerate( g[ len( d_box[ b ] ): ] ):
                    acc[ len( acc ) - len( d_sh ) + t ] = acc[ len( acc ) - len( d_sh ) + t ] + x
            return ( o.stack( g_rows, axis = 0 ) if g_rows else None ), tuple( acc )

        zero = lambda x: o.zeros( tuple( x.shape ), o.dtype_of( x ) )
        acc = tuple( zero( per_box[ b ][ 0 ][ i ] ) for b in range( B ) for i in d_box[ b ] ) + tuple( zero( sh_leaves[ i ] ) for i in d_sh )

        def body( c ):
            r, g_in, acc = c
            g_in, acc = round_back( o.take_row( T, r ), g_in, acc )
            return r - 1, g_in, acc

        _, g_in, acc = while_loop( lambda c: c[ 0 ][ 0 ] >= 1, body,
                                   ( o.astype( rounds, i32 ) - 1, o.zeros( ( B, P * S, W ), ft ), acc ) )
        _, acc = round_back( None, g_in, acc )

        # out: per block, the float gradients of this rank's boxes ( `_Args.gradients` puts them back where the leaves were )
        res, j = [], 0
        per = [ [ acc[ sum( len( d ) for d in d_box[ :b ] ) + t ] for t in range( len( d_box[ b ] ) ) ] for b in range( B ) ]
        for jb, blk in enumerate( blocks ):
            if isinstance( blk, ( list, tuple ) ):
                for b in range( B ):
                    lv, _ = _flatten( blk[ b ] )
                    for l in lv:
                        if _is_float( l ):
                            res.append( o.reshape( per[ b ].pop( 0 ), ( 1, ) + tuple( l.shape ) ) )
            elif _is_float( blk ):
                res.append( o.stack( [ per[ b ].pop( 0 ) for b in range( B ) ], axis = 0 ) )
        res += [ o.reshape( acc[ len( acc ) - len( d_sh ) + t ], ( 1, ) + tuple( sh_leaves[ i ].shape ) ) for t, i in enumerate( d_sh ) ]
        return tuple( res ) if res else ( o.zeros( ( 1, ), ft ), )


# ---- what the forward and the adjoint share, on one rank ---------------------------------------------------------------------------

class _Kit:
    """the boxes of a rank and the moves between them: `by_box` ( rows to the boxes that receive them ), `gather` ( the outboxes of
    the rank, end to end ), `settle` ( adds to the outputs ), `send` ( rows to the inboxes ), `moving` ( rows still travelling )"""

    def __init__( self, prog, ranks, K, names, blocks ):
        from .drivers.framework_defaults import ops
        from .tensor.Dtype import Dtype
        arrays = [ a for a in map( _first_array, blocks ) if a is not None ]
        o = self.o = ops( arrays[ 0 ] if arrays else None )
        i32 = self.i32 = Dtype.si( 32 )
        self.ftype = _reals( o, blocks )
        self.prog, self.ranks, self.K, self.names, self.blocks = prog, ranks, K, names, blocks
        P = self.P = ranks.size
        B = self.B = K // P
        self.W = int( prog.row_width() if callable( prog.row_width ) else prog.row_width )
        self.S, self.A = prog.send_capacity.value, prog.add_capacity.value
        first = o.astype( ranks.rank(), i32 ) * B if P > 1 else o.zeros( ( 1, ), i32 )
        self.boxes = [ Box( first + b, K ) for b in range( B ) ]

    def mine( self, b ):
        return [ x[ b ] for x in self.blocks ]

    def by_box( self, dest, rows, capacity, fill ):
        """`rows` `[ m, w ]` to the boxes `dest`: what the boxes of this rank receive, `[ B, P * C, w ]` ( `capacity` a `Capacity`,
        which reports its overflows, or an int )"""
        o, P, B = self.o, self.P, self.B
        C, w = int( capacity ), int( rows.shape[ 1 ] )
        buf = o.reshape( route( dest, rows, self.K, capacity, fill = fill ), ( P, B * C, w ) )
        got = o.reshape( self.ranks.exchange( buf ), ( P, B, C, w ) )
        return o.reshape( o.transpose( got, ( 1, 0, 2, 3 ) ), ( B, P * C, w ) )

    def gather( self, outboxes ):
        """the ( normalized ) outboxes of the boxes of this rank, end to end: `( dest, rows )`, `{ name: ( box, index, values ) }` --
        every output there, so that every rank makes the same exchanges"""
        o, i32, W, ft = self.o, self.i32, self.W, self.ftype
        sends = [ ob.send for ob in outboxes if ob.send is not None ]
        if not sends:
            # ( nothing to send: one row going nowhere, the exchange keeps its shape )
            sends = [ ( o.full( ( 1, ), -1, i32 ), o.zeros( ( 1, W ), ft ) ) ]
        adds = {}
        for name in self.names:
            parts = [ ob.add[ name ] for ob in outboxes if name in ob.add ]
            parts = parts or [ ( o.full( ( 1, ), -1, i32 ), o.zeros( ( 1, ), i32 ), o.zeros( ( 1, ), ft ) ) ]
            adds[ name ] = tuple( o.concatenate( [ p[ k ] for p in parts ] ) for k in range( 3 ) )
        return o.concatenate( [ d for d, _ in sends ] ), o.concatenate( [ r for _, r in sends ] ), adds

    def settle( self, res, adds ):
        """the adds routed to the boxes that hold them, and added to this rank's blocks of the outputs ( `[ B * L ]` each )"""
        o, i32, ft, B = self.o, self.i32, self.ftype, self.B
        res = list( res )
        for i, name in enumerate( self.names ):
            L = self.prog.outputs[ name ]
            bx, idx, vals = adds[ name ]
            pair = o.stack( [ o.astype( idx, ft ), vals ], axis = 1 )
            got = self.by_box( bx, pair, self.prog.add_capacity, -1 )           # [ B, P * A, 2 ]
            at = o.astype( got[ :, :, 0 ], i32 )
            offset = o.reshape( o.arange( B, i32 ) * L, ( B, 1 ) )
            at = o.where( ( at < 0 ) | ( at >= L ), at * 0 - 1, at + offset )
            res[ i ] = scatter_add( res[ i ], o.reshape( at, ( -1, ) ), o.reshape( got[ :, :, 1 ], ( -1, ) ) )
        return tuple( res )

    def send( self, dest, rows ):
        """the rows to their boxes, with a leading column saying the slot was filled: the inboxes `[ B, P * S, 1 + W ]`"""
        o = self.o
        flagged = o.concatenate( [ rows[ :, 0:1 ] * 0 + 1, rows ], axis = 1 )
        return self.by_box( dest, flagged, self.prog.send_capacity, 0 )

    def moving( self, dest ):
        o = self.o
        return self.ranks.all_sum( o.sum( o.astype( dest >= 0, self.i32 ) ) )


def _normalized( ob, o, ft, W, i32 ):
    """`ob` with flat int32 boxes and indices, rows `[ m, W ]` and values of the program's reals"""
    flat = lambda x, t: o.astype( o.reshape( x, ( -1, ) ), t )
    send = None
    if ob.send is not None:
        send = ( flat( ob.send[ 0 ], i32 ), o.astype( o.reshape( ob.send[ 1 ], ( -1, W ) ), ft ) )
    return Outbox( send, { n: ( flat( bx, i32 ), flat( idx, i32 ), flat( v, ft ) ) for n, ( bx, idx, v ) in ob.add.items() } )


def _reals( o, blocks ):
    """the reals of the rows and of the outputs: those of the first real block, else the default"""
    from .drivers.framework_defaults import ftype as default_ftype
    from .tensor.Dtype import Dtype
    arrays = [ a for a in map( _first_array, blocks ) if a is not None ]
    return next( ( d for d in map( o.dtype_of, arrays ) if d.kind == Dtype.fp().kind ), None ) or default_ftype()


def _is_float( x ):
    x = getattr( x, "value", x )
    if not ( hasattr( x, "shape" ) and hasattr( x, "dtype" ) ):
        return False
    d = x.dtype
    return bool( getattr( d, "is_floating_point", False ) ) or "float" in str( d )


def _flatten( v ):
    """the arrays of `v` ( an array, a loom tensor, or lists / tuples / dicts of them ), native, and how to put others in their place"""
    leaves = []

    def walk( x ):
        if isinstance( x, ( list, tuple ) ):
            return ( type( x ), [ walk( y ) for y in x ] )
        if isinstance( x, dict ):
            return ( dict, [ ( k, walk( y ) ) for k, y in x.items() ] )
        a = getattr( x, "value", x )
        if hasattr( a, "shape" ) and hasattr( a, "dtype" ):
            leaves.append( a )
            return len( leaves ) - 1
        return ( None, x )

    tree = walk( v )

    def rebuild( new ):
        def build( t ):
            if isinstance( t, int ):
                return new[ t ]
            kind, c = t
            if kind is None:
                return c
            if kind is dict:
                return { k: build( y ) for k, y in c }
            return kind( build( y ) for y in c )
        return build( tree )

    return leaves, rebuild


class _Args:
    """the arguments of a call, flattened: `leaves` the native arrays ( what autodiff sees ), and for each argument split along the
    boxes whether it is a list ( one value per box ), spread ( one block per rank ), or whole ( split here )"""

    def __init__( self, blocks, shared ):
        self.leaves, self.parts = [], []
        for b in blocks:
            if isinstance( b, ( list, tuple ) ):
                per = []
                for v in b:
                    lv, rb = _flatten( v )
                    per.append( ( len( self.leaves ), len( lv ), rb ) )
                    self.leaves += lv
                self.parts.append( ( "list", per ) )
            else:
                self.parts.append( ( "spread" if spread_of( b ) is not None else "whole", len( self.leaves ) ) )
                self.leaves.append( getattr( b, "value", b ) )
        lv, self.shared_rebuild = _flatten( list( shared ) )
        self.shared_at = len( self.leaves )
        self.leaves += lv

    def nb_boxes( self, ranks ):
        kind, p = self.parts[ 0 ]
        if kind == "list":
            return len( p )
        n = int( self.leaves[ p ].shape[ 0 ] )
        return n * ranks.size if kind == "spread" and ranks.one_process_per_rank else n

    def rebuild( self, leaves ):
        blocks = []
        for kind, p in self.parts:
            if kind == "list":
                blocks.append( [ rb( leaves[ at:at + n ] ) for at, n, rb in p ] )
            else:
                blocks.append( leaves[ p ] )
        return blocks, self.shared_rebuild( leaves[ self.shared_at: ] )

    def local( self, blocks, ranks, K ):
        """the blocks as the program's ranks hold them: spread ones as they are, whole ones split here, lists cut to the rank's boxes"""
        from .programs import _rank_of, spread
        P = ranks.size
        res = []
        for ( kind, _ ), v in zip( self.parts, blocks ):
            if P == 1:
                res.append( v )
            elif kind == "list":
                res.append( _spread_along_boxes( v, ranks ) )
            elif kind == "spread" or spread_of( v ) is not None:     # ( a sharded Jax array seen traced at the call, concrete here )
                res.append( spread( v, ranks ) )
            elif ranks.one_process_per_rank:
                B, p = K // P, _rank_of( ranks )
                res.append( spread( v[ p * B:( p + 1 ) * B ], ranks ) )
            else:
                res.append( spread_host( np.asarray( to_host( v ) ), ranks ) )
        return res

    def gradients( self, outs, ranks, K, o ):
        """the gradient of each leaf ( `None`: not a float, or not on this rank ) from what `_per_rank_back` gives, in its order.
        Over processes ( Torch ), each rank gets what its boxes contribute -- its own blocks, its own copy of what is whole"""
        from .programs import _rank_of
        P = ranks.size
        procs = ranks.one_process_per_rank and P > 1
        B = K // P
        p = _rank_of( ranks ) if procs else 0
        grads = [ None ] * len( self.leaves )
        it = iter( outs )
        for kind, part in self.parts:
            if kind == "list":
                mine = range( p * B, ( p + 1 ) * B ) if procs else range( len( part ) )
                for b in mine:
                    at, n, _ = part[ b ]
                    for i in range( at, at + n ):
                        if _is_float( self.leaves[ i ] ):
                            grads[ i ] = next( it )[ 0 ]
            elif _is_float( self.leaves[ part ] ):
                g = next( it )
                if kind == "whole" and procs:
                    x = self.leaves[ part ]
                    g = o.concatenate( [ o.zeros( ( p * B, ) + tuple( x.shape[ 1: ] ), o.dtype_of( g ) ), g,
                                         o.zeros( ( K - ( p + 1 ) * B, ) + tuple( x.shape[ 1: ] ), o.dtype_of( g ) ) ] )
                grads[ part ] = g
        for i in range( self.shared_at, len( self.leaves ) ):
            if _is_float( self.leaves[ i ] ):
                g = next( it )
                # ( Jax: one array of the ranks' contributions, summed; one process: its own )
                grads[ i ] = o.sum( g, axis = 0 ) if P > 1 and not procs else g[ 0 ]
        grads = [ g if g is None else o.astype( g, o.dtype_of( x ) ) for g, x in zip( grads, self.leaves ) ]
        return grads


def _vjp_aux( fw, f, primals ):
    """`f( *primals ) -> ( reals, aux )`: `( reals, aux, pullback )`, `pullback( cotangents of reals )` giving one gradient per primal"""
    if fw == "jax":
        import jax
        from .drivers.JaxOperations import varying
        # ( varying along the program's axes: a shared value's cotangent is this rank's contribution, not already summed over them )
        reals, pull, aux = jax.vjp( f, *[ varying( p ) for p in primals ], has_aux = True )
        return reals, aux, pull
    if fw == "torch":
        import torch
        with torch.enable_grad():
            tracked = [ p.detach().requires_grad_( True ) for p in primals ]
            reals, aux = f( *tracked )

        def pull( cts ):
            pairs = [ ( r, c ) for r, c in zip( reals, cts ) if r.requires_grad ]
            gs = [ None ] * len( tracked )
            if pairs and tracked:
                gs = torch.autograd.grad( [ r for r, _ in pairs ], tracked, [ c for _, c in pairs ], allow_unused = True )
            return tuple( torch.zeros_like( p ) if g is None else g for p, g in zip( tracked, gs ) )

        return reals, aux, pull
    raise NotImplementedError( f"the adjoint of a BoxProgram needs Jax or Torch ( { fw } has no autodiff )" )


def _torch_wants_grad( floats ):
    import torch
    return torch.is_grad_enabled() and any( isinstance( l, torch.Tensor ) and l.requires_grad for l in floats )


_torch_function = []


def _torch_call( prog, args, ranks, K ):
    """the call as a `torch.autograd.Function`: the rounds and their inboxes forward, the rounds in reverse backward"""
    if not _torch_function:
        import torch

        class BoxProgramCall( torch.autograd.Function ):
            @staticmethod
            def forward( ctx, prog, args, ranks, K, *leaves ):
                outs, rounds, tape = prog._forward( args, ranks, K, list( leaves ), True )
                ctx.keep = ( prog, args, ranks, K, leaves, rounds, tape )
                return tuple( outs )

            @staticmethod
            def backward( ctx, *cots ):
                prog, args, ranks, K, leaves, rounds, tape = ctx.keep
                del ctx.keep
                grads = prog._backward( args, ranks, K, list( leaves ), tape, rounds, list( cots ) )
                return ( None, None, None, None ) + tuple( grads )

        _torch_function.append( BoxProgramCall )
    return list( _torch_function[ 0 ].apply( prog, args, ranks, K, *args.leaves ) )


# ---- boxes kept out of memory ------------------------------------------------------------------------------------------------------

class BoxStore:
    """the values of `nb_boxes` boxes, read one box at a time: `load( b )` gives box `b` ( an array, or tuples / lists / dicts of them,
    on the host ). What makes a `BoxProgram` run out of core. `BoxStore.of( x )` reads anything indexable along its first axis -- a
    numpy array or memmap, a zarr array, a dask array ( computed box by box ), a list"""

    def __init__( self, load, nb_boxes ):
        self.load = load
        self.nb_boxes = int( nb_boxes )

    @staticmethod
    def of( x ):
        if isinstance( x, BoxStore ):
            return x
        if hasattr( x, "compute" ):         # dask: the box only
            return BoxStore( lambda b: x[ b ].compute(), len( x ) )
        return BoxStore( lambda b: x[ b ], len( x ) )


def _is_store( x ):
    """whether `x` asks for the out-of-core runtime: a `BoxStore`, a zarr or a dask array"""
    if isinstance( x, BoxStore ):
        return True
    mod = type( x ).__module__ or ""
    return mod.startswith( "zarr" ) or mod.startswith( "dask" )


def _map_arrays( v, f ):
    """`f` applied to the arrays of `v` ( an array, or tuples / lists / dicts of them ); the rest kept"""
    if isinstance( v, ( list, tuple ) ):
        return type( v )( _map_arrays( x, f ) for x in v )
    if isinstance( v, dict ):
        return { k: _map_arrays( x, f ) for k, x in v.items() }
    return f( v ) if hasattr( v, "shape" ) and hasattr( v, "dtype" ) else v


# ---- where the boxes go ----------------------------------------------------------------------------------------------------------

def _ranks_for( blocks, ranks ):
    """the ranks of a call: given, else those of a spread argument, else `loom.default_ranks`, else what the process runs in"""
    if ranks is not None:
        return ranks
    for b in blocks:
        s = spread_of( b )
        if s is not None:
            return s.ranks
    import loom
    if loom.default_ranks is not None:
        return loom.default_ranks
    return detect_ranks() or LocalRanks()


def _spread_along_boxes( x, ranks ):
    """`x` as the program's ranks hold it: as it is when spread, its boxes split between the ranks when whole. A list holds one
    value per box, whatever it is ( boxes of different sizes... ): every rank keeps its boxes"""
    if isinstance( x, ( list, tuple ) ):
        if ranks.size == 1:
            return list( x )
        if not ranks.one_process_per_rank:
            raise NotImplementedError( "a list of boxes on Jax devices: one traced program for every device, so the boxes need one shape -- give arrays `[ K, ... ]`" )
        from .programs import _rank_of
        B = len( x ) // ranks.size
        p = _rank_of( ranks )
        return list( x[ p * B:( p + 1 ) * B ] )
    if ranks.size == 1 or spread_of( x ) is not None:
        return x
    return spread_host( np.asarray( to_host( x ) ), ranks )


def _nb_boxes( x, ranks ):
    if isinstance( x, ( list, tuple ) ):
        return len( x )             # ( a list is whole: one value per box, every box )
    n = int( getattr( x, "value", x ).shape[ 0 ] )
    return n * ranks.size if ranks.one_process_per_rank else n


def _first_array( x ):
    """the first array in `x` ( an array, a loom tensor, or lists / tuples / dicts of them ), or `None`"""
    if isinstance( x, ( list, tuple ) ):
        return next( ( a for a in map( _first_array, x ) if a is not None ), None )
    if isinstance( x, dict ):
        return _first_array( list( x.values() ) )
    v = getattr( x, "value", x )
    return v if hasattr( v, "shape" ) and hasattr( v, "dtype" ) else None
