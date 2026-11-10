"""Run by `test_box_program_torch.py`: the travellers of `box_program_cases.py` on 4 Torch processes ( gloo ), spread blocks or
whole tensors ( the ranks then found from `torch.distributed` ). Prints `all ranks ok` when every rank agrees."""
import os
import socket
import sys

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )


def work( rank, size, port ):
    import numpy as np
    import torch
    import torch.distributed as dist
    import loom
    from box_program_cases import run_and_check
    os.environ.update( MASTER_ADDR = "127.0.0.1", MASTER_PORT = str( port ) )
    dist.init_process_group( "gloo", rank = rank, world_size = size )
    loom.default_framework = "torch"
    assert isinstance( loom.detect_ranks(), loom.TorchRanks )
    ranks = loom.TorchRanks()
    host = lambda x: np.asarray( loom.gather_to_host( x ) )

    def spread_call( prog, values, hops, strides, weights ):
        B = len( values ) // size
        mine = lambda a: loom.spread( torch.as_tensor( a[ rank * B:( rank + 1 ) * B ].copy() ), ranks )
        out = prog( mine( values ), mine( hops ), mine( strides ), shared = ( torch.as_tensor( weights ), ) )
        assert isinstance( out[ "res" ].value, torch.Tensor ) and tuple( out[ "res" ].value.shape ) == ( B, values.shape[ 1 ] )
        return host( out[ "res" ] ), host( out[ "visits" ] )

    def whole_call( prog, values, hops, strides, weights ):
        out = prog( *( torch.as_tensor( a ) for a in ( values, hops, strides ) ), shared = ( torch.as_tensor( weights ), ) )
        return host( out[ "res" ] ), host( out[ "visits" ] )

    for call in ( spread_call, whole_call ):
        run_and_check( {}, call, K = 2 * size, L = 8 )
        prog = run_and_check( { "send_capacity": 1, "add_capacity": 1 }, call, K = size, L = 16 )
        assert prog.send_capacity.value > 1

    # the adjoint: each rank gets what its boxes contribute ( its blocks; for what is whole, its copy ), summed here to check
    from box_program_cases import case, check_grads, program
    for how in ( "spread", "whole" ):
        for K, caps in [ ( size, {} ), ( 2 * size, { "send_capacity": 1, "add_capacity": 1 } ) ]:
            L = 8
            B = K // size
            values, hops, strides, weights = case( K, L )
            C = np.random.default_rng( 1 ).random( ( K, L ) )
            prog = program( K, L, **caps )
            prog.tape_rounds.value = 2
            w = torch.tensor( weights, requires_grad = True )
            if how == "spread":
                mine = lambda a: torch.as_tensor( a[ rank * B:( rank + 1 ) * B ].copy() )
                v = mine( values ).requires_grad_( True )
                out = prog( loom.spread( v, ranks ), loom.spread( mine( hops ), ranks ), loom.spread( mine( strides ), ranks ), shared = ( w, ) )
                ( out[ "res" ].value * mine( C ) ).sum().backward()
                got = [ torch.empty_like( v.grad ) for _ in range( size ) ]
                dist.all_gather( got, v.grad )
                gv = torch.cat( got )
            else:
                v = torch.tensor( values, requires_grad = True )
                out = prog( v, torch.as_tensor( hops ), torch.as_tensor( strides ), shared = ( w, ) )
                ( out[ "res" ].value * torch.as_tensor( C[ rank * B:( rank + 1 ) * B ] ) ).sum().backward()
                gv = v.grad.clone()
                dist.all_reduce( gv )
            gw = w.grad.clone()
            dist.all_reduce( gw )
            check_grads( gv.numpy(), gw.numpy(), values, hops, strides, weights, C )
            assert prog.tape_rounds.value >= prog.last_rounds
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind( ( "127.0.0.1", 0 ) )
        port = s.getsockname()[ 1 ]
    mp.spawn( work, args = ( 4, port ), nprocs = 4 )
    print( "all ranks ok" )
