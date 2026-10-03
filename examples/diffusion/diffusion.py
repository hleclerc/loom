"""A DIFFERENTIABLE diffusion solver, and what AXES buy.

This is a FOREIGN USER of loom: it imports only `loom`, and nothing it does resembles
a Laguerre cell -- Cartesian grid, five-point stencil, no ragged, no geometry.

    du/dt = k laplacian( u ),   explicit time step, imposed temperature on the boundary

    u'( a ) = u( a ) + c * sum_{b neighbour} ( u( b ) - u( a ) ),   c = dt k / h^2

WHAT THE EXAMPLE SHOWS: the kernel body NEVER counts dimensions. It asks for its axes,
walks them, and moves along one of them. The same body therefore works in 2D, in 3D, batched or
not -- and a `vmap` adds an axis to it without it knowing that it exists ( this is tested ).
"""
# `loom.X` and not `from loom import X`: in a tutorial, one must see where each name comes from.
import loom


# THE KERNEL, in plain sight: the tutorial reads without navigating through files.
#
# The `namespace { }` is ours -- so our `#include`s go wherever we want, and it gives everything it
# contains INTERNAL linkage ( several kernels end up linked in the same library, see
# `compilation/catalogue.py` ).
#
# Loom calls `void kernel( auto &&queue, auto &&batch_axes, auto &&args )` :
#
#   queue       the execution context. `queue.run_parallel` is ITS tool, not an obligation:
#               a Kokkos or OpenMP user ignores it and takes `queue.stream` plus the pointers and
#               the shapes of `args`.
#   batch_axes  the axes that the call added ( what a `vmap` makes ).
#   args        our arguments under their Python names, plus `machine` and `errors`. `TF` is the
#               real scalar of the call.
#
# THE THREE AXIS PRIMITIVES, and everything follows from them:
#
#   coords[ axis ]              the coordinate BY NAME, not by position
#   coords + axis               the neighbour along THAT axis; the other coordinates do not
#                               move, including those of the batch
#   coords.axes - batch_axes    my own axes, by set subtraction at compile time
#
_forward = loom.FfiCode(
    code = """
        namespace {
            struct OneStep {
                /// the boundary carries an imposed temperature: it is never updated.
                HD bool on_border( auto coords, auto main_axes, const auto &args ) const {
                    return any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.inputs.temperature.size( axis );
                    } );
                }

                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const auto main_axes = coords.axes - batch_axes;

                    const TF uc = args.inputs.temperature( coords );
                    if ( on_border( coords, main_axes, args ) ) {
                        args.outputs.temperature( coords ) = uc;
                        return;
                    }

                    TF total = 0;
                    for_each( main_axes, [&]( auto axis ) {
                        total += args.inputs.temperature( coords + axis )
                               + args.inputs.temperature( coords - axis ) - 2 * uc;
                    } );
                    args.outputs.temperature( coords ) = uc + args.inputs.coef * total;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( OneStep(), args.outputs.temperature.domain(), args, batch_axes );
            }
        }
    """,
)

# THE ADJOINT is a kernel like any other: it is the CALL that takes both and carries the name.
#
# `u( a )` contributes to the output of `a` ( the identity term, and `-2 c D u( a )` if `a` is
# interior, `D` being the number of axes ) and to that of each INTERIOR neighbour `b`. Hence one
# read of the neighbours and a single write: a pure GATHER, without atomic accumulation.
_backward = loom.FfiCode(
    code = """
        namespace {
            struct OneStepAdjoint {
                HD bool on_border( auto coords, auto main_axes, const auto &args ) const {
                    return any_of( main_axes, [&]( auto axis ) {
                        return coords[ axis ] == 0
                            || coords[ axis ] + 1 == args.inputs.temperature.size( axis );
                    } );
                }

                /// true if `coords` shifted by `d` along `axis` is still inside the grid
                HD bool inside( auto coords, auto axis, SI d, const auto &args ) const {
                    const SI c = coords[ axis ] + d;
                    return c >= 0 && c < args.inputs.temperature.size( axis );
                }

                HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                    const auto main_axes = coords.axes - batch_axes;

                    // `coef` is a constant of the problem, never perturbed: its gradient
                    // would require a global reduction, and it is not written.
                    static_assert( ! args.grad_of_inputs.coef.is_valid,
                        "diffusion: the gradient with respect to dt k / h^2 is not implemented" );

                    // an output buffer is NOT guaranteed to be zero: when the cotangent is a
                    // symbolic zero the null gradient must still be written.
                    constexpr bool is_null = args.grad_of_outputs.temperature.surely_null;

                    TF res = 0;
                    if constexpr ( ! is_null ) {
                        constexpr SI D = DECAYED_TYPE_OF( main_axes )::ct_size;
                        const TF c = args.inputs.coef;
                        const TF g = args.grad_of_outputs.temperature( coords );

                        res = on_border( coords, main_axes, args ) ? g : g * ( 1 - c * ( 2 * D ) );

                        for_each( main_axes, [&]( auto axis ) {
                            for ( SI s = -1; s <= 1; s += 2 ) {
                                if ( ! inside( coords, axis, s, args ) )
                                    continue;
                                const auto neighbor = s > 0 ? coords + axis : coords - axis;
                                if ( ! on_border( neighbor, main_axes, args ) )
                                    res += c * args.grad_of_outputs.temperature( neighbor );
                            }
                        } );
                    }

                    if constexpr ( args.grad_of_inputs.temperature.is_valid )
                        args.grad_of_inputs.temperature( coords ) = res;
                }
            };

            void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
                queue.run_parallel( OneStepAdjoint(), args.inputs.temperature.domain(), args, batch_axes );
            }
        }
    """,
)


def step( temperature, coef ):
    """ONE explicit time step.

    `temperature` is a `( ny, nx )` array of the framework ( or a plain numpy one ) and `coef` is
    `dt k / h^2` -- the diffusivity is CONSTANT. Returns the updated temperature, differentiable with
    respect to the one given.

    NOTHING TO WRAP: arrays enter as they are, loom reads their shape and type. The axes
    are DEDUCED from them -- anonymous, named by their position at lowering -- so one writes a
    stencil without uttering the word "axis".

    `loom.mutable` says the only thing left to say: this grid is READ AND REWRITTEN. Since the
    inputs and outputs of a call are disjoint, that makes two buffers -- which the kernel sees
    under `args.inputs.temperature` and `args.outputs.temperature`, and the adjoint under
    `args.grad_of_outputs.temperature` and `args.grad_of_inputs.temperature`."""
    return loom.ffi_call( "diffusion_step", _forward, _backward,
        temperature = loom.mutable( temperature ),
        coef = coef,
    )


def evolve( temperature, coef, nb_steps ):
    """`nb_steps` consecutive steps -- the chain that the adjoint must walk back up."""
    for _ in range( nb_steps ):
        temperature = step( temperature, coef )
    return temperature


if __name__ == "__main__":
    # enough to see the example run without installing anything: `python diffusion.py`
    #
    # NOTHING GOES THROUGH THE HOST. The bump is built by a C++ EXPRESSION of the coordinates,
    # compiled and executed where the tensor lives; the final reads are loom
    # reductions. No numpy, and nothing that assumes one device rather than another.
    n, nb_steps, coef = 21, 40, 0.2

    u = loom.RealTensor[ n, n ].expr(
        # the boundary is imposed at zero, and it is the same expression that says so
        "i_axis_0 == 0 || i_axis_1 == 0 || i_axis_0 + 1 == n_axis_0 || i_axis_1 + 1 == n_axis_1"
        " ? TF( 0 )"
        " : exp( - ( ( i_axis_0 - c ) * ( i_axis_0 - c )"
        "          + ( i_axis_1 - c ) * ( i_axis_1 - c ) ) / 8 )",
        c = ( n - 1 ) / 2,
    )

    v = evolve( u, coef, nb_steps )
    print( f"{ nb_steps } diffusion steps on a { n }x{ n } grid ( c = { coef } )" )
    print( f"  peak { float( u.max() ):.4f} -> { float( v.max() ):.4f}" )
    # the sum DECREASES: the boundary is imposed at 0, so heat escapes through the sides.
    print( f"  sum  { float( u.sum() ):.4f} -> { float( v.sum() ):.4f}   ( it leaks through the boundary )" )
