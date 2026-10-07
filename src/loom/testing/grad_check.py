"""Agnostic derivative checker (Jax today, Torch tomorrow).

`check_grad` compares the derivative of a function, obtained through the driver's
adjoint mode (`loom.vjp`), with its centered finite-difference estimate. Nothing is
framework-specific: everything goes through `loom.vjp` / `loom.random` and through
tensor arithmetic (`+`, `*`, `.sum()`), common to Jax and Torch -- see
the architecture note on agnostic tests.

Principle: we do not materialize the full Jacobian, we test it on random
projections. With an input tangent `v` and an output cotangent `w`,
the adjoint gives exactly `< vjp(w), v > = < w, J v >`, and the right-hand
side is estimated by `( f(x+εv) - f(x-εv) ) / 2ε`. A disagreement signals a
wrong derivative (a null adjoint makes it stand out immediately).

`f` and its arguments are expressed as `Tensor`: an input `Tensor` is differentiated
with respect to its buffer, an output `Tensor` is compared on its dense view
(`.value`) -- the capacity padding is removed for us, without writing
`.raw[ :n ]`.
"""
import numpy

from loom.tensor import Tensor
import loom
from loom.drivers import framework_defaults


def _raw( x ):
    """The differentiable buffer behind `x`: that of a `Tensor`, or `x` as is."""
    return x.raw if isinstance( x, Tensor ) else x


def check_grad( f, *args, eps = 1e-4, rtol = 2e-3, atol = 1e-4, seed = None ):
    """Checks the derivative of `f` by finite difference.

    `f` takes one or more `Tensor` (or raw driver tensors) and returns a `Tensor`
    (or a raw tensor). Raises an `AssertionError` if the adjoint and the finite difference
    differ by more than `atol + rtol * |num|`. Returns the pair ( adjoint, finite diff. ).

    `seed` fixes the random projections (the cotangent `w` and the tangents `vs`). Without it
    they come from the process counter of `loom.random`, hence from HOW MANY draws the
    previous tests made: the same test then draws different directions depending on whether it is
    run alone or in the suite, and a check whose error depends on the direction may pass
    on one side and fail on the other. Passing it makes the test reproducible.

    BEWARE of the SHAPE of the tolerance. `atol + rtol * |num|` assumes that the finite-difference
    error is proportional to what is measured. That is true when it comes from
    truncation or rounding; it is NOT when `f` has small discontinuities (a
    value-adaptive quadrature, a subdivision that flips): the gap is then a JUMP
    DIVIDED BY `2 eps`, an ABSOLUTE quantity, independent of the projection drawn. Since `num`
    is the product of the Jacobian by a random direction, it can be small where the jump
    is not -- and it is `atol`, not `rtol`, that must then carry the floor. See
    `test_PowerDiagram::the_subdivided_quadrature_derives_right`.
    """
    # Centered finite difference amplifies rounding noise (~machine_eps / eps): in FP32
    # (machine_eps ~1.2e-7) with eps=1e-4 it stays marginal, but enough to drown a real
    # gap of the size of `tol`. We force FP64 for the duration of the check, then restore the
    # previous value so as not to leak it into the following tests (driver is a
    # global singleton shared by the whole test process).
    previous_size = loom.default_dtype.size
    loom.default_dtype.size = 64
    try:
        # a host (numpy) primal is cast to the driver's own array type: `p + eps * v` below mixes it
        # with `loom.random` draws, which a framework tensor does not accept from a numpy array
        primals = [ framework_defaults.array( _raw( a ) ) if isinstance( _raw( a ), numpy.ndarray ) else _raw( a ) for a in args ]

        # `f` generally returns a `Tensor`: it is its DENSE view that we compare (the capacity
        # padding is not a real output). The extent of an axis written by the kernel is a DEVICE
        # value under a trace; we therefore capture the dense shape now, at eager execution, as
        # Python integers -- the per-trace cropping then becomes static, hence compatible with the trace.
        probe = f( *primals )
        if isinstance( probe, Tensor ):
            dense_shape = tuple( probe.shape )
            crop  = lambda t: t.raw[ tuple( slice( 0, s ) for s in dense_shape ) ]
            out_f = lambda *r: crop( f( *r ) )
        else:
            out_f = f

        out, pullback = framework_defaults.ops( primals[ 0 ] ).vjp( out_f, *primals )

        # random cotangent on the output, random tangents on the inputs
        w  = framework_defaults.random( out.shape, seed = seed )
        vs = [ framework_defaults.random( p.shape, seed = None if seed is None else seed + 1 + i )
               for i, p in enumerate( primals ) ]

        # adjoint: < vjp(w), v >, summed over the inputs
        grads = pullback( w )
        ana = sum( float( ( g * v ).sum() ) for g, v in zip( grads, vs ) )

        # centered finite difference: < w, ( f(x+εv) - f(x-εv) ) / 2ε >
        plus  = out_f( *[ p + eps * v for p, v in zip( primals, vs ) ] )
        minus = out_f( *[ p - eps * v for p, v in zip( primals, vs ) ] )
        num = float( ( ( plus - minus ) * w ).sum() ) / ( 2 * eps )
    finally:
        loom.default_dtype.size = previous_size

    err = abs( ana - num )
    tol = atol + rtol * abs( num )
    assert err <= tol, (
        f"incorrect derivative: adjoint = { ana }, finite diff. = { num }, "
        f"|Δ| = { err } > { tol }"
    )
    return ana, num
