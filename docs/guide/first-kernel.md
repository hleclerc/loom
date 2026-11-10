# Your first kernel

The smallest complete one: `y = a * x + b`, element by element. One file, a C++ body and a Python
function, and it runs on Jax, Torch, NumPy and CuPy arrays, in any number of dimensions.

```python
# affine.py
import loom

_forward = loom.FfiCode( code = """
    namespace {
        struct Affine {
            HD void operator()( auto coords, auto &&args, auto batch_axes ) const {
                args.outputs.y( coords ) = args.inputs.a * args.inputs.x( coords ) + args.inputs.b;
            }
        };

        void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {
            queue.run_parallel( Affine(), args.inputs.x.domain(), args, batch_axes );
        }
    }
""" )


def affine( x, a, b ):
    """`a * x + b`, element by element."""
    return loom.ffi_call( "affine", _forward,
        x = x,
        a = a,
        b = b,
        y = loom.out( loom.RealTensor.like( x ) ),
    )


loom.export_namespace( globals() )
```

And from the user's side:

```python
import jax.numpy as jnp
from affine import affine

affine( jnp.linspace( 0.0, 1.0, 5 ), 2.0, 1.0 )     # Array([1. , 1.5, 2. , 2.5, 3. ])
```

The first call compiles the kernel (a few seconds, `ninja` prints its steps), the next ones find it
in the cache.

## The body: what happens at one point

`Affine::operator()` is called once per point of the domain, in parallel:

- `coords` is the position of the point. It is not a list of integers to unpack: `x( coords )`
  reads the value there, whatever the rank of `x`.
- `args` holds the arguments of the call **under their Python names**, in groups: what is read in
  `args.inputs`, what is written in `args.outputs`.
- `batch_axes` are the axes a `vmap` added; this kernel has nothing to do with them.

`a` and `b` were Python floats: they arrive as scalars, read as they are. `HD` makes the function
callable on the host and on a CUDA device, so the same body serves both.

## The launch: `kernel( queue, batch_axes, args )`

loom calls one function, `kernel`, with the same three things. `queue.run_parallel( functor, domain,
args, batch_axes )` runs the functor on every point of `domain`, on the CPU's threads or on the
card. Here the domain is the one of `x`: one point per element.

The `namespace { }` is required: several kernels can end up in the same library, and each `kernel`
must stay private to its own.

## The call: `loom.ffi_call`

`loom.ffi_call( name, forward, **arguments )` names the call, gives its kernel, and passes the
arguments. **The role of each argument is said where it is passed**:

- a plain value is an **input**: `x`, `a`, `b`;
- `loom.out( ... )` is an **output**: loom allocates it, the kernel writes it, the call returns it.

`loom.RealTensor.like( x )` declares an output with the axes, element type and device of `x`, and
no content yet. The other roles (`loom.mutable`, `loom.scratch`, ...) are in
[Roles of the arguments](./arguments).

## The last line: the boundary

`loom.export_namespace( globals() )` puts a boundary around the module: `affine` becomes the public
version of itself. Coming in, the arrays become loom tensors, so that inside the module there is
only one kind of data, whatever the framework; going out, what it returns becomes an array of the
framework its arguments came from.

| `x` is | `affine( x, 2.0, 1.0 )` is |
|---|---|
| a Jax array | a Jax array |
| a Torch tensor | a Torch tensor, on the same device |
| a NumPy array | a NumPy array |
| a CuPy array | a CuPy array |

Your users never see a loom tensor, and never write `.value`.

## What you already have

Nothing in the body says "1-D", so the same kernel takes a matrix, or a 3-D grid, as it is. Under
Jax, it also runs inside `jax.jit`, and under `jax.vmap` (or `torch.func.vmap`) as a single launch
over one more axis. What it does not have yet is a gradient: that is a second kernel, see
[The adjoint](./adjoint).

## Putting the C++ in a header

Past a few lines, the body is better in a `.h` beside the `.py`, which is where loom looks for it
by default:

```python
_forward = loom.FfiCode.inline( "affine::run( queue, batch_axes, args );", includes = [ "affine.h" ] )
```

`FfiCode.inline` takes only the inside of `kernel`; the rest of the C++ is in `affine.h`.

## Next

[Axes and coordinates](./axes) · [Roles of the arguments](./arguments) · [Diffusion](/tutorials/diffusion)
