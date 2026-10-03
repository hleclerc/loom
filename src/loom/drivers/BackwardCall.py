"""The backward pass of a `driver.call`, expressed as an ORDINARY kernel call whose body is the code's
backward -- framework-neutral: the caller passes `run( code, ca, device, prefix ) -> ( output
CallArgs, result arrays )`, the framework's way of executing one kernel (see `JaxFfi._run`,
`TorchFfi._run`). Shared by the Jax and the Torch drivers."""
from __future__ import annotations


def _grad_tensor( inst, array ):
    """A bare tensor holding `array`, shaped like `inst` -- a residual (a forward input/output) or
    a cotangent, entering the backward kernel as an input bound at the size its data has."""
    from ..tensor.storage import Fill
    from ..tensor.Tensor import Tensor
    res = Tensor.like( inst )
    if inst.is_fill:
        # a FILL re-enters the backward as a fill too (its residual is the same scalar): keep it
        # symbolic so it lowers to a `FillTensor` again, not a scalar-buffer TensorView with a [n]
        # logical shape. Being a fill is STATED, never inferred -- one scalar looks like any other.
        res.storage = Fill( res._as_declared( array ), inst.reference_shape )
    else:
        res.set_raw( array )
    return res


def _grad_shapevar( inst, raw ):
    """A `ShapeVar`-shaped object carrying a FIXED count `raw` -- the residual for a `ShapeVar`
    member entering the backward, mirroring `_grad_tensor` for a `Tensor` member.

    Needed because `ShapeVar._count` is, by its own contract, "a count produced by a kernel: a
    driver tensor, POSSIBLY TRACED" (`tensor/ShapeVar.py`). Reusing the live, shared `inst` object
    directly (as the backward used to, for every ShapeVar/Axis/CtShapeVar member alike) reads
    whatever `_count` holds AT THE MOMENT `op_bwd` executes -- safe only when that is the very
    trace that resolved it. Under `lax.scan`'s differentiation, `op_bwd` is replayed in a later,
    separate trace, so a resolved-but-still-tracer count from the original trace is dead by then.
    `raw` here is instead the count as it flowed through `driver.call`'s own residual channel
    (`full_in`/`out_values`), which Jax DOES keep valid across that replay."""
    from ..tensor.ShapeVar import ShapeVar
    res = ShapeVar.__new__( ShapeVar )
    res.usages = []
    res._compact_usages_at = 8
    res.dep_axes = inst.dep_axes
    # the batch axes come along: they decide the RANK the count crosses at (one per item, see
    # `CallArg_ShapeVar`), and `raw` is the forward's buffer -- dropping them here would bind a
    # per-item count as a single slot.
    #
    # ... but only when the forward count WAS per item (a kernel wrote it): a host-known count is one
    # number for every item and crossed as such, so the residual -- `raw` is that one number -- must
    # not claim a slot per item either.
    res.batch_axes = list( inst.batch_axes ) if inst.is_kernel_written() else []
    res.prescribed_value = None
    res._count = raw
    return res


def call_backward( code, ca, device, prefix, inputs, outputs,
                   full_in, out_values, perturbed, cotangents, run ):
    """The backward pass, expressed as an ORDINARY kernel call whose body is the code's backward.

    Each forward argument `X` yields two backward arguments, of the SAME type as `X` (a bare
    tensor, or an aggregate mirrored member by member):

    * a RESIDUAL `X`: the forward values, re-entering as backward INPUTS under the very name they
      had -- so the body reads `cell.vertex_positions`, `inp`, ... exactly as the forward did;
    * a gradient `grad_for_X`, whose tensors are, per member:
        - a float forward OUTPUT   -> its cotangent, a backward INPUT (a `SymbolicZero` lowers to a
          `ZeroTensor`: read as 0, no buffer, dropped at compile time);
        - a float forward INPUT     -> a backward OUTPUT when perturbed, else a `NoneTensor` (the
          body skips it at compile time, `grad_for_...is_valid` being false);
        - anything else             -> a `NoneTensor`.

    An aggregate `grad_for_cell` thus carries a MIX of backward-input and backward-output members;
    the per-member io policy already handles that (see `CallArg_Aggregate`). Non-tensor members
    (`Axis`, `ShapeVar`, `CtShapeVar`) are SHARED from the primal, so a gradient buffer resolves
    its capacity from the forward tensor it mirrors.

    Returns the tuple of cotangents, one per input, in `inputs` order -- `None` (Jax's own
    symbolic-zero marker for a `custom_vjp` bwd output) wherever no gradient is wanted, which
    covers both non-float inputs and non-perturbed float ones uniformly.
    """
    from ..tensor.Tensor import Tensor
    from .CallArgsAnalysis import CallArgsAnalysis
    from ..util.annotations import annotations
    from ..util.Aggregate import Aggregate, get_attribute

    # leaf-indexed facts (by tensor identity), so the structural walk below can consult them.
    io_of, residual_of = {}, {}
    for k, t in enumerate( inputs ):
        if hasattr( t, "inst" ):
            io_of[ id( t.inst ) ], residual_of[ id( t.inst ) ] = "input", full_in[ k ]
    for j, t in enumerate( outputs ):
        if hasattr( t, "inst" ):
            io_of[ id( t.inst ) ], residual_of[ id( t.inst ) ] = "output", out_values[ j ]
    cotangent_of = { id( t.inst ): cotangents[ j ]
                     for j, t in enumerate( outputs ) if hasattr( t, "inst" ) }
    perturbed_of = { id( t.inst ): perturbed[ k ] for k, t in enumerate( inputs ) if hasattr( t, "inst" ) }

    output_paths = []
    grad_obj_of = {}   # id( primal input leaf ) -> its gradient tensor (a backward output)

    def _is_agg( obj ):
        return isinstance( obj, Aggregate )

    def _blank( inst ):
        obj = type( inst ).__new__( type( inst ) )
        obj.name = getattr( inst, "name", None )   # only a NESTED aggregate carries a field name
        # carry the batch axes over: the backward is an ordinary call, so `CallArgsAnalysis` must see
        # them on the residual/grad aggregates to build `global_batch_indices` and let the body's
        # `plan( batch_index )` squeeze the batch (without this the backward would run unbatched).
        obj.batch_axes = list( getattr( inst, "batch_axes", [] ) )
        return obj

    def _build( inst, path ):
        """`( residual, grad )` mirroring `inst` (a tensor or a whole aggregate subtree)."""
        if _is_agg( inst ):
            residual, grad = _blank( inst ), _blank( inst )
            for mname in annotations( type( inst ) ):
                member = get_attribute( mname, inst )
                r, g = _build( member, f"{ path }.{ mname }" )
                residual.__dict__[ mname ], grad.__dict__[ mname ] = r, g
            return residual, grad

        if not isinstance( inst, Tensor ):
            from ..tensor.ShapeVar import ShapeVar
            if isinstance( inst, ShapeVar ):
                # a data-dependent ShapeVar (its count came from a kernel, via `driver.call`'s own
                # input/output tracking) reuses the properly-threaded residual value; a purely
                # static one (never bound as an FFI buffer -- `residual_of` has nothing for it)
                # falls through to the plain shared-object case below, same as Axis/CtShapeVar.
                raw = residual_of.get( id( inst ) )
                if raw is not None:
                    shared = _grad_shapevar( inst, raw )
                    return shared, shared
            return inst, inst   # Axis / CtShapeVar (or a static ShapeVar): shared, so shapes
                                 # resolve -- this holds whether `inst` is a nested aggregate
                                 # member OR a bare top-level kwarg (e.g. `Cell.measure`'s
                                 # `nb_map_items`)

        # a tensor leaf: the residual is bound to whatever forward value it held.
        arr = residual_of.get( id( inst ) )
        residual = _grad_tensor( inst, arr ) if arr is not None else Tensor.like( inst )

        io = io_of.get( id( inst ) )
        if io == "output" and inst.is_differentiable:
            # the cotangent enters as a backward INPUT -- a real buffer (a `TensorView`) or the
            # framework's symbolic zero (a `ZeroTensor`): both just get stored, `_grad_tensor` /
            # `is_symbolic_zero` tell them apart, no special case here.
            grad = _grad_tensor( inst, cotangent_of.get( id( inst ) ) )
        elif io == "input" and inst.is_differentiable and perturbed_of.get( id( inst ), False ):
            grad = Tensor.like( inst )
            output_paths.append( path )
            grad_obj_of[ id( inst ) ] = grad
        else:
            grad = Tensor.like( inst )   # non-differentiable or non-perturbed -> a NoneTensor
        return residual, grad

    def _fresh_scratch( inst ):
        """A scratch argument's stand-in for the backward: the same SHAPE, none of the values.

        A tensor gives an empty one of its kind; an AGGREGATE is mirrored member by member -- fresh
        tensors, and its non-tensor members (`Axis` / `ShapeVar` / `CtShapeVar`) SHARED with the
        primal, exactly as `_build` shares them, so the new buffers resolve their capacity from the
        counts the forward already grew. Declaring the aggregate itself as a backward output is then
        enough: the analysis walks it and allocates each member, just as the forward did.
        """
        if not _is_agg( inst ):
            return Tensor.like( inst )
        obj = _blank( inst )
        for mname in annotations( type( inst ) ):
            member = get_attribute( mname, inst )
            obj.__dict__[ mname ] = ( _fresh_scratch( member )
                                      if isinstance( member, Tensor ) or _is_agg( member )
                                      else member )
        return obj

    # THE BACKWARD'S ARGUMENTS, and their groups. A forward argument `X` yields two: the
    # RESIDUAL, under the same path and in the same place (`inputs.cell` stays `inputs.cell`), and
    # its GRADIENT, under `grad_for_<path>` and in the mirror group -- `inputs` -> `grad_of_inputs`,
    # `outputs` -> `grad_of_outputs`.
    #
    # THE GROUP NAME ALWAYS DENOTES THE ROLE IN THE FORWARD, and that is what removes the knot: the
    # backward READS `grad_of_outputs` and WRITES `grad_of_inputs`, because an input cotangent is
    # what we differentiate and an output cotangent is what we receive. Nothing more to remember.
    #
    # The PATHS do not change from one step to the next: that is what lets `bwd_input_exceptions`
    # below make do with `e` and `grad_for_ + e`, exactly as before the groups.
    kwargs = {}
    bwd_groups = None if ca.groups is None else {}

    def _place( group, member, path, obj ):
        kwargs[ path ] = obj
        if bwd_groups is not None:
            bwd_groups.setdefault( group, {} )[ member ] = path

    # ( group, member, path, node ) -- one entry per forward argument, grouped or not.
    if ca.groups is None:
        flat_args = [ ( None, name, name, arg ) for name, arg in ca.args.items() ]
    else:
        flat_args = [ ( g, m, path, ca.args[ g ].attributes[ m ] )
                  for g, members in ca.groups.items() if g in ca.args
                  for m, path in members.items() if m in ca.args[ g ].attributes ]

    for group, member, path, arg in flat_args:
        if not hasattr( arg, "inst" ):
            continue
        # SCRATCH: the backward gets a FRESH writable buffer under the same name (an output of the
        # backward call), NOT the forward's transient per-thread values as a residual. The body
        # re-derives into it whatever it needs (a re-sort, a rebuilt cell). Capacity resolves on its
        # own -- the thread axis is a `CtShapeVar` (static), the item axis is shared with a residual
        # it mirrors, and an aggregate's counts are the primal's, already grown to what fits.
        if path in ca.scratch_paths:
            _place( group, member, path, _fresh_scratch( arg.inst ) )
            output_paths.append( path )
            continue
        residual, grad = _build( arg.inst, "grad_for_" + path )
        _place( group, member, path, residual )
        _place( None if group is None else f"grad_of_{ group }", member,
                "grad_for_" + path, grad )

    # the backward runs as an ordinary forward whose body is our backward one -- of the same kind,
    # so a `FfiCode` scaffolds it over the residual+gradient arguments just as it did the
    # forward over the primal ones.
    bwd_kernel = code.for_backward()
    # the forward's `input_exceptions` hold for the backward too: they say what THIS KERNEL has no
    # business touching (`Cell.measure` never reads the H-representation), and the adjoint of a body
    # that does not read something does not read it either. Passing them on is not an optimization
    # -- an excluded member is not a residual, so nothing threaded its value through the transform,
    # and binding it anyway means reading whatever the live attribute happens to hold, which under
    # a `linearize` is a tracer of the forward's own trace (see `_grad_shapevar` on the same hazard).
    # ... under BOTH names: a forward argument `X` becomes the pair `X` (the residual) and
    # `grad_for_X`, and for a member with nothing to differentiate the two are the SAME object at
    # two paths. Excluding only one of them would bind the other.
    bwd_input_exceptions = [ p for e in ca.input_exceptions
                               for p in ( e, "grad_for_" + e ) ]
    # ... and so do the forward's OUTPUT exceptions, but only those inside a SCRATCH: that scratch
    # is re-declared as a whole here, so without them every member it carved out (a `Cell`'s face
    # lattice in 2D, say) would come back as an output to allocate -- for a count that was never
    # given one, precisely because nothing allocates it. Elsewhere they are moot: a member excluded
    # from the forward's outputs is a residual or unbound, neither of which this call allocates.
    bwd_output_exceptions = [ e for e in ca.output_exceptions
                                if any( e == p or e.startswith( p + "." ) for p in ca.scratch_paths ) ]
    # THE DOMAIN DECLARED BY THE FORWARD HOLDS FOR THE BACKWARD. When the batch came from an
    # aggregate, it crossed over by itself: the aggregate is a RESIDUAL, so the backward received it
    # with its axes. A domain declared by the call ( `nb_items` ) is carried by no object -- that is
    # its whole point -- so it only crosses over if we pass it. Without that the adjoint ran on ONE
    # item and returned a wrong derivative, without any warning.
    bwd_ca = CallArgsAnalysis( kwargs, device, output_attributes = output_paths,
                               output_attribute_exceptions = bwd_output_exceptions,
                               input_exceptions = bwd_input_exceptions,
                               groups = bwd_groups, call_name = prefix + "bwd",
                               nb_items = next( iter( ca.declared_batch_size.values() ), None ) )
    bwd_outputs, bwd_results = run( bwd_kernel, bwd_ca, device, prefix + "bwd_" )

    result_of = { id( o.inst ): r for o, r in zip( bwd_outputs, bwd_results ) if hasattr( o, "inst" ) }

    grads = []
    for t in inputs:
        gobj = grad_obj_of.get( id( t.inst ) ) if hasattr( t, "inst" ) else None
        if gobj is not None and id( gobj ) in result_of:
            grads.append( result_of[ id( gobj ) ] )
        else:
            # non-float, or a non-perturbed float primal: Jax's own symbolic-zero handling for a
            # `custom_vjp` bwd converts a bare `None` leaf into the right zero cotangent -- no
            # `float0`/dtype ceremony needed, and it is well-defined for an integer primal too.
            grads.append( None )
    return tuple( grads )
