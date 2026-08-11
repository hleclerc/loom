class AbstractFfiCode:
    """The C++ a call runs, behind one question: `code_for( "fwd" | "bwd", call_args_analysis )`.

    A subclass may hand the body back VERBATIM (`FfiCode`) or GENERATE the scaffold around it from
    what the call turns out to be (`FfiCodeParallel` wraps the body in a `run_parallel` over the
    call's arguments). Either way the rest of the pipeline only ever asks `code_for`, so it never
    has to know which -- nor how much of the source was written by hand."""

    def code_for( self, code_type: str, call_args_analysis ) -> str:
        raise NotImplementedError


class FfiCode( AbstractFfiCode ):
    """A C++ body, plus what the CALL must know to run it -- today: its batch axes and includes.

    `batch_axes` is what a `vmap` adds: one named axis per mapping (`vmap_0`, `vmap_1`, ...).
    It belongs here and not to the arguments because it is what changes the KERNEL: it is the
    shape of `global_batch_indices`, hence the arity of the `batch_index` the body is handed.
    WHICH arguments carry which axis is another matter (a mapping may leave an argument out) and
    belongs to the call's arguments.

    `includes` are extra headers the body needs (the free functions it calls, e.g.
    `sdot/Cell/init_full.h`). They are emitted after the runtime's own includes, so the body can
    lean on everything the generated source already brings in.

    `fwd_code` does not change when an axis is added -- that is the whole point (a batch index is
    an ordinary index, and an empty one is a no-op). What changes is the generated source around
    it, so the derived code compiles to a target of its own, for free (the target name is a hash
    of the source).
    """

    def __init__( self, fwd_code, bwd_code = "", name = "", batch_axes = (), includes = (),
                  thread_cap = None, group_size = None, local_mem_elems = None,
                  fwd_setup_code = "", bwd_setup_code = "" ) -> None:
        self._code = dict( fwd = fwd_code, bwd = bwd_code )
        # a plain C++ STATEMENT emitted verbatim BEFORE the scaffolded call below (outside the
        # per-item lambda, in the handler's own scope -- where a call's aggregate args are already
        # declared, see `_render_call`'s `decls`). For a one-time, pre-launch step a per-item body
        # cannot express: e.g. zeroing a buffer several items will ATOMICALLY accumulate into next
        # (see `OtPlan1d`'s backward, `grad_for_plan.src_dist` -- a shared points-gradient buffer,
        # accumulated across every angle, that Jax does not guarantee starts zeroed). Empty by
        # default (a no-op statement position, nothing to run).
        self._setup_code = dict( fwd = fwd_setup_code, bwd = bwd_setup_code )
        self.batch_axes = tuple( batch_axes )
        self.includes = tuple( includes )
        self.name = name
        # cap on the number of work-items (work-GROUPS, if `group_size` is set) a `FfiCodeParallel`
        # launches (see `code_for`): with it a body's PER-THREAD (or per-group) scratch is sized on
        # concurrent workers, not items, so a huge batch does not blow up memory. A C++ EXPRESSION
        # string, evaluated at the `run_parallel` call site (where the mapped args are in scope) --
        # e.g. `"sorted_indices.shape( 0 )"`, the RUNTIME extent of the scratch's thread axis. Kept
        # an expression, NOT a baked int, so the same compiled kernel serves every machine/RAM (the
        # count is read at run time, never in the source hash). `None` = one work-item per item (the
        # default). Carried through `for_backward` / `with_batch_axis` so the backward and every
        # `vmap` keep the same cap.
        self.thread_cap = thread_cap
        # opt into a SECOND level of SYCL parallelism: each launched work-item becomes a work-GROUP
        # of `group_size` cooperating work-items (an `nd_range` launch), instead of one lone
        # work-item per `thread_cap` slot. Also a C++ EXPRESSION (same runtime-not-baked reasoning as
        # `thread_cap`), e.g. `"num_local.shape( 0 )"`. Requires `thread_cap` (the cap becomes a cap
        # on concurrent GROUPS, same RAM-budget meaning as before -- cooperating doesn't shrink the
        # O(n)-per-concurrent-item scratch, it just puts more workers on each one). `None` (default)
        # = no group, the plain per-item work-item scaffold (what every non-cooperative kernel uses).
        self.group_size = group_size
        # size (element count) of the raw `int32` local-memory scratch (`local_scratch`) the body
        # gets when `group_size` is set -- another C++ EXPRESSION, typically composed from
        # `group_size`'s own expression (e.g. `f"{group_size} * 256"` for a per-work-item histogram
        # row of 256 buckets). Meaningless without `group_size`.
        self.local_mem_elems = local_mem_elems

    def code_for( self, code_type: str, call_args_analysis ) -> str:
        return self._code[ code_type ]

    def has_code_for( self, code_type: str ) -> bool:
        """Whether there is a body for this direction. A `bwd` is what makes a call differentiable."""
        return bool( self._code.get( code_type ) )

    def for_backward( self ):
        """The backward as an ordinary FORWARD call: a code object of the SAME kind whose forward
        body is our backward one. Run through the normal path, so a `FfiCodeParallel` scaffolds
        the backward exactly as it does the forward (see `_call_backward`)."""
        return type( self )( self._code[ "bwd" ], name = ( self.name or "sdot" ) + "_bwd",
                             includes = self.includes, thread_cap = self.thread_cap,
                             group_size = self.group_size, local_mem_elems = self.local_mem_elems,
                             fwd_setup_code = self._setup_code[ "bwd" ] )

    def with_batch_axis( self ):
        """The same code, mapped over one more axis: what a `vmap` runs. The name is derived from
        how many axes are already there, so a nested `vmap` gets a fresh one, deterministically.
        `type( self )` keeps the subclass, so a `FfiCodeParallel` stays one under `vmap`."""
        name = f"vmap_{ len( self.batch_axes ) }"
        return name, type( self )( self._code[ "fwd" ], self._code[ "bwd" ], self.name,
                                   self.batch_axes + ( name, ), self.includes,
                                   thread_cap = self.thread_cap, group_size = self.group_size,
                                   local_mem_elems = self.local_mem_elems,
                                   fwd_setup_code = self._setup_code[ "fwd" ],
                                   bwd_setup_code = self._setup_code[ "bwd" ] )


class FfiCodeParallel( FfiCode ):
    """A body run over EVERY argument of the call, in parallel: `code_for` here is not the body
    verbatim but a `run_parallel` scaffold GENERATED around it, from the arguments the call turns
    out to have.

    So the caller writes only what happens per item -- `cell( batch_index ).init_full();` -- and
    the boilerplate is filled in from `call_args_analysis`: the lambda's parameters (one per
    argument, plus the `batch_index`), and the `<arg>_io, <arg>` pairs `run_parallel` maps over
    (an io policy or tag, then the value). Add an argument to the call and it appears in both,
    without the body changing.

    Three names are RESERVED (injected by the scaffold, not call arguments): `batch_index` (the
    item's multi-index), `thread_index` (the work-item number, 0..nb_threads-1, stable over the
    strided loop -- index a PER-THREAD scratch with it) and `nb_threads` (their count). A body
    indexes `scratch( thread_index )` to get a row exclusive to its work-item, independent of the
    item count; a body that does not need them just ignores the two `auto` params. A call kwarg
    must not be named `thread_index`/`nb_threads`/`batch_index` (they would shadow the injected
    params) -- the scaffold OWNS `nb_threads`, so a call reads it from the body, never passes it.

    When `group_size` is set, a DIFFERENT set of names is reserved instead of `thread_index`/
    `nb_threads`: `group_index` (the work-GROUP's number, 0..nb_groups-1, stable over the strided
    loop -- the group-level analogue of `thread_index`, index a PER-GROUP scratch with it, e.g.
    `scratch( group_index )`), `local_index` (the work-item's rank WITHIN its work-group,
    0..local_size-1), `local_size` (their count), `group` (the raw `sycl::group<1>` -- for
    `group_barrier( group )`, passed through undressed on purpose: this is the second SYCL
    parallelism level, exposed directly rather than wrapped), `local_scratch` (a raw `int32`
    `local_accessor` view, sized by `local_mem_elems`, SHARED by every work-item of the group --
    race-free access is the body's own responsibility, via `local_index`/`group_barrier`, exactly
    like real SYCL local memory) and `sub_group` (the raw `sycl::sub_group` for THIS work-item, a
    THIRD, warp-granularity level of cooperation -- e.g. `sycl::group_broadcast( sub_group, ... )`
    to share a value among a warp's lanes only, cheaper than a full `local_scratch` round-trip when
    the sharing is warp-local. Reachable only off the original `nd_item`, not off `group`, hence
    passed as its own reserved name instead of derived by the body). `batch_index` keeps its
    meaning: which item (e.g. angle) this GROUP was assigned, one work-group per item now instead
    of one work-item -- since a group's
    `group_index` (hence its scratch row) is REUSED across every item it strides over, a body MUST
    end with a `group_barrier` after its last read of that scratch, so a work-item that finishes
    early does not race ahead into the next item and overwrite the row while a slower sibling
    (e.g. the `local_index==0` leader finishing a sequential sweep) is still reading it.
    """

    def code_for( self, code_type: str, call_args_analysis ) -> str:
        body = self._code[ code_type ]
        names = list( call_args_analysis.args )
        mapped = ", ".join( call_args_analysis.args[ n ].cpp_run_parallel_pair() for n in names )

        # emitted BEFORE the scaffolded call, in the handler's own scope (see `FfiCode.__init__`'s
        # docstring on `_setup_code`) -- a one-time, pre-launch statement, not part of the per-item
        # lambda below.
        setup = self._setup_code[ code_type ]
        setup = ( setup + "\n" ) if setup else ""

        # Cooperative (nd_range): a work-GROUP per item, `group_size` work-items inside it sharing
        # `local_scratch`. `with_group_kernel` wraps the lambda with the `max_nb_threads` (cap on
        # concurrent GROUPS -- same RAM-budget meaning `thread_cap` always had),`group_size` and
        # `local_mem_elems` hooks `run_parallel` reads (see run_parallel.h/.cxx). All three are C++
        # EXPRESSIONS read at RUN TIME (never baked ints), so one compiled kernel (`generic`/SSCP)
        # serves every machine.
        if self.group_size is not None:
            params = ", ".join( [ "auto batch_index", "auto group_index", "auto local_index", "auto local_size",
                                  "auto group", "auto local_scratch", "auto sub_group" ] + [ f"auto { n }" for n in names ] )
            return ( f"{ setup }"
                     "run_parallel(\n"
                     "    queue,\n"
                     "    global_batch_indices,\n"
                     f"    with_group_kernel( { self.thread_cap }, { self.group_size }, { self.local_mem_elems }, []( { params } ) {{\n"
                     f"        { body }\n"
                     "    } ),\n"
                     f"    { mapped }\n"
                     ");" )

        params = ", ".join( [ "auto batch_index", "auto thread_index", "auto nb_threads" ]
                            + [ f"auto { n }" for n in names ] )

        # No cap: the plain lambda, one work-item per item (`run_parallel` then launches `nb_items`
        # threads). This is the default and what every non-scratch kernel uses.
        if self.thread_cap is None:
            return ( f"{ setup }"
                     "run_parallel(\n"
                     "    queue,\n"
                     "    global_batch_indices,\n"
                     f"    []( { params } ) {{\n"
                     f"        { body }\n"
                     "    },\n"
                     f"    { mapped }\n"
                     ");" )

        # Capped: `with_max_threads` wraps the lambda with a `max_nb_threads` hook that `run_parallel`
        # reads to launch at most `min( nb_items, cap )` work-items, each striding over its share of
        # items and reusing its `scratch( thread_index )` row. `thread_cap` is a C++ EXPRESSION read at
        # RUN TIME (e.g. `sorted_indices.shape( 0 )`, the scratch's thread-axis extent), so the same
        # compiled kernel serves every machine -- nothing about the thread count enters the source. The
        # wrapper lives at namespace scope because a local struct cannot carry the templated hook the
        # kernel needs (see run_parallel.h).
        return ( f"{ setup }"
                 "run_parallel(\n"
                 "    queue,\n"
                 "    global_batch_indices,\n"
                 f"    with_max_threads( { self.thread_cap }, []( { params } ) {{\n"
                 f"        { body }\n"
                 "    } ),\n"
                 f"    { mapped }\n"
                 ");" )
