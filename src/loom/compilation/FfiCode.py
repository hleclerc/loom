from pathlib import Path


class AbstractFfiCode:
    """The C++ that a call executes, behind two questions: `code_for( call_args_analysis )`, the
    instructions inside the handler, and `preamble_for( ... )`, what must exist at namespace
    level before it (a functor).

    There is no direction in these two questions: a BACKWARD is a full-fledged kernel,
    which the call obtains through `for_backward()` and then launches like any other forward. The
    rest of the pipeline therefore only knows a single direction."""

    def render_key( self ):
        """What the source rendered from this code can read of it: its fields, all of them (see
        `drivers/render_key.py`). A subclass that holds something that is not a plain fact makes the
        call uncacheable, which is the safe answer."""
        from ..drivers.render_key import plain_object
        return plain_object( self )

    def code_for( self, call_args_analysis ) -> str:
        raise NotImplementedError

    def preamble_for( self, call_args_analysis ) -> str:
        """Namespace-level C++ emitted before the handler. Nothing by default (a verbatim body
        brings its own)."""
        return ""

    @property
    def wants_allocator( self ) -> bool:
        """Whether the handler must receive XLA's allocator (`Scratch`). False by default: it is the
        SOURCE that is hashed to name a kernel, so binding this context without being asked
        would recompile the whole repository for a capability that nobody uses."""
        return False

    @property
    def is_handler( self ) -> bool:
        """Whether the body IS the handler -- i.e. whether IT is what launches. False by default."""
        return False


class FfiCode( AbstractFfiCode ):
    """ONE KERNEL: the C++ that the call executes, and what is needed to compile it.

    = What loom writes, and what it does not write

    `code` is the body OF THE HANDLER, copied as is. Loom writes everything around it -- and that is
    all it has to write, because that is where the pain is: the FFI wrapping, the binding of
    buffers to views, the construction of aggregates, the seeding of outputs, the adjoint on the Jax side. The
    Jax and Torch interfaces are heavy AND different; that is what we do not want to write
    twice.

    In the body, loom makes `queue` and the call's arguments available under their Python
    names. What you do with them is none of its business:

        FfiCode(
            include_roots = [ my_root ],
            includes = [ "diffusion/kernels.h" ],
            code = "diffusion::step( queue, grid, coef, next );",
        )

    The traversal, the choice of parallelism, the launch geometry then live in OUR C++ --
    an ordinary file, which compiles and is tested without loom, and which can be replaced by Kokkos,
    OpenMP or a plain loop without touching Python. `run_parallel` is the tool that
    loom offers, not an obligation that it imposes.

    = A kernel does not contain itself

    The adjoint is not a field of the forward: it is another kernel, and it is the CALL that takes
    several of them.

        loom.ffi_call(
            FfiCode( code = "mypackage::forward( queue, ... );" ),
            FfiCode( code = "mypackage::backward( queue, ... );" ),   # optional
            name = "one_step",
            ... )

    It runs on other buffers (the cotangents), does other work, and has no reason to
    want the same geometry as the forward. The NAME is on the call: it identifies the pair, prefixes
    the compiled target and groups the compilation journal.

    = Where the C++ lives

    `include_roots`: the `-I` roots of THIS kernel. A kernel knows where its headers are -- that should not
    be a module incantation (`compilation.register_include_root`) pronounced before everything
    else and with no visible relation to it.

    `includes`: the headers that the body needs, emitted after those of the runtime. `sources`: the
    C++ units that it LINKS (`"sdot/x.cpp"` or `( "sdot/x.cpp", { "DEF": "1" } )`), compiled once
    per (source, defines, compiler) and shared by all the kernels that name them.

    `prologue`: a C++ statement emitted before the body, in the same scope. A relic of the time
    when the body was per item and could not express a pre-pass; a body that is the
    handler no longer needs it.

    `allocator = True` makes it bind XLA's allocator, hence `args.allocator.view<T>( n )` in the body: memory
    sized at execution time, at a size that only the kernel knows, under `jit` as in
    eager. See `support/kernels/Scratch.h` and `tests/test_scratch_gpu.py`.

    = THE SUGAR: a body per item ( `FfiCode.per_item` )

    When the kernel's parallelism IS that of the call -- one item per batch multi-index, what
    a `vmap` produces -- loom can write the functor AND its launch, and the body is no longer
    anything but what happens for one item. Three names are then reserved: `batch_index`,
    `flat_index`, `thread_index`, `nb_threads` (or, in cooperative mode, `flat_index`, `group_index`,
    `local_index`, `local_size`,
    `group`, `local_scratch`, `sub_group`). The geometry is declared as method bodies of the generated
    functor: `max_nb_threads`, `group_size`, `local_mem_elems` -- `group_size` requires
    `local_mem_elems`, otherwise the cooperative path would be silently ignored.

    The trap, and that is why it is no longer the default: the generated domain is
    `global_batch_indices` and NOTHING ELSE. A kernel whose parallelism is not a `vmap` axis
    -- a Cartesian grid, traversed in (j, i) -- had to disguise one as such, materialize a flat
    rank and split it back up in C++. See `examples/diffusion/README.md`, friction 3.
    """

    def __init__( self, code = "", prologue = "", includes = (), sources = (), include_roots = (),
                  max_nb_threads = "", group_size = "", local_mem_elems = "",
                  allocator = False, _scaffold = False ) -> None:
        if group_size and not local_mem_elems:
            raise ValueError( "FfiCode: `group_size` without `local_mem_elems` -- `run_parallel` "
                              "only takes the cooperative path when BOTH hooks exist, so this "
                              "would be silently ignored" )
        if local_mem_elems and not group_size:
            raise ValueError( "FfiCode: `local_mem_elems` without `group_size` -- there is no "
                              "work-group to share it" )
        if ( max_nb_threads or group_size or local_mem_elems ) and not _scaffold:
            raise ValueError( "FfiCode: a launch geometry (`max_nb_threads`, "
                              "`group_size`, `local_mem_elems`) is made of METHODS of the generated "
                              "functor -- a body that launches by itself has none. Pass it to "
                              "`run_parallel` where you launch, or use `FfiCode.per_item`." )

        # WHERE THIS KERNEL'S C++ LIVES. It used to be a separate module-level call
        # (`compilation.register_include_root( ... )`), hence an incantation before anything else and
        # with no visible relation to the kernel that needs it. A kernel knows where its headers are:
        # it says so here.
        from . import register_include_root
        if not include_roots and not _scaffold:
            # THE DEFAULT ROOT: the directory of the `.py` that builds this kernel. One rule, zero
            # exceptions -- `#include "my_kernel.h"` works for a file placed next to it, which is
            # all a tutorial has to explain. A different layout is stated
            # explicitly.
            # `currentframe` and not `inspect.stack()`: the latter rebuilds ALL the source
            # context of each frame ( it reads the files ), which costs tens of
            # milliseconds -- invisible as long as a kernel was a module constant, measurable as soon
            # as it is built where it is used, that is to say at every call.
            import inspect
            frame = inspect.currentframe()
            while frame is not None:
                path = Path( frame.f_code.co_filename )
                if path.is_file() and "loom/compilation" not in path.as_posix():
                    include_roots = [ path.resolve().parent ]
                    break
                frame = frame.f_back
        for root in include_roots:
            register_include_root( root )

        self.code = code
        self.prologue = prologue
        self.sources = tuple( sources )
        self.includes = tuple( includes )
        self.include_roots = tuple( include_roots )
        self.allocator = bool( allocator )
        self._scaffold = _scaffold

        # the bodies of the hooks that `run_parallel` detects on the functor -- C++, not
        # expressions evaluated in a scope that the caller cannot see. Order matters:
        # `local_mem_elems` may call `group_size`.
        self.hooks = { hook: body for hook, body in
                       ( ( "max_nb_threads", max_nb_threads ), ( "group_size", group_size ),
                         ( "local_mem_elems", local_mem_elems ) ) if body }

    @classmethod
    def handler( cls, code = "", **kwargs ):
        """Redundant: it is what plain `FfiCode` does since a body that launches by itself
        is the NORMAL form. Kept because existing calls name it."""
        return cls( code, **kwargs )

    @classmethod
    def inline( cls, code = "", **kwargs ):
        """THE HANDLER BODY, WITHOUT THE WRAPPING -- the form to write when the C++ lives in a
        header of your own.

        A kernel that launches by itself must provide `void kernel( queue, batch_axes, args )` in an
        anonymous namespace. It is always the same envelope, it teaches nobody anything, and
        it is what forced the kernel to be put in a module constant -- far from the call
        that uses it. `inline` only takes what is inside, so the kernel fits on the line
        where it is launched:

            loom.ffi_call(
                "splats_render",
                loom.FfiCode.inline( "splats::render< 16 >( queue, batch_axes, args );",
                                     includes = [ "splats.h" ] ),
                ... )

        The anonymous namespace is not decorative: `compilation/catalogue.py` compiles each kernel
        in its own object and then links them into ONE library, so two fixed-name `kernel`s without
        internal linkage would be an ODR violation.

        THIS IS NOT `per_item`: the body remains that of the HANDLER, so it is still the body that chooses
        its domain (`queue.run_parallel( ..., args.outputs.image.domain( ... ), ... )`). `inline`
        only removes the braces. What must live at namespace level -- an `#include`, a
        functor -- goes through `includes` or through plain `FfiCode`."""
        return cls( "namespace {\n"
                    "void kernel( auto &&queue, auto &&batch_axes, auto &&args ) {\n"
                    f"    { code }\n"
                    "}\n"
                    "}\n", **kwargs )

    @classmethod
    def per_item( cls, code = "", **kwargs ):
        """THE SUGAR: a body per ITEM, and loom generates the functor AND its launch for you.

        It is convenient when the kernel's parallelism IS that of the call -- one item per
        batch multi-index, what a `vmap` produces. It is not otherwise: the generated domain
        is `global_batch_indices` and nothing else, so a kernel that wants to traverse a grid in
        (j, i) had to disguise a flat batch axis ( see `examples/diffusion/README.md`,
        friction 3 ). In that case, write the launch yourself -- that is plain `FfiCode`."""
        return cls( code, _scaffold = True, **kwargs )

    # ---- what the call asks for ( it passes the functor name, which only it knows ) ----

    @property
    def cooperative( self ):
        return "group_size" in self.hooks

    @property
    def wants_allocator( self ):
        return self.allocator

    @property
    def is_handler( self ):
        """The body IS the handler: IT is what launches. See `FfiCode.handler`."""
        return not self._scaffold

    def _params( self, names ):
        """The parameters of `operator()`: the reserved ones, then one per call argument --
        each with its own template parameter, since their kernel-side type is decided in
        C++."""
        # `flat_index`: THE FLAT RANK OF THE ITEM in the traversed domain -- "who am I?". It
        # comes from the launch's loop variable ( see `CpuQueue::call` ), so it costs
        # nothing; before, a kernel that needed it built a pretext aggregate carrying an
        # `iota` ( `examples/splats::Rangs` ), i.e. a whole tensor written then read for a number
        # that the loop already knew.
        if self.cooperative:
            reserved = [ ( "BatchIndex", "batch_index" ), ( "SI", "flat_index" ), ( "int", "group_index" ),
                         ( "int", "local_index" ), ( "int", "local_size" ), ( "Group", "group" ),
                         ( "LocalScratch", "local_scratch" ), ( "SubGroup", "sub_group" ) ]
        else:
            reserved = [ ( "BatchIndex", "batch_index" ), ( "SI", "flat_index" ),
                         ( "int", "thread_index" ), ( "int", "nb_threads" ) ]
        params = reserved + [ ( f"T_{ n }", n ) for n in names ]
        tparams = [ t for t, _ in params if t not in ( "int", "SI" ) ]
        return tparams, params

    def _hook_methods( self, names ):
        """The launch hooks, as methods of the functor. `run_parallel` calls them with the
        call's arguments (`func.max_nb_threads( args... )`, see `run_parallel.cxx`), so they
        take the same list as `operator()`, without the reserved names -- and they run ON THE
        HOST, before the launch, so no `HD`."""
        if not self.hooks:
            return ""
        tparams = ", ".join( f"class T_{ n }" for n in names )
        params = ", ".join( f"T_{ n } { n }" for n in names )
        res = ""
        for hook, body in self.hooks.items():
            res += ( f"    template<{ tparams }>\n"
                     f"    int { hook }( { params } ) const {{\n"
                     f"        { body }\n"
                     f"    }}\n" )
        return res

    def preamble_for( self, call_args_analysis, functor ) -> str:
        if not self._scaffold:
            # ALL of the user's C++, VERBATIM, at namespace level: their `#include`s, their
            # functors, their `kernel` function. Loom does not touch it -- including the anonymous
            # namespace, which the user writes themself: otherwise their `#include`s would end up INSIDE that
            # namespace, and loom's with them.
            #
            # This namespace is not decorative: `compilation/catalogue.py` compiles each kernel
            # in its own object and then links them into ONE library, so two fixed-name `kernel`s
            # without internal linkage would be an ODR violation.
            return self.code
        names = list( call_args_analysis.args )
        tparams, params = self._params( names )
        return ( f"struct { functor } {{\n"
                 f"{ self._hook_methods( names ) }"
                 f"    template<{ ', '.join( 'class ' + t for t in tparams ) }>\n"
                 f"    HD void operator()( { ', '.join( f'{ t } { n }' for t, n in params ) } ) const {{\n"
                 f"        { self.code }\n"
                 f"    }}\n"
                 f"}};\n" )

    def code_for( self, call_args_analysis, functor ) -> str:
        # emitted BEFORE the launch, in the handler's scope (see the docstring) -- a single
        # pre-pass, not a piece of the functor.
        prologue = ( self.prologue + "\n" ) if self.prologue else ""

        if not self._scaffold:
            # THE CALL, and it is fixed: `kernel( queue, batch_axes, args )`. Three things, and the
            # second is what makes this form as capable as the scaffold -- the call's batch
            # axes are a VALUE that the kernel composes with its own
            # (`batch_axes + args.<tensor>.domain()`), instead of a domain imposed on it.
            return prologue + "kernel( queue, global_batch_indices, args );"

        names = list( call_args_analysis.args )
        mapped = ", ".join( call_args_analysis.args[ n ].cpp_run_parallel_pair() for n in names )
        return ( f"{ prologue }"
                 "run_parallel(\n"
                 "    queue,\n"
                 "    global_batch_indices,\n"
                 f"    { functor }{{}},\n"
                 f"    { mapped }\n"
                 ");" )

    def inheriting( self, other ):
        """Ourselves, but taking from `other` what we have not said: `includes` and `sources`
        are what the CALL compiles and links, identical in both directions. The geometry, on the other hand, is
        never inherited -- that is the whole point of having taken it out."""
        if self.includes and self.sources:
            return self
        res = FfiCode( self.code, self.prologue, self.includes or other.includes,
                       self.sources or other.sources, self.include_roots or other.include_roots,
                       allocator = self.allocator,
                       _scaffold = self._scaffold,
                       **{ hook: self.hooks.get( hook, "" ) for hook in
                           ( "max_nb_threads", "group_size", "local_mem_elems" ) } )
        return res


class Kernels( AbstractFfiCode ):
    """WHAT A CALL LAUNCHES: a forward kernel, an optional backward kernel, and the name that identifies
    both.

    This is not a class that you write yourself: `loom.ffi_call` builds it from the
    `FfiCode`s that you pass it. It exists because the pipeline needs ONE object to ask
    "your preamble", "your body", "your adjoint", "you with one more axis" -- and
    because these last three questions are about the PAIR, not about one kernel.
    """

    def __init__( self, name, forward, backward = None, batch_axes = () ) -> None:
        if not name:
            raise ValueError( "loom.ffi_call: `name` is required -- it names the functors, prefixes "
                              "the compiled target and groups the compilation journal" )
        self.name = name
        self.forward = forward
        self.backward = backward
        self.batch_axes = tuple( batch_axes )

    # what the source rendering reads on us, directly
    @property
    def includes( self ):
        return self.forward.includes

    @property
    def sources( self ):
        return self.forward.sources

    @property
    def wants_allocator( self ):
        return self.forward.wants_allocator

    @property
    def is_handler( self ):
        return self.forward.is_handler

    def cpp_base_name( self ) -> str:
        """The call's name, rendered as a C++ identifier. Everything generated for this call
        derives from it, and that is what keeps them distinct when a catalogue links several kernels into a
        single library."""
        base = "".join( c if c.isalnum() or c == "_" else "_" for c in self.name )
        return "_" + base if base[ 0 ].isdigit() else base

    def functor_name( self ) -> str:
        """The C++ identifier of the generated functor ( `per_item` form )."""
        return f"{ self.cpp_base_name() }_kernel"

    def args_name( self ) -> str:
        """The C++ identifier of the argument aggregate ( general form )."""
        return f"{ self.cpp_base_name() }_args"

    def preamble_for( self, call_args_analysis ) -> str:
        return self.forward.preamble_for( call_args_analysis, self.functor_name() )

    def code_for( self, call_args_analysis ) -> str:
        return self.forward.code_for( call_args_analysis, self.functor_name() )

    @property
    def has_backward( self ):
        """Having an adjoint is what makes a call differentiable."""
        return self.backward is not None

    def for_backward( self ):
        """The adjoint, ready to be launched like an ordinary forward: the backward becomes the kernel
        of a pair with no backward, under a derived name. It keeps OUR batch axes -- what a `vmap`
        added to the call holds for both directions."""
        return Kernels( self.name + "_bwd", self.backward.inheriting( self.forward ),
                        batch_axes = self.batch_axes )

    def with_batch_axis( self ):
        """The same pair, mapped over one more axis: what a `vmap` launches. The axis name is
        derived from the number already present, so a nested `vmap` gets a fresh one, in a
        deterministic way."""
        name = f"vmap_{ len( self.batch_axes ) }"
        return name, Kernels( self.name, self.forward, self.backward, self.batch_axes + ( name, ) )
