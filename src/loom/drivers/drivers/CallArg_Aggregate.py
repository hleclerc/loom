from .CallArg import CallArg


def _cpp_define( name, body ):
    """A `#define name body` where `body` may span several lines: each line but the last gets a
    trailing backslash. `body` must carry no `//` comments -- the escaped newline would swallow
    the code that follows (the generated methods use none)."""
    lines = body.split( "\n" )
    if len( lines ) == 1:
        return f"#define { name } { lines[ 0 ] }"
    return "\n".join( [ f"#define { name } \\" ] + [ l + " \\" for l in lines[ :-1 ] ] + [ lines[ -1 ] ] )


class CallArg_Aggregate( CallArg ):
    """Default lowering of an aggregate instance: one child `CallArg` per declared field.

    Reached when a type provides no `make_CallArg` of its own (a plain `@aggregate`). A field
    whose type is itself an `@aggregate` lands here too: aggregates nest. A field that is a
    mere declaration (an `Axis`) lowers to nothing and is skipped.

    In C++ the aggregate becomes a *template* named after the Python class (`type_name`), one
    definition per class, and each member is exactly ONE type parameter:

        template<class T_vertex_positions, class T_nb_vertices, class T_nb_dims>
        struct Cell { T_vertex_positions vertex_positions; ... };

        auto cell = Cell{ tensor_view( ... ), make_shape_var_view( ... ), Ct<SI,2>{} };

    So *what a member is* is decided per call -- a `TensorView`, a `NoneTensor` when the
    attribute is unbound, a `ZeroTensor` when it is symbolically zero -- while the definition
    stays one template. The same `Cell` can then appear twice in a call with `dim = 2` and
    `dim = 3`, or with one of its tensors bound and the other not, and the C++ body still reads
    a member's scalar type, rank, axis names or compile-time value straight off its type.

    We never SPELL an instantiation, though: C++20 deduces it from the members we build it with
    (aggregate CTAD), here and inside `make_available` -- where the deduced type differs anyway,
    the kernel's views living in another memory space. A nested aggregate is no exception: it is
    one type parameter too, deduced in turn from its own members. Nothing is forwarded, prefixed,
    or written twice.

    What the struct does NOT hold is io: which members a kernel reads and which it writes is a
    property of that `run_parallel`, not of the data -- chain two kernels over one `Cell` and the
    answer changes. So a second template is generated beside it, the io POLICY (`Cell_io`): the
    same shape, holding a category per member. It is passed where `run_parallel` already expects
    a category, and the struct reads its own members out of it (`io_of_<member>`); a plain tag
    (`InpList()`) still works and then holds for every member. Python emits the policy THIS call
    implies as `<arg>_io` -- a default the body may ignore.
    """

    attributes : dict[ str, CallArg ]

    def __init__( self, call_args_analysis, path, name, inst ) -> None:
        # an aggregate is not itself a buffer; its category is the one its children inherit by
        # path (naming an aggregate in `output_attributes` covers everything below it).
        super().__init__( call_args_analysis.io_category( path, True ), name )

        self.type_name = call_args_analysis.cpp_type_name( type( inst ) )
        self.inst = inst

        self.attributes = {}
        # the axes the aggregate DECLARES (an `Axis` lowers to no CallArg): kept so their
        # `DEFINE_AXIS` is emitted whether or not a tensor of this call references them.
        self.declared_axes = []
        for name_attr, attr in call_args_analysis.attributes_of( inst ).items():
            self.declared_axes += getattr( attr, "cpp_axis_names", lambda: [] )()
            ca = call_args_analysis.make_CallArg( f"{ path }.{ name_attr }", name_attr, attr )
            if ca is not None:
                self.attributes[ name_attr ] = ca

    def _clone( self, mapping ):
        res = super()._clone( mapping )
        res.attributes = { n: c._clone( mapping ) for n, c in self.attributes.items() }
        return res

    def children( self ):
        # our members ARE the subtree: a generic walk descends through here, and every per-node
        # concern (axes, attrs) folds over the walk rather than recursing here by hand.
        return list( self.attributes.values() )

    def _fields( self, method ):
        """Child CallArgs implementing `method` (polymorphic filter, no type tests): struct
        members, brace-init entries and writable-back values are each such a subset."""
        return [ ca for ca in self.attributes.values() if hasattr( ca, method ) ]

    @property
    def io_type_name( self ):
        """C++ name of the io POLICY generated beside the struct (`Cell` -> `Cell_io`)."""
        return f"{ self.type_name }_io"

    # -- driver-agnostic C++ (the same for every driver) --
    # the support the struct's own methods lean on: `make_available` / `transfer_cost` (found by
    # ADL at instantiation, but their declarations must be visible) and `Ct` for the transfer-cost
    # fold. The MEMBERS are template parameters, so no container header is needed here -- a
    # `TensorView` vs a `NoneTensor` is decided at the instantiation site, not in the definition.
    _CPP_SUPPORT_INCLUDES = ( "sdot/support/common_types.h", "sdot/support/Ct.h",
                              "sdot/support/kernels/make_avaiable.h",
                              "sdot/support/kernels/transfer_cost.h" )

    def cpp_includes( self ):
        """The headers the call needs for us: our OWN struct's header, plus -- whether that header
        is hand-written or generated -- every NESTED aggregate member's, recursively. A member is a
        type OUR struct instantiates (by CTAD, at the call site) whether the struct wrapping it is
        manual or generated, so a hand-written header is no reason to stop descending: skipping the
        recursion here left a manual `OtPlan1d.h` compiling with `SumOfDiracs`/`Image` undeclared,
        even though its macro-generated members line does spell them.

        `self.attributes` was built from the RUNTIME instance (`attributes_of( inst )` reads off
        `type( inst )`, not the field's declared annotation type), so a member declared as the
        abstract `Distribution` but holding an `Image` still contributes `Image`'s own header here,
        not one for `Distribution` (which may not even exist).

        Two modes for OUR OWN header, chosen by whether the user provides `sdot/<Name>.h`:

        * they do -> that MANUAL header is the struct: it drops in our macros and adds methods of
          its own (the real `Cell`, with `init_as_hypercube`). We use it as-is.
        * they don't -> we generate the struct WHOLE (`_emit_full_header`): the same wrapper, with
          no hand-written methods. So a plain aggregate needs no C++ at all.

        Either way we first emit the macros the struct is built from (`_emit_macros_header`)."""
        from ..compilation.generated_headers import manual_header
        self._emit_macros_header()
        own = manual_header( f"sdot/{ self.type_name }.h" ) or self._emit_full_header()
        incs = [ own ]
        for child in self.attributes.values():
            for inc in getattr( child, "cpp_includes", lambda: () )():
                if inc not in incs:
                    incs.append( inc )
        return incs

    def _emit_full_header( self ):
        """Generate the struct WHOLE, for an aggregate the user did not hand-write a header for:
        the very wrapper they would write, minus any methods of their own -- it just turns the
        macros into an actual `struct`. Returns the include path."""
        from ..compilation.generated_headers import shared_header
        lines = [ "#pragma once", "",
                  f'#include "sdot/generated/aggregates/{ self.type_name }.h"' ]
        lines += [ "", "namespace sdot {",
                   f"SDOT_TEMPLATE_DECL_FOR_{ self.type_name }",
                   f"struct { self.type_name } {{ SDOT_ATTRIBUTES_OF_{ self.type_name } }};",
                   "}", "" ]
        return shared_header( f"sdot/generated/aggregates/{ self.type_name }_full.h",
                              "\n".join( lines ) + "\n" )

    def _emit_macros_header( self ):
        """Write the generated header the manual struct is built from. Rather than emit the struct
        (an opaque generated type nothing can extend or autocomplete), we emit two macros the user
        drops into a struct of their own:

            SDOT_TEMPLATE_DECL_FOR_Cell   ->  template<class T_nb_vertices, ...>
            SDOT_ATTRIBUTES_OF_Cell       ->  the members + the generated methods

        so `sdot/Cell.h` reads:

            SDOT_TEMPLATE_DECL_FOR_Cell
            struct Cell { SDOT_ATTRIBUTES_OF_Cell   void init_full() { ... } };

        The user's struct is the one the call instantiates (the generated brace-init deduces it by
        CTAD), and it can carry methods of its own -- an autocompleted `init_full`, a body the
        kernel calls. It must live in `namespace sdot`: the generated methods rebuild it there
        (`::sdot::Cell`, see `_cpp_make_available`).

        The io policy `Cell_io` is NOT customized, so it is emitted whole beside the macros. The
        header also pulls in the support the methods need and the axes the members name, so the
        manual header that includes it can spell them (`init_full` names `dim`, `num_vertex`)."""
        from ..compilation.generated_headers import shared_header
        from ..tensor.AbstractAxis import AbstractAxis

        for child in self._fields( "_emit_macros_header" ):   # nested aggregates get theirs too
            child._emit_macros_header()

        # every axis the aggregate declares, plus any a member borrows from elsewhere -- all get a
        # `DEFINE_AXIS`, so the manual struct that includes us can spell any of them.
        axis_names = list( self.declared_axes )
        for ca in self.attributes.values():
            for axis_name in getattr( ca, "axis_names", () ):
                if axis_name not in axis_names:
                    axis_names.append( axis_name )
        axis_incs = [ AbstractAxis.cpp_shared_header( a ) for a in axis_names ]

        fields = self._fields( "cpp_member" )
        params_without_type = ", ".join( c.cpp_tpl_name() for c in fields )
        params = ", ".join( c.cpp_tpl_param() for c in fields )

        lines = [ "#pragma once", "" ]
        lines += [ f'#include "{ inc }"' for inc in self._CPP_SUPPORT_INCLUDES ]
        lines += [ f'#include "{ inc }"' for inc in axis_incs ]
        lines += [ "", f"#define SDOT_TEMPLATE_DECL_FOR_{ self.type_name } template<{ params }>", "" ]
        lines += [ "", f"#define SDOT_TEMPLATE_ARGS_FOR_{ self.type_name } { params_without_type }", "" ]
        # the io policy, generated whole -- in `namespace sdot`, where its struct lives.
        lines += [ "namespace sdot {", self.cpp_io_struct_def(), "}", "" ]
        lines += [ _cpp_define( f"SDOT_ATTRIBUTES_OF_{ self.type_name }", self._cpp_attributes( fields ) ) ]
        content = "\n".join( lines ) + "\n"

        shared_header( f"sdot/generated/aggregates/{ self.type_name }.h", content )

    def _cpp_attributes( self, fields ):
        """The struct BODY -- members and the generated methods -- with no `struct { }` wrapper
        and no template line, so it can be dropped into a struct the user declares."""
        members = "\n".join( "    " + c.cpp_member() for c in fields )
        parts = [ members, self._cpp_call_op( fields ), self._cpp_io_of( fields ),
                  self._cpp_transfer_cost( fields ), self._cpp_make_available( fields ) ]
        return "\n\n".join( p for p in parts if p )

    def cpp_io_struct_def( self ):
        """The io policy: the same shape as the struct, holding a category per member.

        An io category is a property of a `run_parallel`, NOT of the data -- chain two kernels
        over one object and they may read and write different parts of it. So the struct knows
        nothing about io: it is handed a policy (this) or a plain tag, and reads what it needs."""
        fields = self._fields( "cpp_member" )
        members = "\n".join( "    " + c.cpp_member() for c in fields )
        params = ", ".join( c.cpp_tpl_param() for c in fields )
        prefix = f"template<{ params }>\n" if params else ""
        # declared, not derived: a base class would cost us the CTAD (see `is_io_policy`).
        return ( f"{ prefix }struct { self.io_type_name } {{\n"
                 f"    static constexpr bool is_io_policy = true;\n{ members }\n}};" )

    # -- selecting axes, as on a tensor --
    def _cpp_call_op( self, fields ):
        """`cell( batch_index )` = the same aggregate, each member indexed. So the two spellings
        below are one and the same, which is the point:

            cell( batch_index ).nb_vertices = 1;
            cell.nb_vertices( batch_index ) = 1;

        A member selects what it carries and lets the rest through (a batch index is OPTIONAL, see
        AxisNames.h), so members mapped along different axes -- or along none, or not even tensors
        -- all take the same index. The result is another instantiation of the same template
        (the members lost an axis), deduced, not spelled; the qualified name reaches the template
        rather than the current instantiation (see `_cpp_make_available`)."""
        values = ", ".join( f"{ c.name }( index... )" for c in fields )
        return ( "    auto operator()( const auto &...index ) const {\n"
                 f"        return ::sdot::{ self.type_name }{{ { values } }};\n"
                 "    }" )

    # -- as an argument of `run_parallel` (see support/kernels/run_parallel.h) --
    def _cpp_io_of( self, fields ):
        """How a member reads its own io category out of what the caller gave the aggregate: its
        entry in the policy, or the plain tag itself, which then holds for every member."""
        return "\n".join(
            f"    static constexpr auto io_of_{ c.name }( auto io ) {{ "
            f"if constexpr ( requires {{ io.{ c.name }; }} ) return io.{ c.name }; else return io; }}"
            for c in fields
        )

    def _cpp_make_available( self, fields ):
        opens = "\n".join(
            f"        return sdot::make_available( queue, io_of_{ c.name }( io ), { c.name }, "
            f"[&]( auto &&a_{ c.name } ) {{" for c in fields
        )
        values = ", ".join( f"a_{ c.name }" for c in fields )
        # the members reaching the kernel have other TYPES than ours (a kernel memory space), so
        # what `cont` receives is another instantiation of the same template -- deduced from the
        # members it is built with (C++20 aggregate CTAD), never spelled out. Qualified name,
        # because inside the class our own name means the CURRENT instantiation (the injected
        # class name), which is exactly the one we are NOT rebuilding. The user defines the struct
        # in `namespace sdot` (that is the convention the macros assume), so `::sdot::` reaches it.
        rebuilt = f"::sdot::{ self.type_name }{{ { values } }}"
        closes = "        " + "} );" * len( fields )
        return ( "    auto make_available( auto &&queue, auto io, auto &&cont ) const {\n"
                 f"{ opens }\n"
                 f"            return cont( { rebuilt } );\n"
                 f"{ closes }\n"
                 "    }" )

    def _cpp_transfer_cost( self, fields ):
        costs = [ f"sdot::transfer_cost( queue, io_of_{ c.name }( io ), { c.name } )" for c in fields ]
        total = "\n             + ".join( costs + [ "Ct<double,0.0>()" ] )
        return ( "    auto transfer_cost( const auto &queue, auto io ) const {\n"
                 f"        return { total };\n"
                 "    }" )

    # -- as a member of another aggregate --
    def cpp_tpl_param( self ):
        return f"class { self.cpp_tpl_name() }"

    def cpp_member( self ):
        return f"{ self.cpp_tpl_name() } { self.name };"

    def cpp_io_list( self ):
        """Our default policy, member by member -- a nested aggregate contributing its own.

        The TYPE follows the class (the shape: which members exist, common to every `Cell`); the
        VALUE follows the argument (the profile: what THIS call does with THIS object). So two
        `Cell`s of one call can have two profiles -- they are two instantiations."""
        cats = ", ".join( c.cpp_io_list() for c in self._fields( "cpp_member" ) )
        return f"{ self.io_type_name }{{ { cats } }}"

    def cpp_run_parallel_pair( self ):
        # the policy VARIABLE declared beside us (`cpp_root_decl` emits `<name>_io`): a category
        # per member, not one blanket tag for the whole aggregate.
        return f"{ self.name }_io, { self.name }"

    # -- as a ROOT argument --
    def cpp_root_decl( self, var_name ):
        """A root argument brace-inits from the buffers bound to it, and comes with `<name>_io`:
        the io policy this CALL implies (what Python knows attribute by attribute). A default,
        not a law -- the body is free to hand `run_parallel` another one, which is the whole
        point of keeping io out of the data.

        Types are DEDUCED from the members (C++20 aggregate CTAD), here and recursively for any
        aggregate held: no instantiation is ever spelled out."""
        inits = ",\n".join( "        " + c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return ( f"    auto { var_name } = { self.type_name }{{\n{ inits }\n    }};\n"
                 f"    auto { var_name }_io = { self.cpp_io_list() };" )

    # -- seeding: what an output must hold before the body runs --
    def cpp_seed_root( self, var_name ):
        seeds = [ c.cpp_seed_member( var_name ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( "    " + s for s in seeds if s )

    def cpp_seed_member( self, owner_name ):
        seeds = [ c.cpp_seed_member( f"{ owner_name }.{ self.name }" ) for c in self._fields( "cpp_seed_member" ) ]
        return "\n".join( s for s in seeds if s )

    # -- Jax FFI ABI --
    def jax_cpp_init( self ):
        # a nested aggregate brace-inits in place; its type is deduced from what it holds.
        inits = ", ".join( c.jax_cpp_init() for c in self._fields( "jax_cpp_init" ) )
        return f"{ self.type_name }{{ { inits } }}"
