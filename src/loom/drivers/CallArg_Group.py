from .CallArg_Aggregate import CallArg_Aggregate
from .IoCategory import IoCategory
from .CallArg import CallArg


class CallArg_Group( CallArg_Aggregate ):
    """A GROUP OF ARGUMENTS: `inputs`, `outputs`, `scratch`, `grad_of_inputs`, `grad_of_outputs`.

    It is an AGGREGATE whose members are GIVEN instead of being found on a python object. All
    that an aggregate already knows how to do applies to it as is -- the struct of views, the io
    policy member by member ( `<Type>_io` ), `kernel_form`, the headers to include -- which is why
    grouping does not need a second machinery next to the first.

    WHAT IT BUYS. The call's arguments used to occupy the FIRST level of `args`, which they
    shared with the call's settings ( `name`, `output_attributes`, ... ): a kernel could not
    have an argument called `name`. Under groups, the first level holds nothing but the groups
    themselves -- an argument called `name`, or even `inputs`, lives at `args.inputs.name` and
    can no longer collide with anything.

    THE NAME AND THE PATH SPLIT HERE, which is what makes `loom.mutable` possible: a mutable
    argument is TWO buffers carrying the SAME C++ name ( `inputs.temperature` and
    `outputs.temperature` ) under TWO distinct paths ( `temperature_input`, `temperature_output`
    -- what capacities and outputs are named on the python side ). `make_CallArg` already took
    the two separately; all that was left was to use it.

    A group is not a buffer: it has no io category of its own, its members carry theirs ( what
    the generated policy expresses member by member ). Hence `IoCategory.INPUT` below, which
    nobody reads for a node without a buffer.
    """

    def __init__( self, call_args_analysis, name, type_name, children ) -> None:
        # NOT `super().__init__`: the one in `CallArg_Aggregate` looks for its members on a python
        # object ( `attributes_of( inst )` ), and a group has none -- its members are already
        # `CallArg`s, built by the caller. So we directly set what the base class emits next:
        # the name, the type, the members.
        CallArg.__init__( self, IoCategory.INPUT, name )
        self.type_name = type_name
        # a group comes from no object of the caller's: nothing to re-derive for the adjoint from
        # it ( `_call_backward` is what rebuilds the backward groups, member by member ).
        self.inst = None
        self.attributes = dict( children )
        # the DECLARED axes are declared by the members, each answers for itself.
        self.declared_axes = []
