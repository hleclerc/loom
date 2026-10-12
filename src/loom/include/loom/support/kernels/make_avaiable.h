#pragma once

#include <loom/support/algorithms/apply_values.h>
#include <loom/support/common_macros.h>
#include <type_traits>

namespace sdot {

/// THE KERNEL FORM of a `run_parallel` argument: what the kernel receives in place of the
/// host value -- the same view, retyped into the kernel's memory zone (see `Ptr.h`), an aggregate
/// rebuilt member by member, a scalar as is.
///
/// A VALUE, not a continuation. The earlier `make_available( queue, io, arg, cont )` form
/// (`cont( ... )` to keep transfer buffers alive) stacked one lambda per
/// member of each aggregate, and nvcc (EDG) can no longer deduce the return type at the bottom of
/// such a chain ("cannot deduce the return type", on `measures_bwd`). None of our devices
/// transfers (the data is already where the kernel runs, `transfer_cost_per_byte == 0`): the kernel
/// form is therefore computed directly. A device that did transfer would return a view on its copy and
/// entrust its lifetime to the `QueueEvent`.
///
/// A type states its own kernel form through `kernel_form( queue, io )`; a tuple takes it member
/// by member; an arithmetic type is its own form.
struct KernelFormProbe { void operator()( auto &&... ) const {} };

auto kernel_form( auto &&queue, auto &&io_category, auto &&arg ) {
    using T = DECAYED_TYPE_OF( arg );
    if constexpr ( requires { arg.kernel_form( queue, io_category ); } )
        return arg.kernel_form( queue, io_category );
    else if constexpr ( requires { apply_values( FORWARD( arg ), KernelFormProbe{} ); } )
        return apply_values( FORWARD( arg ), [&]( auto &&...values ) {
            return T::make_variant( kernel_form( queue, io_category, FORWARD( values ) )... );
        } );
    else if constexpr ( std::is_arithmetic_v<T> )
        return arg;
    else
        return arg.theres_no_kernel_form_func();
}

/// the old form, for callers that pass a continuation
auto make_available( auto &&queue, auto &&io_category, auto &&arg, auto &&cont ) {
    return cont( kernel_form( queue, io_category, FORWARD( arg ) ) );
}

} // namespace sdot
