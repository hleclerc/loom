#pragma once

#include <loom/support/algorithms/apply_values.h>
#include <loom/support/common_macros.h>
#include <type_traits>

namespace sdot {

/// LA FORME NOYAU d'un argument de `run_parallel` : ce que le kernel reçoit à la place de la
/// valeur hôte -- la même vue, retypée dans la zone mémoire du kernel (voir `Ptr.h`), un agrégat
/// rebâti membre à membre, un scalaire tel quel.
///
/// Une VALEUR, pas une continuation. La forme `make_available( queue, io, arg, cont )` d'avant
/// (à la SYCL : `cont( ... )` pour garder vivants des tampons de transfert) empilait un lambda par
/// membre de chaque agrégat, et nvcc (EDG) ne sait plus déduire le type de retour au fond d'une
/// telle chaîne (« cannot deduce the return type », sur `measures_bwd`). Aucun de nos devices ne
/// transfère (la donnée est déjà là où le kernel tourne, `transfer_cost_per_byte == 0`) : la forme
/// noyau se calcule donc directement. Un device qui transférerait rendrait une vue sur sa copie et
/// en confierait la vie au `QueueEvent`.
///
/// Un type dit lui-même sa forme noyau par `kernel_form( queue, io )` ; un tuple la prend membre
/// à membre ; un arithmétique est sa propre forme.
auto kernel_form( auto &&queue, auto &&io_category, auto &&arg ) {
    using T = DECAYED_TYPE_OF( arg );
    if constexpr ( requires { arg.kernel_form( queue, io_category ); } )
        return arg.kernel_form( queue, io_category );
    else if constexpr ( requires { apply_values( FORWARD( arg ), []( auto &&... ) {} ); } )
        return apply_values( FORWARD( arg ), [&]( auto &&...values ) {
            return T::make_variant( kernel_form( queue, io_category, FORWARD( values ) )... );
        } );
    else if constexpr ( std::is_arithmetic_v<T> )
        return arg;
    else
        return arg.theres_no_kernel_form_func();
}

/// l'ancienne forme, pour les appelants qui passent une continuation
auto make_available( auto &&queue, auto &&io_category, auto &&arg, auto &&cont ) {
    return cont( kernel_form( queue, io_category, FORWARD( arg ) ) );
}

} // namespace sdot
