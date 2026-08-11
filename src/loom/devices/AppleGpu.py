from .Device import Device
import os


class AppleGpu( Device ):
    """
    Apple GPU device.

    Two regimes coexist behind this single device:

    * Torch (MPS): tensors live on the Metal GPU; bindings receive device pointers directly.
    * JAX: the graph runs on the *CPU* XLA backend (jax-metal does not dispatch custom calls,
      and is incompatible with recent jax). The generated binding is an Obj-C++ `.mm` whose
      XLA FFI handler launches a Metal compute kernel on the unified-memory pointers it receives.
      So `driver_version_for_jax` resolves to the CPU jax device while the GPU work happens
      inside the binding. `is_apple_gpu` is what drives the Metal codegen branch in JaxDriver.
    """

    @property
    def name( self ):
        return "metal"

    @property
    def signature( self ):
        return "metal"

    @property
    def codegen_target( self ):
        return "metal"

    @property
    def cpp_queue_type( self ):
        # not acpp-reachable (see `acpp_reachable`): no SYCL queue targets Metal.
        raise NotImplementedError

    @property
    def cpp_memory_space( self ):
        raise NotImplementedError

    @property
    def is_apple_gpu( self ):
        return True

    def _hw_thread_cap( self, nb_local_bytes_per_thread = 0, **kwargs ):
        n = os.cpu_count() or 1
        if nb_local_bytes_per_thread > 0:
            try:
                total = os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
                n = min( n, int( total * self.scratch_ram_fraction ) // nb_local_bytes_per_thread )
            except ( AttributeError, ValueError ):
                pass
        return n

    def __repr__( self ) -> str:
        return "AppleGpu"

    def driver_version_for_jax( self, devices ):
        # The Metal work is launched from inside the binding; the JAX arrays themselves stay on
        # the CPU XLA backend (jax-metal is unavailable / cannot dispatch the custom call).
        # Prefer a real METAL device if one happens to be present, otherwise fall back to CPU.
        try:
            return devices( "METAL" )[ 0 ]
        except RuntimeError:
            return devices( "cpu" )[ 0 ]
