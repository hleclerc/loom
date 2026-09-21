from .Device import Device
import os


class Cpu( Device ):
    def copy( self ) -> 'Device':
        return Cpu()

    @property
    def name( self ):
        return "Cpu"

    @property
    def cpp_queue_type( self ):
        return "CpuQueue"

    @property
    def cpp_memory_space( self ):
        return "CpuHostMemorySpace"

    @property
    def signature( self ):
        return "cpu"

    @property
    def codegen_target( self ):
        return "cpu"

    @property
    def is_cpu( self ):
        return True

    @property
    def cpp_queue_include( self ):
        return "loom/support/kernels/CpuQueue.h"

    @property
    def compiler( self ):
        from ..compilation.Compiler import HostCxx
        return HostCxx()

    @property
    def device_is_present( self ):
        return self.compiler.is_available()

    def catalogue_kind( self ):
        return "cpu"

    def catalogue_tags( self ):
        # every level at or below what the machine carries: a v4 machine loads a v3 catalogue
        from ..compilation.Compiler import cpu_variant, X86_LEVELS
        mine = cpu_variant()
        levels = [ level for level, _ in X86_LEVELS ] + [ "x86-64" ]
        if mine in levels:
            return [ f"cpu-{ l }" for l in levels[ levels.index( mine ): ] ]
        return [ f"cpu-{ mine }" ]

    @property
    def ffi_platform( self ):
        return "cpu"

    def __repr__( self ) -> str:
        return "Cpu"

    def _hw_thread_cap( self, nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=1 ):
        # registers managed by compiler; shared memory not applicable to CPU threads
        # both local and pinned bytes draw from host RAM
        n          = _nb_workers()
        per_thread = max( nb_local_bytes_per_thread, nb_pinned_bytes_per_thread )
        if per_thread > 0:
            usable = int( _total_host_ram() * self.scratch_ram_fraction )
            n = min( n, usable // per_thread )
        return n

    def group_size( self, **per_group_item ):
        # A group of more than one lane is, on CPU, that many system threads around a
        # `std::barrier` (see `CpuQueue.h::submit_kernel_grouped`): correct, and slow -- it exists
        # so a kernel written for GPU groups can be TESTED on CPU. Stay at the degenerate `1` (see
        # `Device.group_size`) on purpose; CPU parallelism already comes from
        # `nb_threads`/`_hw_thread_cap` above.
        return 1

    def driver_version_for_jax( self, devices ):
        return devices( "cpu" )[ 0 ]


def _nb_workers():
    """What `CpuQueue` will use: `SDOT_NB_THREADS` if set, else every hardware thread. The
    per-thread scratch is sized on this, so the two must agree."""
    try:
        n = int( os.environ.get( "SDOT_NB_THREADS", "0" ) )
    except ValueError:
        n = 0
    return n if n > 0 else ( os.cpu_count() or 1 )


def _total_host_ram():
    try:
        return os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
    except ( AttributeError, ValueError ):
        return 4 * ( 1 << 30 )  # 4 GB conservative fallback
