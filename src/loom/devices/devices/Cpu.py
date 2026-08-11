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
    def acpp_reachable( self ):
        return True

    @property
    def acpp_aot_targets( self ):
        # the library-only OpenMP backend: no LLVM, hence the fallback that works everywhere
        return "omp"

    @property
    def ffi_platform( self ):
        return "cpu"

    def __repr__( self ) -> str:
        return "Cpu"

    def _hw_thread_cap( self, nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=1 ):
        # registers managed by compiler; shared memory not applicable to CPU threads
        # both local and pinned bytes draw from host RAM
        n          = os.cpu_count() or 1
        per_thread = max( nb_local_bytes_per_thread, nb_pinned_bytes_per_thread )
        if per_thread > 0:
            usable = int( _total_host_ram() * self.scratch_ram_fraction )
            n = min( n, usable // per_thread )
        return n

    def group_size( self, **per_group_item ):
        # AdaptiveCpp's own docs: "Don't use nd_range parallel for unless you absolutely have to,
        # as it is difficult to map efficiently to CPUs" -- this device only ever selects the
        # `omp.library-only` backend (`acpp_aot_targets = "omp"`, no Clang-plugin-accelerated
        # `omp.accelerated` path configured), which implements nd_range barriers via Boost.Fiber
        # (cooperative user-space fibers): "the relative cost of a barrier... is significantly
        # higher... kernels relying on barriers may experience substantial performance degradation."
        # Stay at the degenerate `1` (see `Device.group_size`) on purpose -- do NOT raise this to
        # "use more cores per group" without re-reading that doc, it would make things slower, not
        # faster; CPU parallelism already comes from `nb_threads`/`_hw_thread_cap` above.
        return 1

    def driver_version_for_jax( self, devices ):
        return devices( "cpu" )[ 0 ]


def _total_host_ram():
    try:
        return os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
    except ( AttributeError, ValueError ):
        return 4 * ( 1 << 30 )  # 4 GB conservative fallback
