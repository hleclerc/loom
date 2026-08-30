from .Device import Device
import ctypes
import os

_libcuda = None


class CudaGpu( Device ):
    # a memory-transaction size in BYTES: the flattened batch is padded so its byte size aligns to a
    # coalescing block (see `Device.batch_alignment` / `PhysicalLayout`). In items this is 32 for fp32,
    # 16 for fp64.
    batch_alignment = 128

    def __init__( self, device_id, mem_fraction = 0.5, _attrs = None ):
        self.mem_fraction = mem_fraction  # fraction of total_dev_mem reserved for per-thread scratch
        self.device_id = device_id
        # INDEXÉ, pas dépaqueté par la fin : `shm_per_block` est le DERNIER, donc un
        # `*_, sm_major, sm_minor = attrs` prend `( sm_minor, shm_per_block )` et fabrique un
        # `sm_549152` au lieu de `sm_75`. Resté invisible tant que rien n'utilisait la cible AOT,
        # puisque `generic` gagnait toujours (voir `resolve_targets`).
        self._attrs = _attrs # (nb_sm, max_thr_per_sm, regs_per_sm, shm_per_sm, total_dev_mem, sm_major, sm_minor, shm_per_block)

    def copy( self ) -> 'Device':
        return CudaGpu( self.device_id, self.mem_fraction, self._attrs )

    @property
    def name( self ):
        return "Cuda"

    @property
    def signature( self ):
        attrs = self._get_attrs()
        if attrs is None:
            return "cuda"
        sm_major, sm_minor = attrs[ 5 ], attrs[ 6 ]
        return f"cuda_sm{ sm_major }{ sm_minor }"

    @property
    def codegen_target( self ):
        return "cuda"

    @property
    def cpp_queue_type( self ):
        return "CudaQueue"

    @property
    def cpp_memory_space( self ):
        # XLA already hands us the call's buffers in the device's global memory: the kernel
        # dereferences them where they are, and `make_available` merely retypes the pointers.
        return "CudaGlobalMemorySpace"

    @property
    def is_cuda_gpu( self ):
        return True

    @property
    def acpp_reachable( self ):
        return True

    @property
    def acpp_aot_targets( self ):
        # Only used when the SSCP toolchain is unavailable. Note that this path bakes in the
        # compute capability AND needs a full CUDA toolkit at compile time (clang's CUDA
        # support caps at CUDA 12.8, so a CUDA 13 host cannot compile it at all) -- which is
        # precisely why `generic` is the default.
        attrs = self._get_attrs()
        if attrs is None:
            return "cuda"
        sm_major, sm_minor = attrs[ 5 ], attrs[ 6 ]
        return f"cuda:sm_{ sm_major }{ sm_minor }"

    @property
    def acpp_aot_profile( self ):
        return "full"

    @property
    def acpp_backends( self ):
        return ( "cuda", )

    @property
    def ffi_platform( self ):
        # "gpu", NOT "cuda": `jaxlib.xla_client.register_custom_call_target` translates the
        # platform through `xla_platform_names = { 'cpu': 'Host', 'gpu': 'CUDA' }` and passes
        # anything else through untouched. Registering under "cuda" therefore files the handler
        # in a bucket nothing ever reads, and the call fails at run time with
        # `No FFI handler registered for <name> on a platform CUDA`. "gpu" is also what maps to
        # ROCM on an AMD build of jaxlib.
        return "gpu"

    @property
    def device_is_present( self ):
        # acpp-reachable AND a real CUDA driver/device available (libcuda loads, attrs read).
        return self.acpp_reachable and self._get_attrs() is not None

    def driver_version_for_jax( self, devices ):
        return devices( "gpu" )[ self.device_id ]

    def __repr__( self ) -> str:
        return f"CudaGpu:{ self.device_id }"

    def _get_attrs( self ):
        if self._attrs is not None:
            return self._attrs
        lib = _load_libcuda()
        if lib is None:
            return None

        def attr( a ):
            v = ctypes.c_int( 0 )
            lib.cuDeviceGetAttribute( ctypes.byref( v ), a, self.device_id )
            return v.value

        mem = ctypes.c_size_t( 0 )
        lib.cuDeviceTotalMem_v2( ctypes.byref( mem ), self.device_id )

        # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT                 = 16
        # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR       = 39
        # CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR     = 82
        # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
        # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK          = 8
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR             = 75
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR             = 76
        self._attrs = ( attr( 16 ), attr( 39 ), attr( 82 ), attr( 81 ), mem.value, attr( 75 ), attr( 76 ),
                        attr( 8 ) )
        return self._attrs

    def _hw_thread_cap( self, nb_regs_per_thread=0, nb_shared_bytes_per_thread=0,
                    nb_local_bytes_per_thread=0, nb_pinned_bytes_per_thread=0, nb_waves=4 ):
        attrs = self._get_attrs()
        if attrs is None:
            raise RuntimeError( "CUDA device attributes unavailable (libcuda not found)" )
        nb_sm, max_thr_per_sm, regs_per_sm, shm_per_sm, total_dev_mem, *_ = attrs

        # SM-level occupancy constraints (simultaneous threads per SM)
        thr_per_sm = max_thr_per_sm
        if nb_regs_per_thread > 0:
            thr_per_sm = min( thr_per_sm, regs_per_sm // nb_regs_per_thread )
        if nb_shared_bytes_per_thread > 0:
            thr_per_sm = min( thr_per_sm, shm_per_sm  // nb_shared_bytes_per_thread )

        n = nb_waves * nb_sm * max( 1, thr_per_sm )

        # global memory budgets (constrain total work units, not per-SM occupancy).
        # `total_dev_mem` is the raw physical memory; only a fraction is actually allocatable
        # because the input/output data buffers and the XLA/JAX runtime must also reside on the
        # card. Reserve `mem_fraction` of it for per-thread scratch so the sum of all argument
        # buffers stays under XLA's per-call base limit (~the whole card).
        if nb_local_bytes_per_thread > 0:
            usable_dev_mem = int( total_dev_mem * self.mem_fraction )
            n = min( n, usable_dev_mem // nb_local_bytes_per_thread  )
        if nb_pinned_bytes_per_thread > 0:
            n = min( n, _total_host_ram() // nb_pinned_bytes_per_thread )

        # `nb_threads` (base) applies the batch cap and the floor(1); the device RAM budget above
        # already uses `mem_fraction`, Cuda's own analogue of `scratch_ram_fraction`.
        return n

    @property
    def subgroup_size( self ):
        """Warp size: 32 on every NVIDIA architecture to date (Turing through Blackwell) -- a stable
        hardware constant, not worth a device-attribute round trip. Used by `group_size` (a warp-row
        histogram costs `ceil( group_size / 32 )` shared rows, not `group_size` of them, see
        `nb_shared_bytes_per_subgroup` below) and by `OtPlan1d.py`'s `local_mem_elems` sizing, kept
        in sync with `OtPlan1d.cxx::sort_diracs`'s sub-group-cooperative histogram."""
        return 32

    def group_size( self, nb_shared_bytes_per_group_item=0, nb_shared_bytes_per_subgroup=0,
                     nb_shared_bytes_fixed=0, max_group_size=128, **_ ):
        """How many work-items cooperate per work-group -- a per-BLOCK shared-memory/occupancy
        question, separate from `nb_threads`/`_hw_thread_cap`'s whole-device/global-memory one
        (don't unify them). Start at `max_group_size` and halve until the group's total shared-
        memory usage fits under the PER-BLOCK budget (also capped at the hardware's max threads/SM;
        floored at 1):

            candidate * nb_shared_bytes_per_group_item          -- one row PER WORK-ITEM (old scheme)
          + ceil( candidate / subgroup_size ) * nb_shared_bytes_per_subgroup  -- one row PER WARP
          + nb_shared_bytes_fixed                                -- any extra fixed row(s) on top

        `nb_shared_bytes_per_group_item` and `nb_shared_bytes_per_subgroup` are independent knobs a
        caller mixes as needed (`OtPlan1d`'s radix histogram uses ONLY the per-subgroup term now --
        see below -- but a future cooperative algorithm that genuinely needs one row per work-item
        can still ask for that).

        `max_group_size` was briefly forced to `1` (no cooperation) after measuring that
        cooperating on the SORT ALONE was a net negative for `OtPlan1d`: the forward was flat across
        `group_size` and the backward got monotonically worse, because the leader-only sequential
        `udp_cont` SWEEP dominated and a wider work-group just reserved more of an SM's resources
        while work-item 0 did all the work. Once the sweep itself was ALSO made cooperative (jump-
        started per chunk via `Image::udp_at`, see [[group-cooperative-sort]]), a fresh sweep on the
        same hardware (600 angles, n=1e4, RTX 2080 Ti) showed the backward improving MONOTONICALLY
        with `group_size` (140.9ms at 1 -> 44.6ms at 32, the largest value the 256-bucket radix sort's
        shared-memory footprint allowed WITH THE OLD one-row-per-work-item histogram) -- so
        `max_group_size` reverts to `128` (the original default). With the one-row-per-WARP histogram
        (see `OtPlan1d.cxx::sort_diracs`), the SAME 256-bucket footprint now costs `ceil(n/32)` rows
        instead of `n` rows, so this budget check no longer clamps `group_size` down to 32 -- `ncu`
        measured only 1 resident warp/SM at the old clamp (3.13% occupancy, 83.5% of stall cycles
        waiting on an L1TEX scoreboard with nothing else to hide behind), so the expectation is that
        letting `group_size` grow further keeps helping via better latency hiding. The forward stayed
        flat even with the sweep parallelized, for a reason not yet root-caused (suspected: a
        separate, non-cooperative kernel elsewhere in the per-call graph) -- flagged, not blocking.
        Re-measure (`ncu`/the execution-speed benchmark) before changing this again.

        Deliberately `shm_per_block` (`MAX_SHARED_MEMORY_PER_BLOCK`, the no-opt-in static default,
        49152B on Turing), NOT `shm_per_sm` (`..._PER_MULTIPROCESSOR`, 65536B on Turing): AdaptiveCpp's
        CUDA backend launches SSCP/generic kernels via the driver API without ever calling
        `cuFuncSetAttribute( CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ... )`, so a launch
        requesting more than the static per-block default fails at submission with
        `CUDA_ERROR_INVALID_VALUE` (`error code = CU:1`) -- silently, no compile-time signal -- even
        though the SM could physically hold more."""
        attrs = self._get_attrs()
        if attrs is None:
            raise RuntimeError( "CUDA device attributes unavailable (libcuda not found)" )
        _, max_thr_per_sm, _, _, _, _, _, shm_per_block = attrs

        def shared_bytes( c ):
            b = c * nb_shared_bytes_per_group_item + nb_shared_bytes_fixed
            if nb_shared_bytes_per_subgroup > 0:
                b += -( -c // self.subgroup_size ) * nb_shared_bytes_per_subgroup  # ceil div
            return b

        n = min( max_group_size, max_thr_per_sm )
        if nb_shared_bytes_per_group_item > 0 or nb_shared_bytes_per_subgroup > 0:
            while n > 1 and shared_bytes( n ) > shm_per_block:
                n //= 2
        return max( 1, n )

def _load_libcuda():
    global _libcuda
    if _libcuda is None:
        for lib_name in ( "libcuda.so.1", "libcuda.so", "nvcuda.dll" ):
            try:
                lib = ctypes.cdll.LoadLibrary( lib_name )
                lib.cuInit( 0 )
                _libcuda = lib
                break
            except OSError:
                pass
        if _libcuda is None:
            _libcuda = False
    return _libcuda if _libcuda else None


def _total_host_ram():
    try:
        return os.sysconf( 'SC_PHYS_PAGES' ) * os.sysconf( 'SC_PAGE_SIZE' )
    except ( AttributeError, ValueError ):
        return 4 * ( 1 << 30 )
