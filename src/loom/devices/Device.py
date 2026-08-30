
class Device:
    """

    """

    driver_version: any

    _default_device = None

    @staticmethod
    def factory( value ) -> 'Device':
        if isinstance( value, Device ):
            return value.copy()

        if value is None:
            return Device.default()

        value = str( value ).lower()
        if value.startswith( "cpu" ):
            from .Cpu import Cpu
            return Cpu()

        if value.startswith( "gpu" ) or value.startswith( "cuda" ):
            from .CudaGpu import CudaGpu
            n = 0
            s = value.split( ":" )
            if len( s ) == 2:
                n = int( s[ 1 ] )
            return CudaGpu( n )

        if value.startswith( "metal" ) or value.startswith( "applegpu" ):
            from .AppleGpu import AppleGpu
            return AppleGpu()

        raise ValueError( f"unsupported device name: { value }" )

    @staticmethod
    def default() -> 'Device':
        if Device._default_device is None:
            from ..drivers.driver import driver
            if driver.available_gpus:
                from .CudaGpu import CudaGpu
                Device._default_device = CudaGpu( 0 )
            else:
                from .Cpu import Cpu
                Device._default_device = Cpu()
        return Device._default_device

    @property
    def name( self ) -> str:
        raise NotImplementedError

    # ── what the device IS, in C++ ────────────────────────────────────────────
    # A device is not a runtime parameter of a kernel: it is part of its TYPE. The queue decides
    # which memory space the kernel dereferences (see Ptr.h), and the memory space of a buffer
    # says where it already lives -- so both are typedefs in the generated source, and a call
    # whose data XLA put on the GPU needs no transfer at all.
    @property
    def cpp_queue_type( self ) -> str:
        """The `sdot::Queue` of a generated kernel ("CpuQueue" | "CudaQueue")."""
        raise NotImplementedError

    @property
    def cpp_memory_space( self ) -> str:
        """Where the buffers this device is given already live ("CpuHostMemorySpace" | ...)."""
        raise NotImplementedError

    @property
    def signature( self ) -> str:
        raise NotImplementedError

    @property
    def codegen_target( self ) -> str:
        """Codegen context tag for per-context FfiCode selectors ( see FfiCode.select_for )."""
        raise NotImplementedError

    # ── AdaptiveCpp / Jax-FFI mapping ─────────────────────────────────────────
    # Consumed by sdot.compilation.adaptive_cpp (make_executable / make_library) and by the
    # Jax FFI registration. Defaults describe a device NOT reachable through acpp (e.g. Apple
    # GPU / Metal): `acpp_reachable` False makes the acpp builders raise. Reachable devices
    # override these.
    #
    # NOTE: the *target* is normally NOT a property of the device. The default compilation
    # path is `generic` (AdaptiveCpp's SSCP flow): one target-independent binary, JIT-compiled
    # for whatever hardware is present at run time. What lives here is only what genuinely
    # depends on the device: whether acpp can reach it, which backend the acpp *build* must
    # enable, and the ahead-of-time fallback used when the SSCP toolchain is unavailable.
    # The policy itself is `compilation.adaptive_cpp.resolve_targets`.
    @property
    def acpp_reachable( self ) -> bool:
        """Whether AdaptiveCpp can target this device at all (False for e.g. Apple GPU / Metal)."""
        return False

    @property
    def acpp_aot_targets( self ):
        """`--acpp-targets` for the AHEAD-OF-TIME fallback ("omp", "cuda:sm_80"); None if there
        is none. This is the ONLY place a GPU architecture is ever named — under `generic` the
        architecture is a run-time detail, which is exactly the point."""
        return None

    @property
    def acpp_aot_profile( self ):
        """AdaptiveCpp feature profile the AOT fallback needs ("minimal" | "full")."""
        return "minimal"

    @property
    def acpp_backends( self ):
        """GPU backends to enable when BUILDING acpp (e.g. ("cuda",)); () for CPU-only.

        Needed even under `generic`: the SSCP JIT emits PTX, but talking to the card still
        goes through the compiled-in CUDA backend plugin (which only dlopens the driver)."""
        return ()

    @property
    def ffi_platform( self ) -> str:
        """XLA/Jax platform tag used by jax.ffi.register_ffi_target ("cpu" | "cuda" | ...)."""
        raise NotImplementedError

    @property
    def device_is_present( self ) -> bool:
        """Whether this device is actually usable here (hardware present AND acpp-reachable).

        Default: reachable iff acpp can target it (covers Cpu -> True, Apple GPU / Metal ->
        False). Devices whose hardware may be absent (CUDA) refine this."""
        return self.acpp_reachable

    def __eq__( self, value, / ) -> bool:
        if not isinstance( value, Device ):
            value = Device.factory( value )
        return str( self ) == str( value )

    def __neq__( self, value, / ) -> bool:
        return not self.__eq__( value )

    # fraction of total device/host RAM per-thread scratch is allowed to use. The rest holds the
    # I/O buffers, the runtime and the rest of the machine, none of which we measure here -- this
    # is a deliberately coarse cap, not a reading of what is actually free (which would be a moving
    # runtime quantity, wrong to freeze into a compiled shape; see `Cell.measure`). Overridable
    # per device (Cuda carries its own `mem_fraction`).
    scratch_ram_fraction = 0.5

    # hardware alignment in BYTES for the flattened batch dimension (see `PhysicalLayout`): its item
    # capacity is padded so its byte size is a multiple of this -- hence the ITEM padding differs for
    # fp32 vs fp64. 1 = no meaningful alignment (the CPU default, so the layout is inert); a GPU
    # overrides it with a coalescing / transaction size. A per-call `batch_alignment` wins.
    batch_alignment = 1

    def resolve_batch_alignment( self, override = None ) -> int:
        """The alignment to use: the per-call `override` if given, else this device's default."""
        return int( self.batch_alignment if override is None else override )

    def physical_axis_num( self, axis_names ):
        """Hardware physical-order numbers, one int per axis (lower = more leading), used by
        `PhysicalLayout` to place an output tensor's NON-batch axes in the order this device prefers
        -- a pure-performance reorder (strides keep the logical view). `None` (the default) keeps the
        logical order, so no device reorders anything until it overrides this. A device that does
        care (e.g. a column-major-friendly GPU) returns a list keyed off the `axis_names`."""
        return None

    def nb_threads( self, batch_axes = (), **per_thread ) -> int:
        """How many threads to SIZE per-thread scratch for.

        The hardware/RAM ceiling (`_hw_thread_cap`, device-specific) capped by the total number of
        batch items -- no point reserving more lanes than there is work -- and floored at 1. The
        per-thread footprint is passed through in `**per_thread` (`nb_local_bytes_per_thread`, ...)
        to the device hook; `batch_axes` is the list whose sizes multiply into the item count.

        This is a HOST-side, compile-time decision: the result becomes the static extent of a
        scratch axis, so every input must be knowable at trace time (a prescribed batch size, a
        capacity bound). A genuinely dynamic quantity folded in here would freeze into the compiled
        kernel -- which is why `scratch_ram_fraction` is a fixed fraction, not a live free-RAM read."""
        n = self._hw_thread_cap( **per_thread )
        total = 1
        for axis in batch_axes:
            total *= int( axis.max )
        # Le plafond par NOMBRE D'ITEMS s'applique toujours, y compris quand `batch_axes` est vide
        # -- une liste vide ne veut pas dire « on ne sait pas », elle veut dire UN item. Tous les
        # appelants la passent pour dire sur quoi le travail s'étale ; ne pas plafonner dans ce
        # cas-là réservait un scratch pour tout le parallélisme de la machine alors qu'un seul
        # work-item allait tourner. Invisible sur CPU (le plafond matériel y est le nombre de
        # cœurs), fatal sur GPU où il se déduit de la RAM du device : `Cell.cut` sur UNE cellule
        # y demandait 5.28 GiB de table de compaction.
        return max( 1, min( n, total ) )

    def _hw_thread_cap( self, **per_thread ) -> int:
        """The hardware/RAM ceiling on simultaneous threads, given a per-thread footprint. Device
        specific (cores + host RAM on CPU, SM occupancy + device RAM on CUDA); `nb_threads` wraps
        it with the batch cap and the floor."""
        raise NotImplementedError

    def group_size( self, **per_group_item ) -> int:
        """How many work-items cooperate in ONE SYCL work-group (the second level of parallelism:
        `nb_threads` sizes how many CONCURRENT items/groups run, this sizes how many work-items
        attack EACH one). A SEPARATE, orthogonal decision from `nb_threads`/`_hw_thread_cap` (that
        one is a whole-device/global-memory occupancy question; this one is a per-block/local-memory
        one) -- do not try to unify them.

        Default `1`: a work-group of size 1 makes the cooperative code path degenerate EXACTLY to
        the single-work-item algorithm it generalizes (a `sycl::group` of size 1 hitting
        `group_barrier`/a `local_accessor` is well-defined and cheap) -- so any device that does not
        override this is automatically safe, at the cost of no actual cooperation."""
        return 1

    def driver_version_for_jax( self, devices ):
        raise NotImplementedError

    @property
    def subgroup_size( self ) -> int:
        """Warp/wavefront width: how many work-items of a work-group execute in lockstep and can
        cheaply share a value via `sycl::group_broadcast`/`sycl::sub_group` collectives, one level
        BELOW `group_size` (a work-group is `group_size / subgroup_size` sub-groups). Default `1`
        (no sub-group cooperation, matching `group_size`'s own default degeneration) -- a body using
        `sub_group` (see `FfiCode`'s docstring) must degenerate correctly at `subgroup_size == 1`
        exactly like it must at `group_size == 1`, see `OtPlan1d.cxx::sort_diracs`."""
        return 1

    @property
    def is_apple_gpu( self ):
        return False

    @property
    def is_cuda_gpu( self ):
        return False

    @property
    def is_cpu( self ):
        return False


_default_device = None
