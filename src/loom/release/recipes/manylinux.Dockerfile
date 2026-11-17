# loom's build image for the `linux-x86_64` wheels, for Docker or Podman: the same as `manylinux.def` ( see its %help ).
# x86-64 whatever the machine: emulated on an arm64 one, and several times slower.
FROM --platform=linux/amd64 quay.io/pypa/manylinux_2_28_x86_64

RUN dnf -y install clang llvm-toolset libomp-devel libstdc++-static gcc-toolset-14-gcc-c++ && dnf clean all
# loom's pinned nvcc and CCCL, what builds and checks a wheel, then the project's frameworks and dependencies; the CUDA
# runtime the frameworks bring ( pip: nvidia/*/lib ) linked into one directory, for LD_LIBRARY_PATH
RUN /opt/python/cp313-cp313/bin/python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir numpy ninja pytest wheel twine auditwheel \
        "nvidia-cuda-nvcc>=13.3,<13.4" "nvidia-cuda-crt<13.4" "nvidia-nvvm<13.4" "nvidia-cuda-cccl<13.4" {pip} \
 && mkdir -p /opt/nvidia-lib \
 && for d in /opt/venv/lib/python3.13/site-packages/nvidia/*/lib; do ln -sf "$d"/* /opt/nvidia-lib/ 2>/dev/null || true; done

ENV PATH=/opt/venv/bin:$PATH \
    LOOM_CXX=/usr/bin/clang++ \
    LOOM_CXXFLAGS="-std=c++23 -static-libstdc++ -static-libgcc --gcc-install-dir=/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14" \
    LOOM_LDFLAGS="--gcc-install-dir=/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14" \
    LOOM_CUDA_HOST=/opt/rh/gcc-toolset-14/root/usr/bin/g++ \
    LOOM_CUDA_HOST_CXXFLAGS="-std=c++23" \
    LD_LIBRARY_PATH=/opt/nvidia-lib
