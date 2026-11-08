#!/bin/bash
set -euo pipefail

# Perlmutter/NERSC build using the NVIDIA programming environment.
module load cray-mpich/9.0.1
module load cray-libsci/25.09.0
module load cray-hdf5/1.14.3.7
module load PrgEnv-nvidia
module load cudatoolkit

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic

repo="$(git rev-parse --show-toplevel)"
build="$repo/build_perlmutter_nvidia_cuda"
install="$PWD/installdir"

# Use HDF5 from the loaded Cray module. Boost/GSL default to the local
# prefixes from the supplied setup script; override these variables if needed.
export HDF5_ROOT="${HDF5_ROOT:-${HDF5_DIR:?HDF5_DIR is unset; check cray-hdf5 module}}"
export BASE="${BASE:-/global/common/software/m5228}"
export BOOST_ROOT="${BOOST_ROOT:-$BASE/local/boost-1.88.0}"
export GSL_ROOT_DIR="${GSL_ROOT_DIR:-$BASE/local/gsl-2.8}"
export CMAKE_PREFIX="${CMAKE_PREFIX:-$BASE/local/cmake-3.31.9}"
if [[ -x "$CMAKE_PREFIX/bin/cmake" ]]; then
  export PATH="$CMAKE_PREFIX/bin:$PATH"
fi

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

print_line() { printf '%s\n' "$*"; }

print_line "C compiler:   $(command -v "$CC")"
print_line "C++ wrapper:  $(command -v "$CXX")"
CC --version | head -n 1 || true
print_line "MPICH:        ${CRAY_MPICH_DIR:-not reported}"
print_line "HDF5:         $HDF5_ROOT"
print_line "Boost:        $BOOST_ROOT"
print_line "GSL:          $GSL_ROOT_DIR"

[[ -f "$HDF5_ROOT/include/H5Cpp.h" ]] || {
  printf 'Missing HDF5 C++ header under %s\n' "$HDF5_ROOT" >&2
  exit 1
}
[[ -f "$BOOST_ROOT/include/boost/version.hpp" ]] || {
  printf 'Missing Boost headers under %s\n' "$BOOST_ROOT" >&2
  exit 1
}
[[ -f "$GSL_ROOT_DIR/include/gsl/gsl_version.h" ]] || {
  printf 'Missing GSL headers under %s\n' "$GSL_ROOT_DIR" >&2
  exit 1
}
[[ ! -e "$build/CMakeCache.txt" ]] || {
  printf 'Refusing to reuse existing build: %s\n' "$build" >&2
  exit 1
}

module -t list 2> "$repo/nersc.modules"

cmake -S "$repo" -B "$build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  -DCMAKE_C_COMPILER="$(command -v cc)" \
  -DCMAKE_CXX_COMPILER="$(command -v CC)" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="$install" \
  -DBUILD_TESTING=ON \
  -DDMRGPP_USE_MPI=ON \
  -DDMRG_BATCHED_GEMM_TYPE=Kokkos \
  -DHDF5_ROOT="$HDF5_ROOT" \
  -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON \
  -DHDF5_PREFER_PARALLEL=OFF \
  -DBoost_ROOT="$BOOST_ROOT" \
  -DGSL_ROOT_DIR="$GSL_ROOT_DIR" \
  -DKokkos_ENABLE_SERIAL=OFF \
  -DKokkos_ENABLE_THREADS=OFF \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkosKernels_ENABLE_TPL_CUBLAS=ON \
  -DKokkosKernels_ENABLE_TPL_BLAS=OFF \
  -DKokkosKernels_ENABLE_TPL_LAPACK=OFF \
  -DBLA_VENDOR=All

cmake --build "$build" --parallel "${BUILD_JOBS:-16}"
cmake --install "$build"

executable="$install/bin/dmrg"
[[ -x "$executable" ]] || {
  printf 'DMRG++ executable was not installed: %s\n' "$executable" >&2
  exit 1
}

print_line "Installed executable:"
ls -lh "$executable"
print_line "MPI/library linkage:"
ldd "$executable" | grep -Ei 'mpi|libsci|blas|lapack|gomp|omp|cuda|cublas' || true
