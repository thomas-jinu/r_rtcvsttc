#!/bin/bash
set -euo pipefail

# Perlmutter CPU build: GNU, Cray MPI/LibSci/HDF5, and Kokkos Threads.
module load PrgEnv-gnu/8.7.0
module load cray-mpich/9.1.0
module load cray-libsci/26.03.0
module load cray-hdf5/1.14.3.7

export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic
export BASE="${BASE:-/global/common/software/m5228}"
export HDF5_ROOT="${HDF5_ROOT:-${HDF5_DIR:?HDF5_DIR is unset; check the cray-hdf5 module}}"
export BOOST_ROOT="${BOOST_ROOT:-$BASE/local/boost-1.88.0}"
export GSL_ROOT_DIR="${GSL_ROOT_DIR:-$BASE/local/gsl-2.8}"
export CMAKE_PREFIX="${CMAKE_PREFIX:-$BASE/local/cmake-3.31.9}"
export LIBSCI_DIR="${LIBSCI_DIR:-/opt/cray/pe/libsci/26.03.0/GNU/12/x86_64/lib}"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
if [[ -x "$CMAKE_PREFIX/bin/cmake" ]]; then
  export PATH="$CMAKE_PREFIX/bin:$PATH"
fi

repo="$(git rev-parse --show-toplevel)"
build="${BUILD_DIR:-$repo/build_perlmutter_cpu_threads}"
install="${INSTALL_DIR:-$PWD/installdir}"

printf 'C compiler:   %s\nC++ wrapper:  %s\n' "$(command -v "$CC")" "$(command -v "$CXX")"
printf 'HDF5: %s\nBoost: %s\nGSL: %s\nLibSci: %s\n' "$HDF5_ROOT" "$BOOST_ROOT" "$GSL_ROOT_DIR" "$LIBSCI_DIR"
module -t list 2> "$repo/nersc_cpu.modules"

[[ -f "$HDF5_ROOT/include/H5Cpp.h" ]] || { echo "Missing HDF5 C++ header under $HDF5_ROOT" >&2; exit 1; }
[[ -f "$BOOST_ROOT/include/boost/version.hpp" ]] || { echo "Missing Boost headers under $BOOST_ROOT" >&2; exit 1; }
[[ -f "$GSL_ROOT_DIR/include/gsl/gsl_version.h" ]] || { echo "Missing GSL headers under $GSL_ROOT_DIR" >&2; exit 1; }
[[ -f "$LIBSCI_DIR/libsci_gnu_mpi_mp.so" ]] || { echo "Missing threaded MPI LibSci library: $LIBSCI_DIR/libsci_gnu_mpi_mp.so" >&2; exit 1; }
[[ ! -e "$build/CMakeCache.txt" ]] || { echo "Refusing to reuse existing build: $build" >&2; exit 1; }

cmake -S "$repo" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF \
  -DCMAKE_C_COMPILER="$(command -v cc)" \
  -DCMAKE_CXX_COMPILER="$(command -v CC)" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_INSTALL_PREFIX="$install" \
  -DCMAKE_C_FLAGS="-fopenmp" \
  -DCMAKE_CXX_FLAGS="-fopenmp" \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
  -DBUILD_TESTING=ON \
  -DDMRGPP_USE_MPI=ON \
  -DDMRG_BATCHED_GEMM_TYPE=Kokkos \
  -DHDF5_ROOT="$HDF5_ROOT" \
  -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON \
  -DHDF5_PREFER_PARALLEL=OFF \
  -DBoost_ROOT="$BOOST_ROOT" \
  -DGSL_ROOT_DIR="$GSL_ROOT_DIR" \
  -DKokkos_ENABLE_SERIAL=OFF \
  -DKokkos_ENABLE_THREADS=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF \
  -DKokkosKernels_ENABLE_TPL_BLAS=ON \
  -DKokkosKernels_ENABLE_TPL_LAPACK=ON \
  -DBLAS_LIBRARIES=sci_gnu_mpi_mp \
  -DBLAS_LIBRARY_DIRS="$LIBSCI_DIR" \
  -DLAPACK_LIBRARIES=sci_gnu_mpi_mp \
  -DLAPACK_LIBRARY_DIRS="$LIBSCI_DIR"

cmake --build "$build" --parallel "${BUILD_JOBS:-16}"
cmake --install "$build"
executable="$install/bin/dmrg"
[[ -x "$executable" ]] || { echo "DMRG++ executable was not installed: $executable" >&2; exit 1; }
printf 'Installed executable:\n'
ls -lh "$executable"
printf 'MPI/library linkage:\n'
ldd "$executable" | grep -Ei 'mpi|libsci|blas|lapack|gomp|omp' || true
