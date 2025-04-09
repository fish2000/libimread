
set(LLVM_DIR "${IM_LLVM_DIR}")

if("${IM_REQUIRE_LLVM_VERSION}" STREQUAL "")
  # Find any version present.
  find_package(LLVM REQUIRED CONFIG)
else()
  # Find a specific version.
  string(SUBSTRING "${IM_REQUIRE_LLVM_VERSION}" 0 1 MAJOR)
  string(SUBSTRING "${IM_REQUIRE_LLVM_VERSION}" 1 1 MINOR)
  # string(SUBSTRING "${IM_REQUIRE_LLVM_VERSION}" 2 1 PATCH)
  message("Looking for LLVM version ${MAJOR}.${MINOR}")
  find_package(LLVM "${MAJOR}.${MINOR}" REQUIRED CONFIG)
  if(NOT "${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR}" STREQUAL "${MAJOR}${MINOR}")
    message(FATAL_ERROR "LLVM version mismatch: required ${MAJOR}.${MINOR} but found ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
  endif()
endif()


set(LLVM_VERSION "${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR}")
set(LLVM_VERSION_TRIPLE "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")

# Notify the user what paths and LLVM version we are using
# message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found LLVM ${LLVM_VERSION_TRIPLE}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Check reported LLVM version
if (NOT "${LLVM_VERSION}" MATCHES "^[0-9][0-9]$")
  message(FATAL_ERROR "LLVM_VERSION not specified correctly. Must be <major><minor> E.g. LLVM 4.0 is \"40\"")
endif()
if (LLVM_VERSION LESS 40)
  message(FATAL_ERROR "LLVM version must be 4.0 or newer")
endif()

file(TO_NATIVE_PATH "${LLVM_TOOLS_BINARY_DIR}/llvm-config${CMAKE_EXECUTABLE_SUFFIX}" LLVM_CONFIG)
execute_process(COMMAND ${LLVM_CONFIG} --libdir OUTPUT_VARIABLE LLVM_CONFIG_LIBDIR)
string(STRIP "${LLVM_CONFIG_LIBDIR}" LLVM_CONFIG_LIBDIR)  # strip whitespace from start & end
string(REPLACE " " ";" LLVM_CONFIG_LIBDIR "${LLVM_CONFIG_LIBDIR}")  # convert into a list
if("${LLVM_CONFIG_LIBDIR}" STREQUAL "")
    message(WARNING "'llvm-config --libdir' is empty; this is likely somehow fucked.")
endif()
file(TO_NATIVE_PATH "${LLVM_CONFIG_LIBDIR}/clang/${LLVM_VERSION_TRIPLE}/include" CLANG_RUNTIME_HEADERS)
message(STATUS "Using Clang headers in: ${CLANG_RUNTIME_HEADERS}")

set(IM_CLANG_RUNTIME_HEADERS "${CLANG_RUNTIME_HEADERS}" CACHE INTERNAL "Clang headers, dogg" FORCE)