
if(MSVC)
else()
  # src/CMakeLists.txt always enables C++11 for non-Windows,
  # so we'll give the same courtesy to tests
  add_definitions("-std=c++1z")
  add_definitions("-fno-rtti")
endif(MSVC)

set(HALOGEN_SUFFIX ".generator")

file(GLOB HALOGENS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
              "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(halogen_src ${HALOGENS})
    string(REPLACE ".cpp"       ""
                   halogen_name "${halogen_src}")
    
    set(halogen_exe "${halogen_name}${HALOGEN_SUFFIX}")
    
    halide_project(${halogen_name} "generator" "${CMAKE_CURRENT_SOURCE_DIR}/tools/GenGen.cpp"
                                               "${halogen_src}")
    
    # GENERATOR_ARGS "target=x86-64-osx-sse41-avx,x86-64-osx-sse41,x86-64-osx"
    halide_add_generator_dependency(TARGET "iod_symbolize"
                                    GENERATOR_TARGET "${halogen_name}"
                                    GENERATOR_NAME "${halogen_name}"
                                    GENERATED_FUNCTION "${halogen_name}"
                                    GENERATOR_ARGS "target=host" # x86-64-osx
                                    GENERATOR_POSTPROCESS TRUE)

endforeach()
