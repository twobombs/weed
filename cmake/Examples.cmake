add_executable (nand
    examples/nand.cpp
    )
set_target_properties(nand PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")
target_link_libraries (nand ${WEED_LIBS})

set(EXAMPLE_COMPILE_OPTS ${TEST_COMPILE_OPTS} -DCATCH_CONFIG_FAST_COMPILE)
target_compile_options (nand PUBLIC ${EXAMPLE_COMPILE_OPTS})
