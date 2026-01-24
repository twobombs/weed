set(VCAPPOW "6" CACHE STRING "Log-2 of maximum qubit capacity of a single Tensor (must be at least 5, equivalent to >= 32 qubits)")

if (VCAPPOW LESS 5)
    message(FATAL_ERROR "VCAPPOW must be at least 5, equivalent to >= 32 qubits!")
endif (VCAPPOW LESS 5)

if (VCAPPOW LESS UINTPOW)
    message(FATAL_ERROR "VCAPPOW must be greater than or equal to UINTPOW!")
endif (VCAPPOW LESS UINTPOW)

if (VCAPPOW GREATER 6)
    target_sources(weed PRIVATE
        src/common/big_integer.cpp
        )
endif (VCAPPOW GREATER 6)
