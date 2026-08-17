function(llm_lab_set_warnings target)
    # The flags below are C dialect options. nvcc rejects them, so CUDA sources
    # in a mixed target are excluded rather than warned about.
    if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        set(warnings
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wstrict-prototypes
            -Wmissing-prototypes)
        if(LLM_LAB_WARNINGS_AS_ERRORS)
            list(APPEND warnings -Werror)
        endif()
        target_compile_options(${target} PRIVATE
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${warnings}>)
    endif()
endfunction()
