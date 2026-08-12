#include <stdlib.h>

#include "backend_contract_suite.h"
#include "runtime/backend.h"
#include "test_support.h"

static int run_with_threads(size_t thread_count) {
    const llm_cpu_backend_config config = {.thread_count = thread_count};
    llm_backend *backend = NULL;
    TEST_ASSERT(llm_backend_cpu_create_with_config(&config, &backend) == LLM_OK);
    const int result = runtime_backend_contract_suite(backend);
    llm_backend_destroy(backend);
    return result;
}

int main(void) {
    return run_with_threads(1U) == EXIT_SUCCESS && run_with_threads(4U) == EXIT_SUCCESS
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
