#ifndef LLM_LAB_BACKEND_CONTRACT_SUITE_H
#define LLM_LAB_BACKEND_CONTRACT_SUITE_H

#include "runtime/backend.h"

/* Runs the backend-independent training contract against an existing backend. */
int runtime_backend_contract_suite(llm_backend *backend);

#endif
