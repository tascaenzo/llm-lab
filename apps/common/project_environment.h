#ifndef LLM_LAB_PROJECT_ENVIRONMENT_H
#define LLM_LAB_PROJECT_ENVIRONMENT_H

/**
 * Loads KEY=VALUE entries from LLM_LAB_ENV_FILE, or from .env in the current
 * directory when that variable is unset. Existing process variables always
 * win. A missing default .env is not an error; malformed or explicitly missing
 * files are reported to standard error.
 */
int llm_project_environment_load(void);

#endif
