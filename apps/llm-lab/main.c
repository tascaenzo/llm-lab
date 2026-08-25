#include <stdlib.h>

#include "cli_commands.h"
#include "project_environment.h"

int main(int argc, char **argv) {
    if (llm_project_environment_load() == 0) {
        return EXIT_FAILURE;
    }
    return llm_lab_run_command(argc, argv);
}
