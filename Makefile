.PHONY: configure build run test format check-format clean

FORMAT_SOURCES := $(shell find apps src include tests utils -type f \
	\( -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.metal' \))

configure:
	cmake --preset debug

build: configure
	cmake --build --preset debug

run: build
	./build/debug/llm-lab

test: build
	ctest --preset debug

format:
	clang-format -i $(FORMAT_SOURCES)

check-format:
	clang-format --dry-run --Werror $(FORMAT_SOURCES)

clean:
	cmake -E rm -rf build
