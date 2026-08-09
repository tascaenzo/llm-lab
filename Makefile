.PHONY: configure build run format check-format clean

FORMAT_SOURCES := $(shell find apps src -type f -name '*.c') $(shell find include -type f -name '*.h')

configure:
	cmake --preset debug

build: configure
	cmake --build --preset debug

run: build
	./build/debug/llm-lab

format:
	clang-format -i $(FORMAT_SOURCES)

check-format:
	clang-format --dry-run --Werror $(FORMAT_SOURCES)

clean:
	cmake -E rm -rf build
