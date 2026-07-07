CXX ?= g++
CXXFLAGS ?= -std=c++23 -O2 -Wall -Wextra
BUILD := build

BINS := $(BUILD)/demo_embed $(BUILD)/demo_search $(BUILD)/demb_selftest
HEADERS := $(wildcard include/distillembed/*.hpp)

all: $(BINS)

$(BUILD)/%: src/%.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
