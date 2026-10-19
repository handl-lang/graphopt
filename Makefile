# mlopt / research-mlopt — evidence generation (the Rust arm; the
# Python gates live in scripts/gates.sh).
# `make experiment` reproduces the paper's headline measurement.
#
# `mlopt-experiments` is deliberately *not* a member of the root cargo
# workspace (see the `exclude` row in Cargo.toml — it floats on the
# post-rename `hnl-engine-*` stack mlopt-kit can't share a lockfile
# with), so every cargo invocation goes through --manifest-path.

REPEAT ?= 200
EXPERIMENTS = experiments/Cargo.toml
MLOPT_RAW = https://raw.githubusercontent.com/handl-lang/research-mlopt/main
ARTIFACT = experiments/.cache/mlopt-kit/hnl/conv_bn.hnl

.PHONY: experiment verify build test artifact

build:
	cargo build --release --manifest-path $(EXPERIMENTS)

# Fetch the artifact under measurement from its declared git source —
# used only when no local arm of the resolver ladder applies (no in-tree
# or sibling mlopt checkout, no ~/.handl/pkg slot). The file lands in
# the ignored .cache dir: a fetched dependency artifact, not a
# committed copy.
artifact:
	@if [ -f packages/mlopt-kit/hnl/conv_bn.hnl ] || \
	    [ -f ../mlopt/packages/mlopt-kit/hnl/conv_bn.hnl ] || \
	    [ -f "$$HOME/.handl/pkg/mlopt-kit/hnl/conv_bn.hnl" ] || \
	    [ -f "$(ARTIFACT)" ]; then \
	  : local resolution arm present — no fetch needed; \
	else \
	  mkdir -p $(dir $(ARTIFACT)) && \
	  curl -fsSL "$(MLOPT_RAW)/packages/mlopt-kit/hnl/conv_bn.hnl" -o $(ARTIFACT) && \
	  echo "fetched $(ARTIFACT)"; \
	fi

experiment: build artifact
	cargo run --release --manifest-path $(EXPERIMENTS) --bin conv-bn-fold -- --repeat $(REPEAT) --json

# The certification leg only — value-equality, no timing.
verify: build artifact
	cargo run --release --manifest-path $(EXPERIMENTS) --bin conv-bn-fold -- --repeat 0

test:
	cargo test --manifest-path $(EXPERIMENTS)
