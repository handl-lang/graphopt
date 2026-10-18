# catopt / research-mlopt — evidence generation (the Rust arm; the
# Python gates live in scripts/gates.sh).
# `make experiment` reproduces the paper's headline measurement.
#
# `mlopt-experiments` is deliberately *not* a member of the root cargo
# workspace (see the `exclude` row in Cargo.toml — it floats on the
# post-rename `hnl-engine-*` stack catopt-kit can't share a lockfile
# with), so every cargo invocation goes through --manifest-path.

REPEAT ?= 200
EXPERIMENTS = experiments/Cargo.toml
CATOPT_RAW = https://raw.githubusercontent.com/handl-lang/catopt/main
ARTIFACT = experiments/.cache/catopt-kit/hnl/conv_bn.hnl

.PHONY: experiment verify build test artifact

build:
	cargo build --release --manifest-path $(EXPERIMENTS)

# Fetch the artifact under measurement from its declared git source —
# used only when no local arm of the resolver ladder applies (no in-tree
# or sibling catopt checkout, no ~/.handl/pkg slot). The file lands in
# the ignored .cache dir: a fetched dependency artifact, not a
# committed copy.
artifact:
	@if [ -f packages/catopt-kit/hnl/conv_bn.hnl ] || \
	    [ -f ../catopt/packages/catopt-kit/hnl/conv_bn.hnl ] || \
	    [ -f "$$HOME/.handl/pkg/catopt-kit/hnl/conv_bn.hnl" ] || \
	    [ -f "$(ARTIFACT)" ]; then \
	  : local resolution arm present — no fetch needed; \
	else \
	  mkdir -p $(dir $(ARTIFACT)) && \
	  curl -fsSL "$(CATOPT_RAW)/packages/catopt-kit/hnl/conv_bn.hnl" -o $(ARTIFACT) && \
	  echo "fetched $(ARTIFACT)"; \
	fi

experiment: build artifact
	cargo run --release --manifest-path $(EXPERIMENTS) --bin conv-bn-fold -- --repeat $(REPEAT) --json

# The certification leg only — value-equality, no timing.
verify: build artifact
	cargo run --release --manifest-path $(EXPERIMENTS) --bin conv-bn-fold -- --repeat 0

test:
	cargo test --manifest-path $(EXPERIMENTS)
