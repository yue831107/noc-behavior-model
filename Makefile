# NoC Behavior Model Makefile
# Usage: make [target] [VARIABLE=value]

# Variables (can be overridden)
# Windows: use "py -3" to avoid GTKWave python.exe PATH conflict
PYTHON = py -3

# Host-to-NoC payload settings
PAYLOAD_DIR = examples/Host_to_NoC/payload
CONFIG_DIR = examples/Host_to_NoC/config
PAYLOAD_SIZE = 1024
PAYLOAD_PATTERN = sequential
PAYLOAD_FILE = $(PAYLOAD_DIR)/payload.bin

# NoC-to-NoC payload settings
NOC_PAYLOAD_DIR = examples/NoC_to_NoC/payload
NOC_NODES = 16
NOC_SIZE = 256
NOC_PATTERN = sequential
SEED = 42

# Phony targets
.PHONY: help gen_payload gen_noc_payload sim_write sim_read sim_scatter sim_gather sim_all \
        sim_noc sim_noc_neighbor sim_noc_shuffle sim_noc_bit_reverse sim_noc_random sim_noc_transpose sim_noc_all \
        test test_unit test_integration test_coverage clean clean_payload clean_noc_payload all quick \
        viz_heatmap viz_latency viz_throughput viz_curves viz_dashboard viz_all viz_save

# Default target
help:
	@echo "NoC Behavior Model - Available Commands"
	@echo "========================================"
	@echo "[Host-to-NoC Workflow] (3-step)"
	@echo "  make gen_payload              Step 1: Generate payload BIN file"
	@echo "  make gen_config               Step 2: Generate transfer config YAML"
	@echo "  make sim                      Step 3: Run simulation"
	@echo ""
	@echo "  Options:"
	@echo "    gen_payload SIZE=N PATTERN=X    Payload size and pattern"
	@echo "    gen_config NUM_TRANSFERS=N      Number of transfers (default: 10)"
	@echo ""
	@echo "[NoC-to-NoC Workflow]"
	@echo "  make gen_noc_payload          Generate per-node payloads"
	@echo "  make sim_noc_neighbor         Run neighbor pattern"
	@echo "  make sim_noc_shuffle          Run shuffle pattern"
	@echo "  make sim_noc_bit_reverse      Run bit_reverse pattern"
	@echo "  make sim_noc_random           Run random pattern"
	@echo "  make sim_noc_transpose        Run transpose pattern"
	@echo "  make sim_noc_all              Run all patterns"
	@echo ""
	@echo "[Visualization]"
	@echo "  make viz                      Generate charts from latest sim"
	@echo ""
	@echo "[Testing]"
	@echo "  make test                     Run all pytest tests"
	@echo "  make test_unit                Unit tests only"
	@echo "  make test_integration         Integration tests only"
	@echo ""
	@echo "[Utilities]"
	@echo "  make clean                    Clean all generated files"
	@echo "----------------------------------------"
	@echo "Patterns: sequential random constant address walking_ones walking_zeros checkerboard"

# Host-to-NoC Payload generation
gen_payload:
	$(PYTHON) -c "from pathlib import Path; Path('$(PAYLOAD_DIR)').mkdir(parents=True, exist_ok=True)"
	$(PYTHON) tools/pattern_gen.py -p $(PAYLOAD_PATTERN) -s $(PAYLOAD_SIZE) -o $(PAYLOAD_FILE) --seed $(SEED) --hex-dump

# NoC-to-NoC Payload generation (per-node files)
gen_noc_payload:
	$(PYTHON) tools/pattern_gen.py --nodes $(NOC_NODES) -p $(NOC_PATTERN) -s $(NOC_SIZE) -o $(NOC_PAYLOAD_DIR) --seed $(SEED) --hex-dump

# Host-to-NoC Transfer Config generation
# Usage: make gen_config NUM_TRANSFERS=500
NUM_TRANSFERS = 10
TRANSFER_MODE = random
TRANSFER_MIN = 256
TRANSFER_MAX = 4096
TRANSFER_OUTPUT = examples/Host_to_NoC/config/generated.yaml

gen_config:
	$(PYTHON) tools/gen_transfer_config.py -n $(NUM_TRANSFERS) --mode $(TRANSFER_MODE) --min-size $(TRANSFER_MIN) --max-size $(TRANSFER_MAX) --seed $(SEED) -o $(TRANSFER_OUTPUT)

# Host-to-NoC Simulation (Specific Cases)
sim_write:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/broadcast_write.yaml --bin $(PAYLOAD_FILE)

sim_read:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/broadcast_read.yaml --bin $(PAYLOAD_FILE)

sim_scatter:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/scatter_write.yaml --bin $(PAYLOAD_FILE)

sim_gather:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/gather_read.yaml --bin $(PAYLOAD_FILE)

sim_all:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/multi_transfer.yaml --bin $(PAYLOAD_FILE)

# Host-to-NoC Simulation (Unified/Randomized)
# Requires: gen_payload, gen_config first
# Usage: make sim
sim:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(TRANSFER_OUTPUT) --bin $(PAYLOAD_FILE)

# NoC-to-NoC Simulations (requires gen_noc_payload first)
sim_noc: sim_noc_neighbor

sim_noc_neighbor:
	$(PYTHON) examples/NoC_to_NoC/run.py neighbor -P $(NOC_PAYLOAD_DIR)

sim_noc_shuffle:
	$(PYTHON) examples/NoC_to_NoC/run.py shuffle -P $(NOC_PAYLOAD_DIR)

sim_noc_bit_reverse:
	$(PYTHON) examples/NoC_to_NoC/run.py bit_reverse -P $(NOC_PAYLOAD_DIR)

sim_noc_random:
	$(PYTHON) examples/NoC_to_NoC/run.py random -P $(NOC_PAYLOAD_DIR)

sim_noc_transpose:
	$(PYTHON) examples/NoC_to_NoC/run.py transpose -P $(NOC_PAYLOAD_DIR)

sim_noc_all:
	$(PYTHON) examples/NoC_to_NoC/run.py --all -P $(NOC_PAYLOAD_DIR)

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test_unit:
	$(PYTHON) -m pytest tests/unit/ -v

test_integration:
	$(PYTHON) -m pytest tests/integration/ -v

test_coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing

# Cleaning
clean_payload:
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(PAYLOAD_DIR)').glob('*.bin')] if Path('$(PAYLOAD_DIR)').exists() else None"
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(PAYLOAD_DIR)').glob('*.hex')] if Path('$(PAYLOAD_DIR)').exists() else None"

clean_noc_payload:
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(NOC_PAYLOAD_DIR)').glob('*.bin')] if Path('$(NOC_PAYLOAD_DIR)').exists() else None"
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(NOC_PAYLOAD_DIR)').glob('*.hex')] if Path('$(NOC_PAYLOAD_DIR)').exists() else None"

clean: clean_payload clean_noc_payload
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(d) for d in Path('.').rglob('__pycache__') if d.is_dir()]"
	$(PYTHON) -c "import shutil; from pathlib import Path; shutil.rmtree('.pytest_cache', ignore_errors=True)"
	$(PYTHON) -c "import shutil; from pathlib import Path; shutil.rmtree('output', ignore_errors=True)"
	@echo "Clean complete."

# Workflows
all: gen_payload sim_all test

quick: gen_payload sim_write

# Visualization (uses latest simulation results)
viz:
	$(PYTHON) -m src.visualization.report_generator all --from-metrics output/metrics/latest.json

# Web Visualization




