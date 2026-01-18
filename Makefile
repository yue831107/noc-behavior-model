# NoC Behavior Model Makefile
# Usage: make [target] [VARIABLE=value]
#
# All testing and simulation operations are available through make commands.
# Run 'make help' to see all available commands.

# ==============================================================================
# Variables (can be overridden)
# ==============================================================================

# Note: Use "py -3" on Windows CMD, "python" on MSYS/Git Bash
PYTHON ?= python

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

# Transfer config settings
NUM_TRANSFERS = 10
TRANSFER_MODE = random
TRANSFER_MIN = 256
TRANSFER_MAX = 4096
TRANSFER_OUTPUT = examples/Host_to_NoC/config/generated.yaml

# Multi-parameter sweep settings
SWEEP_BUFFER_DEPTH = 2,4,8,16

# Regression settings
REGRESSION_CONFIG = tools/regression_config.yaml
REGRESSION_CONFIG_NOC = tools/regression_config_noc.yaml
REGRESSION_OUTPUT = output/regression

# Parallel execution (number of workers, 'auto' for automatic)
PARALLEL = auto

# Channel mode settings (general or axi)
MODE = general
MODE_FLAG = $(if $(filter axi,$(MODE)),--mode axi,)

# ==============================================================================
# Phony targets
# ==============================================================================

.PHONY: help \
        test test_quick test_fast test_unit test_integration test_axi test_coverage \
        perf perf_theory perf_consistency perf_batch perf_batch_full perf_baseline \
        report report_perf report_coverage \
        gen_payload gen_noc_payload gen_config gen_config_sweep \
        sim sim_write sim_read sim_scatter sim_gather sim_all \
        sim_noc sim_noc_neighbor sim_noc_shuffle sim_noc_bit_reverse sim_noc_random sim_noc_transpose sim_noc_all \
        viz multi_para \
        regression regression_noc regression_quick test_regression \
        ci ci_quick \
        clean clean_payload clean_noc_payload \
        all quick

# ==============================================================================
# Help (Default Target)
# ==============================================================================

help:
	@echo "NoC Behavior Model - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "[Testing] - 測試命令"
	@echo "  make test              Run all tests (parallel)"
	@echo "  make test_quick        Quick smoke test (~10s)"
	@echo "  make test_fast         Run all tests, stop on first failure"
	@echo "  make test_unit         Unit tests only"
	@echo "  make test_integration  Integration tests only"
	@echo "  make test_axi          AXI Mode tests only"
	@echo "  make test_coverage     Run with coverage report"
	@echo ""
	@echo "[Performance] - 效能測試"
	@echo "  make perf              Performance tests"
	@echo "  make perf_theory       Theory validation tests"
	@echo "  make perf_consistency  Consistency validation tests"
	@echo "  make perf_batch        Batch test (100 configs)"
	@echo "  make perf_batch_full   Full batch test (500 configs)"
	@echo "  make perf_baseline     Regression detection tests"
	@echo ""
	@echo "[Reports] - 報告生成"
	@echo "  make report            Full test report + coverage"
	@echo "  make report_perf       Performance test report"
	@echo "  make report_coverage   Coverage report only"
	@echo ""
	@echo "[Host-to-NoC Workflow] - 主機到NoC模擬 (支援 MODE=general|axi)"
	@echo "  make gen_payload       Step 1: Generate payload"
	@echo "  make gen_config        Step 2: Generate config"
	@echo "  make sim               Step 3: Run simulation"
	@echo "  make sim_write         Run broadcast write"
	@echo "  make sim_read          Run broadcast read"
	@echo "  make sim_scatter       Run scatter write"
	@echo "  make sim_gather        Run gather read"
	@echo ""
	@echo "[NoC-to-NoC Workflow] - NoC內部通訊模擬 (支援 MODE=general|axi)"
	@echo "  make gen_noc_payload   Generate per-node payloads"
	@echo "  make sim_noc_neighbor  Run neighbor pattern"
	@echo "  make sim_noc_shuffle   Run shuffle pattern"
	@echo "  make sim_noc_bit_reverse  Run bit_reverse pattern"
	@echo "  make sim_noc_random    Run random pattern"
	@echo "  make sim_noc_transpose Run transpose pattern"
	@echo "  make sim_noc_all       Run all patterns"
	@echo ""
	@echo "[Regression] - 硬體參數優化"
	@echo "  make regression        Find optimal params (Host-to-NoC)"
	@echo "  make regression_noc    Find optimal params (NoC-to-NoC)"
	@echo "  make regression_quick  Quick search (early-stop)"
	@echo ""
	@echo "[CI/CD] - 持續整合"
	@echo "  make ci                CI full workflow"
	@echo "  make ci_quick          CI quick check"
	@echo ""
	@echo "[Utilities] - 工具"
	@echo "  make viz               Generate visualization charts"
	@echo "  make multi_para        Multi-parameter sweep"
	@echo "  make clean             Clean all generated files"
	@echo ""
	@echo "----------------------------------------"
	@echo "Options:"
	@echo "  MODE=general|axi       Channel mode (default: general)"
	@echo "  PARALLEL=N             Number of parallel workers (default: auto)"
	@echo "  PAYLOAD_SIZE=N         Payload size in bytes (default: 1024)"
	@echo "  NUM_TRANSFERS=N        Number of transfers (default: 10)"
	@echo ""
	@echo "Examples:"
	@echo "  make sim_write MODE=general    # Broadcast write (General Mode)"
	@echo "  make sim_write MODE=axi        # Broadcast write (AXI Mode)"
	@echo "  make sim_read MODE=axi         # Broadcast read (AXI Mode)"
	@echo ""
	@echo "Patterns: sequential random constant address walking_ones walking_zeros checkerboard"

# ==============================================================================
# Testing - 測試命令
# ==============================================================================

# Main test command (parallel execution)
test:
	$(PYTHON) -m pytest tests/ -n $(PARALLEL) -v

# Quick smoke test (~10s)
test_quick:
	$(PYTHON) -m pytest tests/unit/test_router_port.py tests/unit/test_xy_routing.py tests/unit/test_buffer.py -q

# Run all tests, stop on first failure
test_fast:
	$(PYTHON) -m pytest tests/ -n $(PARALLEL) -x -q

# Unit tests only (parallel)
test_unit:
	$(PYTHON) -m pytest tests/unit/ -n $(PARALLEL) -v

# Integration tests only (parallel)
test_integration:
	$(PYTHON) -m pytest tests/integration/ -n $(PARALLEL) -v

# AXI Mode tests only (parallel)
test_axi:
	$(PYTHON) -m pytest tests/ -k "axi" -n $(PARALLEL) -v

# Run with coverage report
test_coverage:
	$(PYTHON) -m pytest tests/ -n $(PARALLEL) --cov=src --cov-report=term-missing

# ==============================================================================
# Performance Testing - 效能測試
# ==============================================================================

# Performance tests
perf:
	$(PYTHON) -m pytest tests/performance/ -v

# Theory validation tests
perf_theory:
	$(PYTHON) -m pytest tests/performance/test_theory_validation.py -v

# Consistency validation tests
perf_consistency:
	$(PYTHON) -m pytest tests/performance/test_consistency_validation.py -v

# Batch performance test (100 configs)
perf_batch:
	$(PYTHON) tools/run_batch_perf_test.py --mode both --count 100

# Full batch performance test (500 configs)
perf_batch_full:
	$(PYTHON) tools/run_batch_perf_test.py --mode both --count 500

# Performance regression detection
perf_baseline:
	$(PYTHON) -m pytest tests/performance/test_performance_regression.py -v

# ==============================================================================
# Reports - 報告生成
# ==============================================================================

# Full test report with coverage
report:
	@mkdir -p output
	$(PYTHON) -m pytest tests/ -n $(PARALLEL) \
		--html=output/test_report.html --self-contained-html \
		--cov=src --cov-report=html:output/coverage
	@echo ""
	@echo "========================================="
	@echo "Reports generated:"
	@echo "  Test Report: output/test_report.html"
	@echo "  Coverage:    output/coverage/index.html"
	@echo "========================================="

# Performance test report
report_perf:
	@mkdir -p output
	$(PYTHON) -m pytest tests/performance/ -v \
		--html=output/perf_report.html --self-contained-html
	@echo ""
	@echo "Report: output/perf_report.html"

# Coverage report only
report_coverage:
	@mkdir -p output
	$(PYTHON) -m pytest tests/ -n $(PARALLEL) \
		--cov=src --cov-report=html:output/coverage -q
	@echo ""
	@echo "Coverage: output/coverage/index.html"

# ==============================================================================
# Host-to-NoC Workflow
# ==============================================================================

# Step 1: Generate payload
gen_payload:
	$(PYTHON) -c "from pathlib import Path; Path('$(PAYLOAD_DIR)').mkdir(parents=True, exist_ok=True)"
	$(PYTHON) tools/pattern_gen.py -p $(PAYLOAD_PATTERN) -s $(PAYLOAD_SIZE) -o $(PAYLOAD_FILE) --seed $(SEED) --hex-dump

# Step 2: Generate transfer config
gen_config:
	$(PYTHON) tools/gen_transfer_config.py -n $(NUM_TRANSFERS) --mode $(TRANSFER_MODE) --min-size $(TRANSFER_MIN) --max-size $(TRANSFER_MAX) --seed $(SEED) -o $(TRANSFER_OUTPUT)

# Generate config with sweep parameters
gen_config_sweep:
	$(PYTHON) tools/gen_transfer_config.py -n $(NUM_TRANSFERS) --mode $(TRANSFER_MODE) --min-size $(TRANSFER_MIN) --max-size $(TRANSFER_MAX) --seed $(SEED) -o $(TRANSFER_OUTPUT) --sweep-buffer-depth $(SWEEP_BUFFER_DEPTH)

# Step 3: Run simulation (use MODE=axi for AXI Mode)
sim:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(TRANSFER_OUTPUT) --bin $(PAYLOAD_FILE) $(MODE_FLAG)

# Specific simulation patterns (all support MODE=general or MODE=axi)
sim_write:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/broadcast_write.yaml --bin $(PAYLOAD_FILE) $(MODE_FLAG)

sim_read:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/broadcast_read.yaml --bin $(PAYLOAD_FILE) $(MODE_FLAG)

sim_scatter:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/scatter_write.yaml --bin $(PAYLOAD_FILE) $(MODE_FLAG)

sim_gather:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/gather_read.yaml --bin $(PAYLOAD_FILE) $(MODE_FLAG)

sim_all:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/multi_transfer.yaml --bin $(PAYLOAD_FILE) $(MODE_FLAG)

# ==============================================================================
# NoC-to-NoC Workflow
# ==============================================================================

# Generate per-node payloads
gen_noc_payload:
	$(PYTHON) tools/pattern_gen.py --nodes $(NOC_NODES) -p $(NOC_PATTERN) -s $(NOC_SIZE) -o $(NOC_PAYLOAD_DIR) --seed $(SEED) --hex-dump

# Default NoC simulation
sim_noc: sim_noc_neighbor

# Traffic patterns (all support MODE=general or MODE=axi)
sim_noc_neighbor:
	$(PYTHON) examples/NoC_to_NoC/run.py neighbor -P $(NOC_PAYLOAD_DIR) $(MODE_FLAG)

sim_noc_shuffle:
	$(PYTHON) examples/NoC_to_NoC/run.py shuffle -P $(NOC_PAYLOAD_DIR) $(MODE_FLAG)

sim_noc_bit_reverse:
	$(PYTHON) examples/NoC_to_NoC/run.py bit_reverse -P $(NOC_PAYLOAD_DIR) $(MODE_FLAG)

sim_noc_random:
	$(PYTHON) examples/NoC_to_NoC/run.py random -P $(NOC_PAYLOAD_DIR) $(MODE_FLAG)

sim_noc_transpose:
	$(PYTHON) examples/NoC_to_NoC/run.py transpose -P $(NOC_PAYLOAD_DIR) $(MODE_FLAG)

sim_noc_all:
	$(PYTHON) examples/NoC_to_NoC/run.py --all -P $(NOC_PAYLOAD_DIR) $(MODE_FLAG)

# ==============================================================================
# Regression - 硬體參數優化
# ==============================================================================

regression:
	$(PYTHON) tools/run_regression.py --config $(REGRESSION_CONFIG) -o $(REGRESSION_OUTPUT)

regression_noc:
	$(PYTHON) tools/run_regression.py --config $(REGRESSION_CONFIG_NOC) -o $(REGRESSION_OUTPUT)/noc

regression_quick:
	$(PYTHON) tools/run_regression.py --config $(REGRESSION_CONFIG) -o $(REGRESSION_OUTPUT) --early-stop

test_regression:
	$(PYTHON) -m pytest tests/performance/test_regression.py -v

# ==============================================================================
# CI/CD
# ==============================================================================

# CI full workflow
ci:
	$(PYTHON) -m pytest tests/ -n $(PARALLEL) --cov=src --cov-report=xml -q

# CI quick check
ci_quick:
	$(PYTHON) -m pytest tests/unit/ tests/integration/ -n $(PARALLEL) -x -q

# ==============================================================================
# Utilities
# ==============================================================================

# Generate visualization charts
viz:
	$(PYTHON) -m src.visualization.report_generator all --from-metrics output/metrics/latest.json

# Multi-parameter sweep
multi_para:
	$(PYTHON) tools/run_multi_para.py --config $(TRANSFER_OUTPUT) --bin $(PAYLOAD_FILE) -o output/multi_para

# ==============================================================================
# Cleaning
# ==============================================================================

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

# ==============================================================================
# Workflows
# ==============================================================================

all: gen_payload sim_all test

quick: gen_payload sim_write
