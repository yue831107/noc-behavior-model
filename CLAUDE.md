# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NoC (Network-on-Chip) Behavior Model - A parameterizable cycle-accurate behavioral model for performance analysis and hardware verification. Simulates a 5x4 2D Mesh network with physically separated Request and Response networks.

## Common Commands

```bash
# Run all tests
py -3 -m pytest tests/ -v

# Quick smoke test (~10 seconds)
make test_smoke

# Run all tests, stop on first failure
make test_fast

# Run unit tests only
py -3 -m pytest tests/unit/ -v

# Run integration tests only
py -3 -m pytest tests/integration/ -v

# Run a single test file
py -3 -m pytest tests/unit/test_router_port.py -v

# Run a single test
py -3 -m pytest tests/unit/test_router_port.py::test_function_name -v

# Quick demo (Host-to-NoC write)
make quick

# Full workflow: generate payload, config, run simulation
make gen_payload PAYLOAD_SIZE=1024 PAYLOAD_PATTERN=sequential
make gen_config NUM_TRANSFERS=10
make sim

# NoC-to-NoC simulation
make gen_noc_payload
make sim_noc_neighbor

# Generate visualization charts
make viz

# Performance validation tests
py -3 -m pytest tests/performance/ -v

# Batch performance tests (500+ configs)
py -3 tools/run_batch_perf_test.py --mode both --count 500
```

Note: On Windows, use `py -3` instead of `python3` due to PATH conflicts with GTKWave.

## Development Setup

```bash
# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# Run linters manually
pre-commit run --all-files
```

## Architecture Overview

### Two Operation Modes

1. **Host-to-NoC (V1System)**: Single entry point via Routing Selector
   - Host CPU/DMA → SlaveNI → Routing Selector → Edge Routers → Mesh → MasterNI → Local Memory
   - Transfer modes: BROADCAST (same data to all nodes), SCATTER (different data per node)
   - Read modes: BROADCAST_READ, GATHER

2. **NoC-to-NoC (NoCSystem)**: Direct node-to-node communication
   - Each compute node has: NodeController + SlaveNI + MasterNI + LocalMemory
   - Traffic patterns: neighbor, shuffle, bit_reverse, random, transpose

### Key Components

- **Mesh** (`src/core/mesh.py`): 5x4 2D mesh topology
  - Column 0: Edge Routers (connect to Routing Selector, no NI)
  - Columns 1-4: Compute Nodes (Router + NI + Local Memory)

- **Router** (`src/core/router.py`): Combined Req/Resp router pair
  - XY Routing (X-first, then Y) with Y→X turn prevention
  - Wormhole switching with packet-level locking (WormholeArbiter)
  - Credit-based flow control via PortWire signal interface

- **Network Interface** (`src/core/ni.py`):
  - SlaveNI: AXI Slave, converts AXI → NoC flits (request path)
  - MasterNI: AXI Master, converts NoC flits → memory operations (response path)

- **Routing Selector** (`src/core/routing_selector/`): V1 ingress/egress point
  - Path selection based on hop count and credits
  - Connects to 4 Edge Routers via PortWire
  - Package structure: `selector.py`, `v1_system.py`, `noc_system.py`, `config.py`, `edge_port.py`

### Simulation Cycle Phases

The cycle-accurate model follows this sequence (see `conftest.py:run_multi_router_cycle`):
1. Sample inputs (from signals propagated at end of last cycle)
2. Clear input signals
3. Update ready signals
4. Route and forward flits
5. Propagate wire signals (PortWire)
6. Clear accepted outputs
7. Handle credit release

### Data Flow

```
Flit → Packet → AXI Transaction
  ↑        ↑
FlitFactory  PacketFactory, PacketAssembler/Disassembler
```

- Flit format: FlooNoC style with 20-bit header + AXI payload
  - Header contains: `rob_req`, `rob_idx`, `dst_id`, `src_id`, `last`, `axi_ch`
  - `hdr.last` bit marks packet end (replaces old FlitType enum)
  - AXI channels: AW (53b), W (288b), AR (53b), B (10b), R (266b)
- PacketType: WRITE_REQ, WRITE_RESP, READ_REQ, READ_RESP

### Golden Data Verification

GoldenManager (`src/verification/golden_manager.py`) handles expected data tracking:
- Captures write data for later verification
- For NoC-to-NoC: longest-distance-wins conflict resolution
- Verification compares actual memory contents vs. expected

## Directory Structure

```
src/
├── core/          # Core NoC hardware model
│   ├── router.py  # XYRouter, WormholeArbiter, PortWire, ChannelMode
│   ├── mesh.py    # 2D Mesh topology
│   ├── ni.py      # SlaveNI, MasterNI
│   ├── routing_selector/    # Package: V1System, NoCSystem, RoutingSelector
│   │   ├── selector.py      # RoutingSelector core logic
│   │   ├── v1_system.py     # V1System (Host-to-NoC)
│   │   ├── noc_system.py    # NoCSystem (NoC-to-NoC)
│   │   ├── config.py        # RoutingSelectorConfig
│   │   └── edge_port.py     # EdgePortManager
│   ├── channel_mode_strategy.py  # Strategy pattern for General/AXI mode
│   ├── flit.py    # FlitHeader, Flit, AXI payloads
│   ├── packet.py  # PacketAssembler, PacketDisassembler
│   └── buffer.py  # FlitBuffer
├── metrics/       # Performance metrics
│   └── booksim_metrics.py   # BookSim2-style metrics
├── testbench/     # DUT peripherals (not part of NoC)
│   ├── memory.py  # HostMemory, LocalMemory
│   ├── axi_master.py         # AXIMasterController
│   ├── host_axi_master.py    # HostAXIMaster (V1System)
│   ├── local_axi_master.py   # LocalAXIMaster (NoCSystem)
│   └── node_controller.py    # NodeController
├── verification/  # Validators and golden reference
│   ├── golden_manager.py     # GoldenManager
│   ├── theory_validator.py   # TheoryValidator
│   ├── consistency_validator.py  # ConsistencyValidator
│   └── metrics_provider.py   # MetricsProvider protocol
├── axi/           # AXI protocol definitions
├── address/       # Address map and translation
├── traffic/       # Traffic pattern generators
├── visualization/ # Charts and metrics
└── config.py      # TransferConfig, NoCTrafficConfig

tests/
├── fixtures/      # Test resources
│   └── configs/   # Test-specific config files
├── unit/          # Component-level tests
├── integration/   # Multi-component tests
│   └── deadlock_helpers.py  # Deadlock detection utilities
├── performance/   # Performance validation tests
│   ├── sweep/     # Parameter sweep infrastructure
│   └── regression/  # Regression test configs
└── conftest.py    # Shared fixtures, run_multi_router_cycle()

tools/
├── pattern_gen.py          # Payload generator
├── gen_transfer_config.py  # Config generator
├── run_multi_para.py       # Multi-parameter sweep runner
└── run_batch_perf_test.py  # Batch performance test runner (500+ tests)

examples/
├── Host_to_NoC/   # run.py, config/*.yaml
└── NoC_to_NoC/    # run.py, config/*.yaml
```

## Testing Patterns

Tests use fixtures from `conftest.py`:
- `router_config`, `ni_config`: Configuration objects
- `single_flit_factory`, `multi_flit_packet_factory`: Flit creation
- `two_routers_horizontal`, `router_chain_horizontal`: Connected router topologies
- `run_router_cycle()`, `run_multi_router_cycle()`: Cycle-accurate simulation helpers

## Configuration Files

YAML configs in `examples/*/config/`:
- `src_addr`, `src_size`: Source memory location and size
- `dst_addr`: Destination address in node local memory
- `target_nodes`: "all" or list like [0, 1, 2]
- `transfer_mode`: broadcast, scatter, broadcast_read, gather

Generated configs use `tools/gen_transfer_config.py`.
