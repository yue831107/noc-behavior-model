# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NoC (Network-on-Chip) Behavior Model - A parameterizable cycle-accurate behavioral model for performance analysis and hardware verification. Simulates a 5x4 2D Mesh network with physically separated Request and Response networks.

## Common Commands

```bash
# Run all tests
py -3 -m pytest tests/ -v

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
```

Note: On Windows, use `py -3` instead of `python3` due to PATH conflicts with GTKWave.

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

- **Routing Selector** (`src/core/routing_selector.py`): V1 ingress/egress point
  - Path selection based on hop count and credits
  - Connects to 4 Edge Routers via PortWire

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

- FlitType: HEAD, BODY, TAIL, HEAD_TAIL (single-flit packet)
- PacketType: WRITE_REQ, WRITE_RESP, READ_REQ, READ_RESP

### Golden Data Verification

GoldenManager (`src/core/golden_manager.py`) handles expected data tracking:
- Captures write data for later verification
- For NoC-to-NoC: longest-distance-wins conflict resolution
- Verification compares actual memory contents vs. expected

## Directory Structure

```
src/
├── core/          # Core simulation components
│   ├── router.py  # XYRouter, WormholeArbiter, PortWire
│   ├── mesh.py    # 2D Mesh topology
│   ├── ni.py      # SlaveNI, MasterNI
│   ├── routing_selector.py  # V1System, NoCSystem
│   └── golden_manager.py    # Verification
├── axi/           # AXI protocol definitions
├── address/       # Address map and translation
├── traffic/       # Traffic pattern generators
├── visualization/ # Charts and metrics
└── config.py      # TransferConfig, NoCTrafficConfig

examples/
├── Host_to_NoC/   # run.py, config/*.yaml
└── NoC_to_NoC/    # run.py, config/*.yaml

tests/
├── unit/          # Component-level tests
├── integration/   # Multi-component tests
└── conftest.py    # Shared fixtures, run_multi_router_cycle()
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
