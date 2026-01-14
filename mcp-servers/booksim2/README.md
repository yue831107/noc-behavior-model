# BookSim2 MCP Server

MCP Server for integrating [BookSim2](https://github.com/booksim/booksim2) NoC simulator with Claude Code, enabling performance comparison with NoC Behavior Model.

## Features

- Run BookSim2 simulations with configurable parameters
- Parse and extract metrics from BookSim2 output
- Compare results between BookSim2 and NoC Behavior Model
- Generate equivalent configurations for fair comparison

## Prerequisites

### 1. Install BookSim2

```bash
# Clone repository
git clone https://github.com/booksim/booksim2.git
cd booksim2/src

# Build (requires flex, bison, and C++ compiler)
make
```

### 2. Set Environment Variable

```bash
# Linux/macOS
export BOOKSIM_PATH=/path/to/booksim2/src/booksim

# Windows (PowerShell)
$env:BOOKSIM_PATH = "C:\path\to\booksim2\src\booksim.exe"
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `booksim2_run` | Run BookSim2 simulation with specified configuration |
| `booksim2_parse_output` | Parse raw BookSim2 output to extract metrics |
| `booksim2_compare` | Compare BookSim2 results with NoC Behavior Model |
| `booksim2_list_options` | List available configuration options |
| `booksim2_create_config` | Generate BookSim2 configuration file |
| `booksim2_get_equivalent_config` | Get config matching NoC Behavior Model settings |

## Configuration Parameters

### Topology
- `topology`: mesh, torus, fly, flatfly, fattree, dragonfly
- `k`: Radix (nodes per dimension)
- `n`: Number of dimensions

### Routing
- `routing_function`: dor (XY), romm, min_adapt, planar_adapt, ugal

### Traffic Patterns
- uniform, transpose, bitcomp, bitrev, shuffle
- neighbor, randperm, diagonal, asymmetric, hotspot

### Flow Control
- `num_vcs`: Number of virtual channels
- `vc_buf_size`: VC buffer size (flits)

## Usage in Claude Code

After adding to MCP settings, you can:

```
# Run simulation
"Run BookSim2 with 8x8 mesh, transpose traffic, injection rate 0.1"

# Compare with NoC Model
"Compare BookSim2 results with our NoC Behavior Model simulation"

# Get equivalent config
"Generate BookSim2 config matching our 5x4 mesh setup"
```

## Comparison Notes

| Feature | BookSim2 | NoC Behavior Model |
|---------|----------|-------------------|
| Topology | Square mesh only | 5x4 non-square mesh |
| Routing | Multiple algorithms | XY routing |
| Flow Control | VC-based | Wormhole + Credits |
| Protocol | Abstract flits | AXI transactions |

For fair comparison:
1. Use DOR routing in BookSim2 (equivalent to XY)
2. Match injection rates
3. Use similar packet sizes
4. Compare normalized latency/throughput metrics
