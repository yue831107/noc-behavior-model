#!/usr/bin/env python3
"""
BookSim2 MCP Server.

This server provides tools to interact with BookSim2 NoC simulator,
enabling performance comparison with NoC Behavior Model.
"""

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Initialize MCP server
mcp = FastMCP("booksim2_mcp")

# ============================================================================
# Configuration Constants
# ============================================================================

# Default BookSim2 executable path (can be overridden via environment variable)
BOOKSIM_EXECUTABLE = os.environ.get("BOOKSIM_PATH", "booksim")

# Supported options
SUPPORTED_TOPOLOGIES = ["mesh", "torus", "fly", "flatfly", "fattree", "dragonfly"]
SUPPORTED_ROUTING = ["dor", "romm", "min_adapt", "planar_adapt", "ugal"]
SUPPORTED_TRAFFIC = [
    "uniform", "transpose", "bitcomp", "bitrev", "shuffle",
    "neighbor", "randperm", "diagonal", "asymmetric", "hotspot"
]
SUPPORTED_VC_ALLOCATORS = ["islip", "pim", "select", "separable_input_first", "separable_output_first"]
SUPPORTED_SW_ALLOCATORS = ["islip", "pim", "select", "separable_input_first", "separable_output_first"]


# ============================================================================
# Enums
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class SimulationType(str, Enum):
    """BookSim2 simulation type."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"


# ============================================================================
# Pydantic Models
# ============================================================================

class BookSimConfig(BaseModel):
    """BookSim2 configuration parameters."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    # Topology
    topology: str = Field(default="mesh", description="Network topology (mesh, torus, fly, etc.)")
    k: int = Field(default=8, description="Radix (nodes per dimension)", ge=2, le=64)
    n: int = Field(default=2, description="Number of dimensions", ge=1, le=4)

    # Routing
    routing_function: str = Field(default="dor", description="Routing algorithm (dor, romm, min_adapt, etc.)")

    # Flow control
    num_vcs: int = Field(default=4, description="Number of virtual channels", ge=1, le=32)
    vc_buf_size: int = Field(default=8, description="VC buffer size (flits)", ge=1, le=128)

    # Router architecture
    vc_allocator: str = Field(default="islip", description="VC allocator (islip, pim, select)")
    sw_allocator: str = Field(default="islip", description="Switch allocator (islip, pim, select)")

    # Timing
    credit_delay: int = Field(default=2, description="Credit delay cycles", ge=0, le=10)
    routing_delay: int = Field(default=0, description="Routing delay cycles", ge=0, le=10)
    vc_alloc_delay: int = Field(default=1, description="VC allocation delay cycles", ge=0, le=10)
    sw_alloc_delay: int = Field(default=1, description="Switch allocation delay cycles", ge=0, le=10)

    # Traffic
    traffic: str = Field(default="uniform", description="Traffic pattern (uniform, transpose, bitcomp, etc.)")
    packet_size: int = Field(default=5, description="Packet size in flits", ge=1, le=100)
    injection_rate: float = Field(default=0.1, description="Injection rate (packets/cycle/node)", ge=0.001, le=1.0)

    # Simulation
    sim_type: SimulationType = Field(default=SimulationType.LATENCY, description="Simulation type")
    warmup_periods: int = Field(default=3, description="Warmup periods", ge=0, le=20)
    sample_period: int = Field(default=1000, description="Sample period (cycles)", ge=100, le=100000)
    max_samples: int = Field(default=10, description="Maximum samples", ge=1, le=100)

    @field_validator('topology')
    @classmethod
    def validate_topology(cls, v: str) -> str:
        if v.lower() not in SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unsupported topology: {v}. Supported: {SUPPORTED_TOPOLOGIES}")
        return v.lower()

    @field_validator('routing_function')
    @classmethod
    def validate_routing(cls, v: str) -> str:
        if v.lower() not in SUPPORTED_ROUTING:
            raise ValueError(f"Unsupported routing: {v}. Supported: {SUPPORTED_ROUTING}")
        return v.lower()

    @field_validator('traffic')
    @classmethod
    def validate_traffic(cls, v: str) -> str:
        if v.lower() not in SUPPORTED_TRAFFIC:
            raise ValueError(f"Unsupported traffic: {v}. Supported: {SUPPORTED_TRAFFIC}")
        return v.lower()

    def to_config_string(self) -> str:
        """Convert to BookSim2 configuration file format."""
        lines = [
            "// BookSim2 Configuration (auto-generated)",
            "",
            "// Topology",
            f"topology = {self.topology};",
            f"k = {self.k};",
            f"n = {self.n};",
            "",
            "// Routing",
            f"routing_function = {self.routing_function};",
            "",
            "// Flow control",
            f"num_vcs = {self.num_vcs};",
            f"vc_buf_size = {self.vc_buf_size};",
            "wait_for_tail_credit = 1;",
            "",
            "// Router architecture",
            f"vc_allocator = {self.vc_allocator};",
            f"sw_allocator = {self.sw_allocator};",
            "alloc_iters = 1;",
            "",
            "// Timing",
            f"credit_delay = {self.credit_delay};",
            f"routing_delay = {self.routing_delay};",
            f"vc_alloc_delay = {self.vc_alloc_delay};",
            f"sw_alloc_delay = {self.sw_alloc_delay};",
            "",
            "input_speedup = 1;",
            "output_speedup = 1;",
            "internal_speedup = 1.0;",
            "",
            "// Traffic",
            f"traffic = {self.traffic};",
            f"packet_size = {self.packet_size};",
            f"injection_rate = {self.injection_rate};",
            "",
            "// Simulation",
            f"sim_type = {self.sim_type.value};",
            f"warmup_periods = {self.warmup_periods};",
            f"sample_period = {self.sample_period};",
            f"max_samples = {self.max_samples};",
        ]
        return "\n".join(lines)


class RunSimulationInput(BaseModel):
    """Input for running BookSim2 simulation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    config: BookSimConfig = Field(..., description="BookSim2 configuration")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class ParseOutputInput(BaseModel):
    """Input for parsing BookSim2 output."""
    model_config = ConfigDict(str_strip_whitespace=True)

    output_text: str = Field(..., description="Raw BookSim2 output text", min_length=10)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="Output format")


class CompareInput(BaseModel):
    """Input for comparing BookSim2 with NoC Behavior Model."""
    model_config = ConfigDict(str_strip_whitespace=True)

    booksim_results: Dict[str, Any] = Field(..., description="BookSim2 simulation results")
    noc_model_results: Dict[str, Any] = Field(..., description="NoC Behavior Model results")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class ListOptionsInput(BaseModel):
    """Input for listing available options."""
    model_config = ConfigDict(str_strip_whitespace=True)

    category: Optional[str] = Field(
        default=None,
        description="Category to list (topology, routing, traffic, allocator). None for all."
    )


class CreateConfigInput(BaseModel):
    """Input for creating configuration file."""
    model_config = ConfigDict(str_strip_whitespace=True)

    config: BookSimConfig = Field(..., description="BookSim2 configuration")
    output_path: Optional[str] = Field(default=None, description="Output file path (optional)")


# ============================================================================
# Result Dataclass
# ============================================================================

@dataclass
class SimulationResult:
    """Parsed BookSim2 simulation results."""
    # Basic metrics
    avg_latency: float = 0.0
    min_latency: float = 0.0
    max_latency: float = 0.0

    # Throughput
    avg_throughput: float = 0.0
    avg_accepted_rate: float = 0.0

    # Network stats
    avg_hops: float = 0.0
    total_packets: int = 0

    # Raw output
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_latency": self.avg_latency,
            "min_latency": self.min_latency,
            "max_latency": self.max_latency,
            "avg_throughput": self.avg_throughput,
            "avg_accepted_rate": self.avg_accepted_rate,
            "avg_hops": self.avg_hops,
            "total_packets": self.total_packets,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def parse_booksim_output(output: str) -> SimulationResult:
    """Parse BookSim2 output to extract metrics."""
    result = SimulationResult(raw_output=output)

    # Parse average latency
    lat_match = re.search(r"Packet latency average\s*=\s*([\d.]+)", output)
    if lat_match:
        result.avg_latency = float(lat_match.group(1))

    # Parse min/max latency
    min_match = re.search(r"minimum\s*=\s*([\d.]+)", output)
    max_match = re.search(r"maximum\s*=\s*([\d.]+)", output)
    if min_match:
        result.min_latency = float(min_match.group(1))
    if max_match:
        result.max_latency = float(max_match.group(1))

    # Parse throughput/accepted rate
    accepted_match = re.search(r"accepted\s*=\s*([\d.]+)", output)
    if accepted_match:
        result.avg_accepted_rate = float(accepted_match.group(1))
        result.avg_throughput = result.avg_accepted_rate

    # Parse hops
    hops_match = re.search(r"Hops average\s*=\s*([\d.]+)", output)
    if hops_match:
        result.avg_hops = float(hops_match.group(1))

    # Parse total packets
    packets_match = re.search(r"Total number of flits\s*=\s*(\d+)", output)
    if packets_match:
        result.total_packets = int(packets_match.group(1))

    return result


def run_booksim(config: BookSimConfig) -> tuple[bool, str]:
    """Run BookSim2 with given configuration."""
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
        f.write(config.to_config_string())
        config_path = f.name

    try:
        # Run BookSim2
        result = subprocess.run(
            [BOOKSIM_EXECUTABLE, config_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            return False, f"BookSim2 error:\n{result.stderr}"

        return True, result.stdout

    except FileNotFoundError:
        return False, (
            f"BookSim2 executable not found at '{BOOKSIM_EXECUTABLE}'. "
            "Please set BOOKSIM_PATH environment variable or install BookSim2."
        )
    except subprocess.TimeoutExpired:
        return False, "Simulation timed out after 5 minutes."
    except Exception as e:
        return False, f"Error running BookSim2: {str(e)}"
    finally:
        # Clean up temp file
        try:
            os.unlink(config_path)
        except OSError:
            pass


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="booksim2_run",
    annotations={
        "title": "Run BookSim2 Simulation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def booksim2_run(params: RunSimulationInput) -> str:
    """Run a BookSim2 NoC simulation with specified configuration.

    This tool executes BookSim2 with the provided configuration parameters
    and returns parsed simulation results including latency and throughput metrics.

    Args:
        params (RunSimulationInput): Configuration and output format options

    Returns:
        str: Simulation results in markdown or JSON format
    """
    success, output = run_booksim(params.config)

    if not success:
        return f"Error: {output}"

    result = parse_booksim_output(output)

    if params.response_format == ResponseFormat.MARKDOWN:
        lines = [
            "# BookSim2 Simulation Results",
            "",
            "## Configuration",
            f"- **Topology**: {params.config.k}x{params.config.k} {params.config.topology}",
            f"- **Routing**: {params.config.routing_function}",
            f"- **Traffic**: {params.config.traffic}",
            f"- **Injection Rate**: {params.config.injection_rate}",
            "",
            "## Performance Metrics",
            f"- **Average Latency**: {result.avg_latency:.2f} cycles",
            f"- **Min Latency**: {result.min_latency:.2f} cycles",
            f"- **Max Latency**: {result.max_latency:.2f} cycles",
            f"- **Average Throughput**: {result.avg_throughput:.4f}",
            f"- **Average Hops**: {result.avg_hops:.2f}",
            f"- **Total Packets**: {result.total_packets}",
        ]
        return "\n".join(lines)
    else:
        return json.dumps({
            "config": params.config.model_dump(),
            "results": result.to_dict()
        }, indent=2)


@mcp.tool(
    name="booksim2_parse_output",
    annotations={
        "title": "Parse BookSim2 Output",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def booksim2_parse_output(params: ParseOutputInput) -> str:
    """Parse raw BookSim2 output text to extract simulation metrics.

    Args:
        params (ParseOutputInput): Raw output text and format options

    Returns:
        str: Parsed metrics in specified format
    """
    result = parse_booksim_output(params.output_text)

    if params.response_format == ResponseFormat.MARKDOWN:
        lines = [
            "# Parsed BookSim2 Results",
            "",
            f"- **Average Latency**: {result.avg_latency:.2f} cycles",
            f"- **Min/Max Latency**: {result.min_latency:.2f} / {result.max_latency:.2f} cycles",
            f"- **Throughput**: {result.avg_throughput:.4f}",
            f"- **Average Hops**: {result.avg_hops:.2f}",
            f"- **Total Packets**: {result.total_packets}",
        ]
        return "\n".join(lines)
    else:
        return json.dumps(result.to_dict(), indent=2)


@mcp.tool(
    name="booksim2_compare",
    annotations={
        "title": "Compare with NoC Behavior Model",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def booksim2_compare(params: CompareInput) -> str:
    """Compare BookSim2 results with NoC Behavior Model results.

    Generates a detailed comparison report showing differences in latency,
    throughput, and other metrics between the two simulators.

    Args:
        params (CompareInput): Results from both simulators

    Returns:
        str: Comparison report in specified format
    """
    bs = params.booksim_results
    noc = params.noc_model_results

    # Calculate differences
    def get_diff(bs_val: float, noc_val: float) -> tuple[float, str]:
        if bs_val == 0:
            return 0.0, "N/A"
        diff_pct = ((noc_val - bs_val) / bs_val) * 100
        sign = "+" if diff_pct > 0 else ""
        return diff_pct, f"{sign}{diff_pct:.1f}%"

    latency_diff, latency_str = get_diff(
        bs.get("avg_latency", 0),
        noc.get("avg_latency", 0)
    )
    throughput_diff, throughput_str = get_diff(
        bs.get("avg_throughput", 0),
        noc.get("avg_throughput", 0)
    )

    if params.response_format == ResponseFormat.MARKDOWN:
        lines = [
            "# BookSim2 vs NoC Behavior Model Comparison",
            "",
            "| Metric | BookSim2 | NoC Model | Difference |",
            "|--------|----------|-----------|------------|",
            f"| Avg Latency | {bs.get('avg_latency', 'N/A'):.2f} | {noc.get('avg_latency', 'N/A'):.2f} | {latency_str} |",
            f"| Min Latency | {bs.get('min_latency', 'N/A'):.2f} | {noc.get('min_latency', 'N/A'):.2f} | - |",
            f"| Max Latency | {bs.get('max_latency', 'N/A'):.2f} | {noc.get('max_latency', 'N/A'):.2f} | - |",
            f"| Throughput | {bs.get('avg_throughput', 'N/A'):.4f} | {noc.get('avg_throughput', 'N/A'):.4f} | {throughput_str} |",
            "",
            "## Analysis",
            "",
        ]

        # Add analysis
        if abs(latency_diff) < 5:
            lines.append("- Latency results are **well-matched** (within 5%)")
        elif abs(latency_diff) < 15:
            lines.append("- Latency results show **moderate difference** (5-15%)")
        else:
            lines.append("- Latency results show **significant difference** (>15%)")

        return "\n".join(lines)
    else:
        return json.dumps({
            "booksim2": bs,
            "noc_model": noc,
            "differences": {
                "latency_pct": latency_diff,
                "throughput_pct": throughput_diff,
            }
        }, indent=2)


@mcp.tool(
    name="booksim2_list_options",
    annotations={
        "title": "List Available Options",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def booksim2_list_options(params: ListOptionsInput) -> str:
    """List available BookSim2 configuration options.

    Args:
        params (ListOptionsInput): Optional category filter

    Returns:
        str: Available options in markdown format
    """
    options = {
        "topology": SUPPORTED_TOPOLOGIES,
        "routing": SUPPORTED_ROUTING,
        "traffic": SUPPORTED_TRAFFIC,
        "vc_allocator": SUPPORTED_VC_ALLOCATORS,
        "sw_allocator": SUPPORTED_SW_ALLOCATORS,
    }

    if params.category:
        cat = params.category.lower()
        if cat in options:
            return f"## {cat.title()} Options\n\n" + "\n".join(f"- `{opt}`" for opt in options[cat])
        elif cat == "allocator":
            return (
                "## Allocator Options\n\n"
                "### VC Allocators\n" + "\n".join(f"- `{opt}`" for opt in options["vc_allocator"]) +
                "\n\n### Switch Allocators\n" + "\n".join(f"- `{opt}`" for opt in options["sw_allocator"])
            )
        else:
            return f"Unknown category: {params.category}. Available: topology, routing, traffic, allocator"

    lines = ["# BookSim2 Configuration Options", ""]
    for cat, opts in options.items():
        lines.append(f"## {cat.replace('_', ' ').title()}")
        for opt in opts:
            lines.append(f"- `{opt}`")
        lines.append("")

    return "\n".join(lines)


@mcp.tool(
    name="booksim2_create_config",
    annotations={
        "title": "Create Configuration File",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def booksim2_create_config(params: CreateConfigInput) -> str:
    """Create a BookSim2 configuration file.

    Args:
        params (CreateConfigInput): Configuration parameters and optional output path

    Returns:
        str: Configuration file content or confirmation of file creation
    """
    config_content = params.config.to_config_string()

    if params.output_path:
        try:
            with open(params.output_path, 'w') as f:
                f.write(config_content)
            return f"Configuration file created: {params.output_path}"
        except Exception as e:
            return f"Error creating file: {e}"

    return f"```\n{config_content}\n```"


@mcp.tool(
    name="booksim2_get_equivalent_config",
    annotations={
        "title": "Get Equivalent Configuration",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def booksim2_get_equivalent_config(mesh_cols: int = 5, mesh_rows: int = 4) -> str:
    """Get BookSim2 configuration equivalent to NoC Behavior Model settings.

    Creates a BookSim2 configuration that matches the NoC Behavior Model's
    5x4 mesh topology with XY routing for fair comparison.

    Args:
        mesh_cols: Number of columns (default 5 for NoC Model)
        mesh_rows: Number of rows (default 4 for NoC Model)

    Returns:
        str: Equivalent BookSim2 configuration
    """
    # NoC Behavior Model uses 5x4 mesh, but BookSim2 requires square dimensions
    # Use the larger dimension for BookSim2
    k = max(mesh_cols, mesh_rows)

    config = BookSimConfig(
        topology="mesh",
        k=k,
        n=2,
        routing_function="dor",  # Dimension-order = XY routing
        num_vcs=4,
        vc_buf_size=8,
        traffic="uniform",
        packet_size=5,
        injection_rate=0.1,
    )

    lines = [
        "# BookSim2 Configuration for NoC Behavior Model Comparison",
        "",
        f"NoC Behavior Model: {mesh_cols}x{mesh_rows} mesh with XY routing",
        f"BookSim2 Equivalent: {k}x{k} mesh with DOR (dimension-order) routing",
        "",
        "## Notes:",
        "- BookSim2 requires square mesh (k x k), using larger dimension",
        "- DOR routing in BookSim2 = XY routing in NoC Model",
        "- Adjust injection_rate to match NoC Model's traffic load",
        "",
        "## Configuration:",
        "```",
        config.to_config_string(),
        "```"
    ]

    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
