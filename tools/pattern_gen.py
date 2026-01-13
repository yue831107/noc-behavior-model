#!/usr/bin/env python3
"""
Pattern Generator for NoC File-In/File-Out Testing.

Generates payload.bin files with various data patterns for hardware verification.

Host-to-NoC Usage:
    python tools/pattern_gen.py -p sequential -s 1024 -o payload.bin
    python tools/pattern_gen.py -p random -s 2048 --seed 42 -o test.bin

NoC-to-NoC Usage (per-node files):
    python tools/pattern_gen.py --nodes 16 -p sequential -s 256 -o payload/
    python tools/pattern_gen.py --nodes 16 -p random -s 256 --seed 42 -o payload/
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional


class PatternType(Enum):
    """Supported pattern types."""
    SEQUENTIAL = "sequential"   # 0x00, 0x01, 0x02, ...
    RANDOM = "random"           # Random bytes
    CONSTANT = "constant"       # Fixed value fill
    ADDRESS = "address"         # Address value as data (4-byte aligned)
    WALKING_ONES = "walking_ones"   # 0x01, 0x02, 0x04, 0x08, ...
    WALKING_ZEROS = "walking_zeros"  # 0xFE, 0xFD, 0xFB, 0xF7, ...
    CHECKERBOARD = "checkerboard"    # 0xAA, 0x55, 0xAA, 0x55, ...


class PatternGenerator:
    """Generate various data patterns for testing."""

    def __init__(
        self,
        pattern: PatternType,
        size: int,
        seed: Optional[int] = None,
        value: int = 0,
        start_addr: int = 0,
    ):
        """
        Initialize pattern generator.

        Args:
            pattern: Pattern type to generate.
            size: Total size in bytes.
            seed: Random seed (for random pattern).
            value: Constant value (for constant pattern).
            start_addr: Starting address (for address pattern).
        """
        self.pattern = pattern
        self.size = size
        self.seed = seed
        self.value = value & 0xFF
        self.start_addr = start_addr
        self._rng = random.Random(seed)

    def generate(self) -> bytes:
        """Generate the pattern data."""
        generators = {
            PatternType.SEQUENTIAL: self._gen_sequential,
            PatternType.RANDOM: self._gen_random,
            PatternType.CONSTANT: self._gen_constant,
            PatternType.ADDRESS: self._gen_address,
            PatternType.WALKING_ONES: self._gen_walking_ones,
            PatternType.WALKING_ZEROS: self._gen_walking_zeros,
            PatternType.CHECKERBOARD: self._gen_checkerboard,
        }
        gen_func = generators.get(self.pattern)
        if gen_func is None:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        return bytes(gen_func())

    def _gen_sequential(self) -> Iterator[int]:
        """Generate sequential bytes: 0x00, 0x01, 0x02, ..."""
        for i in range(self.size):
            yield i & 0xFF

    def _gen_random(self) -> Iterator[int]:
        """Generate random bytes."""
        for _ in range(self.size):
            yield self._rng.randint(0, 255)

    def _gen_constant(self) -> Iterator[int]:
        """Generate constant value fill."""
        for _ in range(self.size):
            yield self.value

    def _gen_address(self) -> Iterator[int]:
        """Generate address-based data (4-byte little-endian)."""
        for offset in range(0, self.size, 4):
            addr = self.start_addr + offset
            # Little-endian 4-byte address
            addr_bytes = struct.pack("<I", addr & 0xFFFFFFFF)
            for i in range(min(4, self.size - offset)):
                yield addr_bytes[i]

    def _gen_walking_ones(self) -> Iterator[int]:
        """Generate walking ones: 0x01, 0x02, 0x04, 0x08, ..."""
        for i in range(self.size):
            yield 1 << (i % 8)

    def _gen_walking_zeros(self) -> Iterator[int]:
        """Generate walking zeros: 0xFE, 0xFD, 0xFB, 0xF7, ..."""
        for i in range(self.size):
            yield ~(1 << (i % 8)) & 0xFF

    def _gen_checkerboard(self) -> Iterator[int]:
        """Generate checkerboard: 0xAA, 0x55, 0xAA, 0x55, ..."""
        for i in range(self.size):
            yield 0xAA if i % 2 == 0 else 0x55

    def write_to_file(self, path: Path) -> int:
        """
        Write pattern to binary file.

        Args:
            path: Output file path.

        Returns:
            Number of bytes written.
        """
        data = self.generate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return len(data)

    def write_hex_dump(self, path: Path, bytes_per_line: int = 16) -> None:
        """
        Write pattern as hex dump (human-readable).

        Args:
            path: Output file path.
            bytes_per_line: Bytes per line in hex dump.
        """
        data = self.generate()
        lines = []

        for offset in range(0, len(data), bytes_per_line):
            chunk = data[offset:offset + bytes_per_line]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(
                chr(b) if 32 <= b < 127 else "." for b in chunk
            )
            lines.append(f"{offset:08X}  {hex_part:<{bytes_per_line*3}}  {ascii_part}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")


def create_payload(
    output: Path,
    pattern: str = "sequential",
    size: int = 1024,
    seed: Optional[int] = None,
    value: int = 0,
    start_addr: int = 0,
    hex_dump: bool = False,
) -> int:
    """
    Convenience function to create a payload file.

    Args:
        output: Output file path.
        pattern: Pattern name (sequential, random, constant, address, etc.).
        size: Size in bytes.
        seed: Random seed.
        value: Constant value.
        start_addr: Starting address for address pattern.
        hex_dump: Also create a .hex dump file.

    Returns:
        Number of bytes written.
    """
    try:
        pattern_type = PatternType(pattern.lower())
    except ValueError:
        valid = [p.value for p in PatternType]
        raise ValueError(f"Invalid pattern '{pattern}'. Valid: {valid}")

    gen = PatternGenerator(
        pattern=pattern_type,
        size=size,
        seed=seed,
        value=value,
        start_addr=start_addr,
    )

    bytes_written = gen.write_to_file(Path(output))

    if hex_dump:
        hex_path = Path(output).with_suffix(".hex")
        gen.write_hex_dump(hex_path)

    return bytes_written


def create_noc_payloads(
    output_dir: Path,
    num_nodes: int,
    pattern: str = "sequential",
    size: int = 256,
    seed: int = 42,
    value: int = 0,
    start_addr: int = 0,
    hex_dump: bool = False,
) -> int:
    """
    Create per-node payload files for NoC-to-NoC testing.

    Each node gets unique data by incorporating node_id into the pattern.
    Output files: node_00.bin, node_01.bin, ..., node_15.bin

    Args:
        output_dir: Output directory path.
        num_nodes: Number of nodes (typically 16 for 5x4 mesh).
        pattern: Pattern name.
        size: Size per node in bytes.
        seed: Base random seed.
        value: Constant value (for constant pattern).
        start_addr: Base starting address.
        hex_dump: Also create .hex dump files.

    Returns:
        Total bytes written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0

    for node_id in range(num_nodes):
        # Generate unique data for each node
        data = _generate_node_data(
            node_id=node_id,
            pattern=pattern,
            size=size,
            seed=seed,
            value=value,
            start_addr=start_addr,
        )

        # Write binary file
        bin_path = output_dir / f"node_{node_id:02d}.bin"
        bin_path.write_bytes(data)
        total_bytes += len(data)

        # Write hex dump if requested
        if hex_dump:
            hex_path = output_dir / f"node_{node_id:02d}.hex"
            _write_hex_dump(data, hex_path)

    return total_bytes


def _generate_node_data(
    node_id: int,
    pattern: str,
    size: int,
    seed: int = 42,
    value: int = 0,
    start_addr: int = 0,
) -> bytes:
    """
    Generate unique data for a specific node.

    Each pattern incorporates node_id to ensure uniqueness:
    - sequential: offset by node_id * 16
    - random: seed = base_seed + node_id
    - constant: prefix with node_id byte
    - node_id: fill with node_id value
    - address: offset address by node_id << 16
    - walking_ones/zeros: prefix with node_id, rotate by node_id
    - checkerboard: prefix with node_id, invert for odd nodes
    """
    pattern = pattern.lower()
    data = []

    if pattern == "sequential":
        # Sequential with node_id offset
        for i in range(size):
            data.append((node_id * 16 + i) & 0xFF)

    elif pattern == "random":
        # Deterministic random with node-specific seed
        rng = random.Random(seed + node_id)
        for _ in range(size):
            data.append(rng.randint(0, 255))

    elif pattern == "constant":
        # Node_id prefix + constant value
        data.append(node_id & 0xFF)
        for _ in range(size - 1):
            data.append(value & 0xFF)

    elif pattern == "node_id":
        # Fill with node_id value (first byte is node_id for uniqueness)
        data.append(node_id & 0xFF)
        for _ in range(size - 1):
            data.append(node_id & 0xFF)

    elif pattern == "address":
        # Address-based with node_id offset
        base_addr = start_addr + (node_id << 16)
        for offset in range(0, size, 4):
            addr = base_addr + offset
            addr_bytes = struct.pack("<I", addr & 0xFFFFFFFF)
            for i in range(min(4, size - offset)):
                data.append(addr_bytes[i])

    elif pattern == "walking_ones":
        # Node_id prefix + walking ones rotated by node_id
        data.append(node_id & 0xFF)
        for i in range(size - 1):
            bit_pos = (i + node_id) % 8
            data.append(1 << bit_pos)

    elif pattern == "walking_zeros":
        # Node_id prefix + walking zeros rotated by node_id
        data.append(node_id & 0xFF)
        for i in range(size - 1):
            bit_pos = (i + node_id) % 8
            data.append(~(1 << bit_pos) & 0xFF)

    elif pattern == "checkerboard":
        # Node_id prefix + checkerboard (inverted for odd nodes)
        data.append(node_id & 0xFF)
        invert = (node_id % 2 == 1)
        for i in range(size - 1):
            if i % 2 == 0:
                data.append(0x55 if invert else 0xAA)
            else:
                data.append(0xAA if invert else 0x55)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return bytes(data)


def _write_hex_dump(data: bytes, path: Path, bytes_per_line: int = 16) -> None:
    """Write data as hex dump."""
    lines = []
    for offset in range(0, len(data), bytes_per_line):
        chunk = data[offset:offset + bytes_per_line]
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        ascii_part = "".join(
            chr(b) if 32 <= b < 127 else "." for b in chunk
        )
        lines.append(f"{offset:08X}  {hex_part:<{bytes_per_line*3}}  {ascii_part}")
    path.write_text("\n".join(lines) + "\n")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate payload files for NoC testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Patterns:
  sequential    : 0x00, 0x01, 0x02, ... (wraps at 0xFF)
  random        : Random bytes (use --seed for reproducibility)
  constant      : Fill with constant value (use --value)
  address       : 4-byte little-endian address values
  walking_ones  : 0x01, 0x02, 0x04, 0x08, ...
  walking_zeros : 0xFE, 0xFD, 0xFB, 0xF7, ...
  checkerboard  : 0xAA, 0x55, 0xAA, 0x55, ...

Host-to-NoC Examples:
  %(prog)s -p sequential -s 1024 -o payload.bin
  %(prog)s -p random -s 2048 --seed 42 -o test.bin

NoC-to-NoC Examples (per-node files):
  %(prog)s --nodes 16 -p sequential -s 256 -o payload/
  %(prog)s --nodes 16 -p random -s 256 --seed 42 -o payload/
        """
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="sequential",
        choices=[p.value for p in PatternType],
        help="Pattern type (default: sequential)"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=1024,
        help="Size in bytes (default: 1024, or 256 for NoC-to-NoC)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (or directory for --nodes)"
    )
    parser.add_argument(
        "--nodes", "-n",
        type=int,
        default=None,
        help="Number of nodes for NoC-to-NoC (generates per-node files)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--value", "-v",
        type=lambda x: int(x, 0),  # Support hex input like 0xAB
        default=0,
        help="Constant value (for constant pattern, supports hex)"
    )
    parser.add_argument(
        "--start-addr",
        type=lambda x: int(x, 0),
        default=0,
        help="Starting address (for address pattern)"
    )
    parser.add_argument(
        "--hex-dump",
        action="store_true",
        help="Also create .hex human-readable dump files"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    try:
        if args.nodes is not None:
            # NoC-to-NoC mode: generate per-node files
            size = args.size if args.size != 1024 else 256  # Default 256 for NoC
            bytes_written = create_noc_payloads(
                output_dir=Path(args.output),
                num_nodes=args.nodes,
                pattern=args.pattern,
                size=size,
                seed=args.seed,
                value=args.value,
                start_addr=args.start_addr,
                hex_dump=args.hex_dump,
            )

            if not args.quiet:
                print(f"Generated {args.nodes} node files ({bytes_written} total bytes)")
                print(f"  Output: {args.output}/node_XX.bin")
                print(f"  Pattern: {args.pattern}")
                print(f"  Size per node: {size} bytes")
                if args.pattern == "random":
                    print(f"  Base seed: {args.seed}")
                if args.hex_dump:
                    print(f"  Hex dumps: {args.output}/node_XX.hex")

        else:
            # Host-to-NoC mode: single payload file
            bytes_written = create_payload(
                output=Path(args.output),
                pattern=args.pattern,
                size=args.size,
                seed=args.seed,
                value=args.value,
                start_addr=args.start_addr,
                hex_dump=args.hex_dump,
            )

            if not args.quiet:
                print(f"Generated {bytes_written} bytes to {args.output}")
                print(f"  Pattern: {args.pattern}")
                if args.pattern == "random":
                    print(f"  Seed: {args.seed}")
                if args.pattern == "constant":
                    print(f"  Value: 0x{args.value:02X}")
                if args.pattern == "address":
                    print(f"  Start Address: 0x{args.start_addr:08X}")
                if args.hex_dump:
                    print(f"  Hex dump: {Path(args.output).with_suffix('.hex')}")

    except (ValueError, OSError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
