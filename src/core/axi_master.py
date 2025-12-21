"""
AXI Master Controller for File-In/File-Out.

Implements a DMA-like controller that reads data from Host Memory
and generates AXI transactions to transfer data to NoC nodes.
Handles burst splitting according to AXI spec (4KB boundary, max burst length).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Iterator, Tuple, TYPE_CHECKING
from enum import Enum
import math

from ..config import TransferConfig, TransferMode
from ..axi.interface import (
    AXI_AW, AXI_W, AXI_B, AXI_AR, AXI_R,
    AXISize, AXIBurst,
    AXIWriteTransaction, create_write_transaction,
    AXIReadTransaction, create_read_transaction,
)
from ..address.address_map import SystemAddressMap, create_address_map

if TYPE_CHECKING:
    from .memory import Memory


# =============================================================================
# AXI ID Generation
# =============================================================================

@dataclass
class AXIIdConfig:
    """AXI ID generation configuration."""
    id_width: int = 4           # ID bits (e.g., 4 = IDs 0-15)
    cyclic: bool = True         # Enable cyclic ID generation
    start_id: int = 0           # Starting ID

    def __post_init__(self):
        if self.id_width < 1 or self.id_width > 16:
            raise ValueError("id_width must be between 1 and 16")
        max_id = (1 << self.id_width) - 1
        if self.start_id < 0 or self.start_id > max_id:
            raise ValueError(f"start_id must be between 0 and {max_id}")


class AXIIdGenerator:
    """
    Cyclic AXI ID generator.

    Generates AXI transaction IDs in a cyclic manner (0 -> max_id -> 0 -> ...).
    Also tracks which IDs are currently in-flight to prevent ID conflicts.
    """

    def __init__(self, config: Optional[AXIIdConfig] = None):
        """
        Initialize AXI ID Generator.

        Args:
            config: AXI ID configuration. Uses defaults if not provided.
        """
        self.config = config or AXIIdConfig()
        self.max_id = (1 << self.config.id_width) - 1
        self._next_id = self.config.start_id
        self._in_flight: set = set()  # Track IDs currently in use

    def get_next_id(self) -> int:
        """
        Get next available AXI ID.

        Returns:
            Next AXI ID (cyclic if enabled).

        Raises:
            RuntimeError: If all IDs are in-flight (no available ID).
        """
        # Find next available ID
        attempts = 0
        while self._next_id in self._in_flight:
            self._next_id = (self._next_id + 1) % (self.max_id + 1)
            attempts += 1
            if attempts > self.max_id:
                raise RuntimeError("All AXI IDs are in-flight, cannot allocate new ID")

        current = self._next_id
        self._in_flight.add(current)

        if self.config.cyclic:
            self._next_id = (self._next_id + 1) % (self.max_id + 1)
        else:
            self._next_id = min(self._next_id + 1, self.max_id)

        return current

    def release_id(self, axi_id: int) -> None:
        """
        Release an AXI ID (transaction completed).

        Args:
            axi_id: AXI ID to release.
        """
        self._in_flight.discard(axi_id)

    def is_in_flight(self, axi_id: int) -> bool:
        """Check if an AXI ID is currently in-flight."""
        return axi_id in self._in_flight

    @property
    def available_ids(self) -> int:
        """Number of available (not in-flight) IDs."""
        return self.max_id + 1 - len(self._in_flight)

    @property
    def in_flight_count(self) -> int:
        """Number of IDs currently in-flight."""
        return len(self._in_flight)

    def reset(self) -> None:
        """Reset to starting state."""
        self._next_id = self.config.start_id
        self._in_flight.clear()


class AXIMasterState(Enum):
    """AXI Master Controller state."""
    IDLE = "idle"
    GENERATING = "generating"
    WAITING = "waiting"
    COMPLETE = "complete"


@dataclass
class PendingTransaction:
    """Track a pending AXI write transaction."""
    txn_id: int
    node_id: int
    dst_addr: int
    data: bytes
    inject_cycle: int
    complete_cycle: int = 0

    @property
    def is_complete(self) -> bool:
        return self.complete_cycle > 0

    @property
    def latency(self) -> int:
        if self.complete_cycle > 0:
            return self.complete_cycle - self.inject_cycle
        return 0


@dataclass
class PendingReadTransaction:
    """Track a pending AXI read transaction."""
    txn_id: int
    node_id: int
    local_addr: int
    read_size: int
    inject_cycle: int
    complete_cycle: int = 0
    
    # New fields for out-of-order reconstruction
    base_addr: int = 0
    total_read_size: int = 0
    buffer: Optional[bytearray] = None
    
    received_data_internal: bytes = field(default_factory=bytes) # Raw data for this burst
    expected_data: Optional[bytes] = None

    @property
    def received_data(self) -> bytes:
        """
        Return the received data.
        If a shared buffer is provided, return the full logical read data.
        Otherwise, return the raw data for this burst.
        """
        if self.buffer is not None:
            return bytes(self.buffer)
        return self.received_data_internal

    @property
    def is_complete(self) -> bool:
        return self.complete_cycle > 0

    @property
    def latency(self) -> int:
        if self.complete_cycle > 0:
            return self.complete_cycle - self.inject_cycle
        return 0

    @property
    def data_match(self) -> Optional[bool]:
        """Check if received data matches expected (None if no expected)."""
        if not self.is_complete or self.expected_data is None:
            return None
        # For verification, we still check the data for THIS burst if expected_data is set per burst
        return self.received_data_internal == self.expected_data


@dataclass
class AXIMasterStats:
    """Statistics for AXI Master operations."""
    # Write stats
    total_transactions: int = 0
    total_bytes: int = 0
    completed_transactions: int = 0
    completed_bytes: int = 0
    total_latency: int = 0

    # Read stats
    read_transactions: int = 0
    read_bytes_requested: int = 0
    read_completed: int = 0
    read_bytes_received: int = 0
    read_latency: int = 0
    read_matches: int = 0
    read_mismatches: int = 0

    @property
    def avg_latency(self) -> float:
        if self.completed_transactions == 0:
            return 0.0
        return self.total_latency / self.completed_transactions

    @property
    def avg_read_latency(self) -> float:
        if self.read_completed == 0:
            return 0.0
        return self.read_latency / self.read_completed


class AXIMasterController:
    """
    AXI Master Controller (DMA-like).

    Reads data from Host Memory and generates AXI Write transactions
    to transfer data to NoC nodes according to TransferConfig.
    """

    # AXI 4KB boundary (burst cannot cross this boundary)
    AXI_4KB_BOUNDARY = 4096

    def __init__(
        self,
        config: TransferConfig,
        host_memory: "Memory",
        address_map: Optional[SystemAddressMap] = None,
        mesh_cols: int = 5,
        mesh_rows: int = 4,
        axi_id_config: Optional[AXIIdConfig] = None,
    ):
        """
        Initialize AXI Master Controller.

        Args:
            config: Transfer configuration.
            host_memory: Host Memory to read from.
            address_map: System address map (created if not provided).
            mesh_cols: Mesh columns (used if address_map not provided).
            mesh_rows: Mesh rows (used if address_map not provided).
            axi_id_config: AXI ID generation config (uses defaults if not provided).
        """
        self.config = config
        self.host_memory = host_memory
        self.address_map = address_map or create_address_map(mesh_cols, mesh_rows)

        # State
        self._state = AXIMasterState.IDLE
        self._current_node_idx = 0
        self._current_offset = 0

        # AXI ID Generator (cyclic)
        self._id_generator = AXIIdGenerator(axi_id_config)

        # Pending write transactions (txn_id -> PendingTransaction)
        self._pending: Dict[int, PendingTransaction] = {}
        self._completed: List[PendingTransaction] = []

        # Transaction generation queue (for writes)
        self._txn_queue: List[Tuple[int, int, int, bytes]] = []  # (node_id, dst_addr, src_offset, data)

        # Read transaction state
        self._read_txn_queue: List[Tuple[int, int, int, Optional[bytes], int, int, bytearray]] = []  # (node_id, local_addr, size, expected, base_addr, total_size, buffer)
        self._pending_reads: Dict[int, PendingReadTransaction] = {}
        self._completed_reads: List[PendingReadTransaction] = []
        self._golden_store: Dict[Tuple[int, int], bytes] = {}  # (node_id, addr) -> expected_data
        self._read_initialized = False
        self._read_buffers: Dict[Tuple[int, int], bytearray] = {}  # (node_id, base_addr) -> buffer

        # Statistics
        self.stats = AXIMasterStats()

        # Pre-compute target nodes and data size per node
        total_nodes = self.address_map.num_nodes
        self._target_nodes = config.get_target_node_list(total_nodes)
        self._compute_transfer_plan()

    def _compute_transfer_plan(self) -> None:
        """Compute the transfer plan based on config."""
        config = self.config

        if config.transfer_mode == TransferMode.BROADCAST:
            # Same data to all nodes
            self._data_per_node = config.src_size
            self._total_data = config.src_size * len(self._target_nodes)
        else:
            # Scatter: distribute data across nodes
            nodes_count = len(self._target_nodes)
            self._data_per_node = config.src_size // nodes_count
            self._total_data = self._data_per_node * nodes_count

    def _get_axi_size(self) -> AXISize:
        """Convert beat_size to AXISize enum."""
        size_map = {
            1: AXISize.SIZE_1,
            2: AXISize.SIZE_2,
            4: AXISize.SIZE_4,
            8: AXISize.SIZE_8,
            16: AXISize.SIZE_16,
            32: AXISize.SIZE_32,
            64: AXISize.SIZE_64,
            128: AXISize.SIZE_128,
        }
        return size_map.get(self.config.beat_size, AXISize.SIZE_8)

    def _split_into_bursts(
        self,
        dst_addr: int,
        data: bytes,
    ) -> Iterator[Tuple[int, bytes]]:
        """
        Split data into AXI-compliant bursts.

        Handles:
        - Max burst length (max_burst_len * beat_size)
        - 4KB boundary crossing

        Args:
            dst_addr: Destination address.
            data: Data to transfer.

        Yields:
            Tuple of (address, data) for each burst.
        """
        beat_size = self.config.beat_size
        max_burst_bytes = self.config.max_burst_len * beat_size
        offset = 0

        while offset < len(data):
            current_addr = dst_addr + offset
            remaining = len(data) - offset

            # Calculate max bytes before 4KB boundary
            boundary_offset = self.AXI_4KB_BOUNDARY - (current_addr % self.AXI_4KB_BOUNDARY)

            # Take minimum of: remaining data, max burst, boundary
            burst_size = min(remaining, max_burst_bytes, boundary_offset)

            # Align to beat size (round down)
            burst_size = (burst_size // beat_size) * beat_size
            if burst_size == 0:
                burst_size = min(remaining, beat_size)

            yield current_addr, data[offset:offset + burst_size]
            offset += burst_size

    def initialize(self) -> None:
        """Initialize the controller and prepare transaction queue."""
        self._state = AXIMasterState.GENERATING
        self._txn_queue.clear()
        self._pending.clear()
        self._completed.clear()
        self._id_generator.reset()
        self.stats = AXIMasterStats()

        config = self.config

        # Generate transaction queue
        for node_idx, node_id in enumerate(self._target_nodes):
            if config.transfer_mode == TransferMode.BROADCAST:
                # Read same data for each node
                src_offset = config.src_addr
                data_size = config.src_size
            else:
                # Scatter: each node gets different chunk
                src_offset = config.src_addr + node_idx * self._data_per_node
                data_size = self._data_per_node

            # Read data from host memory
            data, _ = self.host_memory.read(src_offset, data_size)

            # Split into bursts
            for burst_addr, burst_data in self._split_into_bursts(config.dst_addr, data):
                local_addr = burst_addr  # dst_addr is local address
                self._txn_queue.append((node_id, local_addr, src_offset, burst_data))

    def generate(self, cycle: int) -> Iterator[AXIWriteTransaction]:
        """
        Generate AXI transactions for current cycle.
        
        Limits to 1 transaction per cycle for cycle-accurate timing.

        Args:
            cycle: Current simulation cycle.

        Yields:
            AXIWriteTransaction to be submitted.
        """
        if self._state == AXIMasterState.COMPLETE:
            return

        # Generate at most 1 transaction per cycle
        if (
            len(self._pending) < self.config.max_outstanding
            and self._txn_queue
            and self._id_generator.available_ids > 0
        ):
            node_id, local_addr, src_offset, data = self._txn_queue.pop(0)

            # Build global AXI address
            global_addr = self.address_map.build_axi_addr(node_id, local_addr)

            # Create AXI transaction with cyclic ID
            txn_id = self._id_generator.get_next_id()

            txn = create_write_transaction(
                addr=global_addr,
                data=data,
                axi_id=txn_id,
                burst_size=self._get_axi_size(),
            )
            txn.timestamp_start = cycle

            # Track pending transaction
            pending = PendingTransaction(
                txn_id=txn_id,
                node_id=node_id,
                dst_addr=local_addr,
                data=data,
                inject_cycle=cycle,
            )
            self._pending[txn_id] = pending

            # Update stats
            self.stats.total_transactions += 1
            self.stats.total_bytes += len(data)

            yield txn

        # Update state
        if not self._txn_queue and not self._pending:
            self._state = AXIMasterState.COMPLETE
        elif not self._txn_queue:
            self._state = AXIMasterState.WAITING

    def handle_response(self, response: AXI_B, cycle: int) -> Optional[PendingTransaction]:
        """
        Handle AXI write response.

        Args:
            response: AXI_B response.
            cycle: Current cycle.

        Returns:
            Completed PendingTransaction or None.
        """
        txn_id = response.bid

        if txn_id not in self._pending:
            return None

        pending = self._pending.pop(txn_id)
        pending.complete_cycle = cycle

        # Release the AXI ID for reuse
        self._id_generator.release_id(txn_id)

        # Update stats
        self.stats.completed_transactions += 1
        self.stats.completed_bytes += len(pending.data)
        self.stats.total_latency += pending.latency

        self._completed.append(pending)

        # Check if complete
        if not self._txn_queue and not self._pending:
            self._state = AXIMasterState.COMPLETE

        return pending

    @property
    def is_complete(self) -> bool:
        """Check if all transactions are complete."""
        return self._state == AXIMasterState.COMPLETE

    @property
    def read_is_complete(self) -> bool:
        """Check if all read transactions are complete."""
        return (self._state == AXIMasterState.COMPLETE and
                not self._read_txn_queue and
                not self._pending_reads)

    @property
    def is_idle(self) -> bool:
        """Check if controller is idle."""
        return self._state == AXIMasterState.IDLE

    @property
    def pending_count(self) -> int:
        """Number of pending transactions."""
        return len(self._pending)

    @property
    def completed_transactions(self) -> List[PendingTransaction]:
        """List of completed transactions."""
        return self._completed

    def get_progress(self) -> float:
        """Get transfer progress (0.0 - 1.0)."""
        if self.stats.total_bytes == 0:
            return 0.0  # No transfer configured yet
        return self.stats.completed_bytes / self.stats.total_bytes

    def get_summary(self) -> Dict:
        """Get transfer summary."""
        return {
            "state": self._state.value,
            "target_nodes": self._target_nodes,
            "transfer_mode": self.config.transfer_mode.value,
            "total_transactions": self.stats.total_transactions,
            "completed_transactions": self.stats.completed_transactions,
            "pending_transactions": len(self._pending),
            "total_bytes": self.stats.total_bytes,
            "completed_bytes": self.stats.completed_bytes,
            "avg_latency": self.stats.avg_latency,
            "progress": self.get_progress(),
        }

    def print_summary(self) -> None:
        """Print transfer summary."""
        summary = self.get_summary()
        print("=" * 60)
        print("AXI Master Controller Summary")
        print("=" * 60)
        print(f"State: {summary['state']}")
        print(f"Transfer Mode: {summary['transfer_mode']}")
        print(f"Target Nodes: {summary['target_nodes']}")
        print(f"Progress: {summary['progress']*100:.1f}%")
        print()
        print(f"Transactions: {summary['completed_transactions']}/{summary['total_transactions']}")
        print(f"Bytes: {summary['completed_bytes']}/{summary['total_bytes']}")
        print(f"Average Latency: {summary['avg_latency']:.2f} cycles")
        print("=" * 60)

    # =========================================================================
    # Read Transaction Support
    # =========================================================================

    def set_golden_data(
        self,
        node_id: int,
        local_addr: int,
        data: bytes
    ) -> None:
        """
        Set expected (golden) data for a specific read location.

        Args:
            node_id: Target node ID.
            local_addr: Local memory address.
            data: Expected data.
        """
        self._golden_store[(node_id, local_addr)] = data

    def set_golden_store(
        self,
        golden_store: Dict[Tuple[int, int], bytes]
    ) -> None:
        """
        Set all golden data from a dictionary.

        Args:
            golden_store: Dict of (node_id, local_addr) -> expected data.
        """
        self._golden_store = golden_store.copy()

    def initialize_read(
        self,
        read_config: Optional[TransferConfig] = None,
        golden_store: Optional[Dict[Tuple[int, int], bytes]] = None
    ) -> None:
        """
        Initialize read transaction queue.

        Args:
            read_config: Read configuration (uses self.config if not provided).
            golden_store: Dictionary of (node_id, local_addr) -> expected_data.
        """
        config = read_config or self.config

        # Must be a read mode
        if not config.is_read:
            raise ValueError(f"TransferMode {config.transfer_mode} is not a read mode")

        self._state = AXIMasterState.GENERATING
        self._read_txn_queue.clear()
        self._pending_reads.clear()
        self._completed_reads.clear()
        self._read_buffers.clear()
        self._read_initialized = True

        # Merge golden stores
        if golden_store:
            self._golden_store.update(golden_store)

        read_size = config.effective_read_size
        read_addr = config.read_src_addr

        # Generate read transaction queue based on transfer mode
        for node_idx, node_id in enumerate(self._target_nodes):
            if config.transfer_mode == TransferMode.BROADCAST_READ:
                # Same address for all nodes
                addr = read_addr
                size = read_size
            else:  # GATHER
                # Different offset for each node
                size_per_node = read_size // len(self._target_nodes)
                addr = read_addr
                size = size_per_node

            # Get expected data if available
            expected = self._golden_store.get((node_id, addr))

            # Split into bursts
            for burst_addr, burst_size in self._split_into_read_bursts(addr, size):
                # Ensure we have a buffer for this node/base_addr
                key = (node_id, addr)
                if key not in self._read_buffers:
                    self._read_buffers[key] = bytearray(size)
                
                buffer = self._read_buffers[key]
                self._read_txn_queue.append((node_id, burst_addr, burst_size, expected, addr, size, buffer))

    def _split_into_read_bursts(
        self,
        addr: int,
        size: int,
    ) -> Iterator[Tuple[int, int]]:
        """
        Split read into AXI-compliant bursts.

        Args:
            addr: Local memory address.
            size: Total bytes to read.

        Yields:
            Tuple of (address, size) for each burst.
        """
        beat_size = self.config.beat_size
        max_burst_bytes = self.config.max_burst_len * beat_size
        offset = 0

        while offset < size:
            current_addr = addr + offset
            remaining = size - offset

            # Calculate max bytes before 4KB boundary
            boundary_offset = self.AXI_4KB_BOUNDARY - (current_addr % self.AXI_4KB_BOUNDARY)

            # Take minimum of: remaining, max burst, boundary
            burst_size = min(remaining, max_burst_bytes, boundary_offset)

            # Align to beat size
            burst_size = (burst_size // beat_size) * beat_size
            if burst_size == 0:
                burst_size = min(remaining, beat_size)

            yield current_addr, burst_size
            offset += burst_size

    def generate_read(self, cycle: int) -> Iterator[AXIReadTransaction]:
        """
        Generate AXI read transactions for current cycle.
        
        Limits to 1 transaction per cycle for cycle-accurate timing.

        Args:
            cycle: Current simulation cycle.

        Yields:
            AXIReadTransaction to be submitted.
        """
        if not self._read_initialized:
            return

        if self._state == AXIMasterState.COMPLETE:
            return

        # Generate at most 1 read transaction per cycle
        if (
            len(self._pending_reads) < self.config.max_outstanding
            and self._read_txn_queue
            and self._id_generator.available_ids > 0
        ):
            node_id, local_addr, read_size, expected, base_addr, total_size, buffer = self._read_txn_queue.pop(0)

            # Build global AXI address
            global_addr = self.address_map.build_axi_addr(node_id, local_addr)

            # Create AXI read transaction with cyclic ID
            txn_id = self._id_generator.get_next_id()

            txn = create_read_transaction(
                addr=global_addr,
                size=read_size,
                axi_id=txn_id,
                burst_size=self._get_axi_size(),
            )
            txn.timestamp_start = cycle

            # Track pending read transaction
            pending = PendingReadTransaction(
                txn_id=txn_id,
                node_id=node_id,
                local_addr=local_addr,
                read_size=read_size,
                inject_cycle=cycle,
                expected_data=expected,
                base_addr=base_addr,
                total_read_size=total_size,
                buffer=buffer,
            )
            self._pending_reads[txn_id] = pending

            # Update stats
            self.stats.read_transactions += 1
            self.stats.read_bytes_requested += read_size

            yield txn

        # Update state
        if not self._read_txn_queue and not self._pending_reads:
            self._state = AXIMasterState.COMPLETE
        elif not self._read_txn_queue:
            self._state = AXIMasterState.WAITING

    def handle_read_response(
        self,
        response: AXI_R,
        cycle: int
    ) -> Optional[PendingReadTransaction]:
        """
        Handle AXI read response.

        Args:
            response: AXI_R response.
            cycle: Current cycle.

        Returns:
            Completed PendingReadTransaction if rlast, else None.
        """
        txn_id = response.rid

        if txn_id not in self._pending_reads:
            return None

        pending = self._pending_reads[txn_id]

        # Accumulate received data for this burst
        pending.received_data_internal = pending.received_data_internal + response.rdata

        # Also place in shared buffer at correct offset
        if pending.buffer is not None:
            # Calculate offset within the logical read for this node
            # pending.local_addr is the start of THIS burst
            # pending.base_addr is the start of the logical read
            # response.rdata position within burst is tracked by current length of internal data
            burst_offset = len(pending.received_data_internal) - len(response.rdata)
            total_offset = (pending.local_addr - pending.base_addr) + burst_offset
            
            # Boundary check - trim rdata if it exceeds buffer (due to beat alignment)
            chunk_len = len(response.rdata)
            if total_offset + chunk_len > len(pending.buffer):
                chunk_len = max(0, len(pending.buffer) - total_offset)
            
            if chunk_len > 0:
                pending.buffer[total_offset:total_offset + chunk_len] = response.rdata[:chunk_len]

        # Check if complete (rlast)
        if response.rlast:
            pending.complete_cycle = cycle
            self._id_generator.release_id(txn_id)

            # Update stats
            self.stats.read_completed += 1
            self.stats.read_bytes_received += len(pending.received_data)
            self.stats.read_latency += pending.latency

            # Check verification result
            if pending.expected_data is not None:
                if pending.data_match:
                    self.stats.read_matches += 1
                else:
                    self.stats.read_mismatches += 1

            del self._pending_reads[txn_id]
            self._completed_reads.append(pending)

            # Check if all reads complete
            if not self._read_txn_queue and not self._pending_reads:
                self._state = AXIMasterState.COMPLETE

            return pending

        return None

    @property
    def read_is_complete(self) -> bool:
        """Check if all read transactions are complete."""
        if not self._read_initialized:
            return True
        return (
            not self._read_txn_queue
            and not self._pending_reads
            and self._state == AXIMasterState.COMPLETE
        )

    @property
    def completed_reads(self) -> List[PendingReadTransaction]:
        """List of completed read transactions."""
        return self._completed_reads

    def get_read_data(self) -> Dict[Tuple[int, int], bytes]:
        """
        Get all received read data.

        Returns:
            Dictionary of (node_id, local_addr) -> received_data.
        """
        result = {}
        for txn in self._completed_reads:
            result[(txn.node_id, txn.local_addr)] = txn.received_data
        return result

    def get_read_progress(self) -> float:
        """Get read transfer progress (0.0 - 1.0)."""
        if self.stats.read_bytes_requested == 0:
            return 0.0  # No read transfer configured yet
        return self.stats.read_bytes_received / self.stats.read_bytes_requested
