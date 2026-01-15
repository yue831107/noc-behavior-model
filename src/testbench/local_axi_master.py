"""
Local AXI Master for NoC-to-NoC transfers.

Unlike HostAXIMaster which uses address encoding (node_id << 32 | local_addr),
LocalAXIMaster uses AXI user signal for destination routing.

Key Differences from HostAXIMaster:
- Address: 32-bit local address (not 64-bit global)
- Destination: Encoded in awuser[15:0] as (dest_y << 8) | dest_x

AXI Compliance:
- Respects max burst length (configurable, default 256 for AXI4)
- Handles 4KB boundary crossing
- One AXI beat per cycle (AW or W, not both)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque, TYPE_CHECKING
from collections import deque
from enum import Enum

from ..axi.interface import (
    AXI_AW, AXI_W, AXI_B, AXI_AR, AXI_R,
    AXISize, AXIBurst, AXIResp,
)

if TYPE_CHECKING:
    from .memory import Memory
    from .ni import SlaveNI


class LocalAXIMasterState(Enum):
    """Local AXI Master state."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass
class LocalTransferConfig:
    """Configuration for a single local AXI transfer."""
    dest_coord: Tuple[int, int]     # (x, y) destination coordinate
    local_src_addr: int = 0x0000    # Source address in local memory
    local_dst_addr: int = 0x1000    # Destination address at target
    transfer_size: int = 256        # Bytes to transfer

    def encode_user_signal(self) -> int:
        """Encode destination coordinate into AXI user signal.

        Format: awuser[7:0] = dest_x, awuser[15:8] = dest_y
        """
        dest_x, dest_y = self.dest_coord
        return (dest_y << 8) | dest_x

    @staticmethod
    def decode_user_signal(user_signal: int) -> Tuple[int, int]:
        """Decode AXI user signal to destination coordinate.

        Args:
            user_signal: AXI user signal value.

        Returns:
            Tuple of (dest_x, dest_y).
        """
        dest_x = user_signal & 0xFF
        dest_y = (user_signal >> 8) & 0xFF
        return (dest_x, dest_y)


@dataclass
class LocalAXIMasterStats:
    """Statistics for Local AXI Master."""
    aw_sent: int = 0
    w_sent: int = 0
    b_received: int = 0
    total_cycles: int = 0
    first_aw_cycle: int = 0
    last_b_cycle: int = 0
    transactions_completed: int = 0


@dataclass
class PendingBurst:
    """A single AXI burst transaction."""
    axi_id: int
    dst_addr: int           # Destination address for this burst
    data: bytes             # Data for this burst
    user_signal: int        # AXI user signal (destination encoding)

    # State tracking
    aw_sent: bool = False
    w_beats_sent: int = 0
    w_beats_total: int = 0
    b_received: bool = False


class LocalAXIMaster:
    """
    Local AXI Master for NoC-to-NoC transfers.

    Each node has one LocalAXIMaster for initiating transfers to other nodes.
    Uses AXI user signal for destination routing instead of address encoding.

    AXI Compliance:
    - Max burst length respected (configurable)
    - 4KB boundary crossing handled
    - One beat per cycle (cycle-accurate)

    Signal Flow:
        LocalMemory -> LocalAXIMaster -> SlaveNI -> Mesh -> MasterNI -> LocalMemory
                                             |                              |
                                             <-------- Response ------------->
    """

    # AXI 4KB boundary (burst cannot cross this boundary)
    AXI_4KB_BOUNDARY = 4096

    def __init__(
        self,
        node_id: int,
        local_memory: "Memory",
        mesh_cols: int = 5,
        mesh_rows: int = 4,
        max_burst_len: int = 256,  # AXI4 max is 256
        beat_size: int = 32,       # Flit payload size
    ):
        """
        Initialize Local AXI Master.

        Args:
            node_id: This node's ID.
            local_memory: Local memory to read source data from.
            mesh_cols: Mesh columns (for coordinate calculation).
            mesh_rows: Mesh rows (for coordinate calculation).
            max_burst_len: Maximum burst length (1-256 for AXI4).
            beat_size: Bytes per beat (flit payload size).
        """
        self.node_id = node_id
        self.local_memory = local_memory
        self.mesh_cols = mesh_cols
        self.mesh_rows = mesh_rows
        self.max_burst_len = min(max_burst_len, 256)  # AXI4 limit
        self.beat_size = beat_size

        # Calculate this node's coordinate
        self.src_coord = self._node_id_to_coord(node_id)

        # Transfer configuration
        self._transfer_config: Optional[LocalTransferConfig] = None

        # Connected SlaveNI
        self._slave_ni: Optional["SlaveNI"] = None

        # State
        self._state = LocalAXIMasterState.IDLE
        self._current_cycle = 0

        # Burst queue (multiple AXI transactions from splitting)
        self._burst_queue: Deque[PendingBurst] = deque()
        self._current_burst: Optional[PendingBurst] = None

        # AXI ID counter
        self._next_axi_id = 0

        # Statistics
        self.stats = LocalAXIMasterStats()

    def _node_id_to_coord(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (x, y) coordinate.

        Note: Assumes edge_column=0, so compute nodes start at x=1.
        """
        compute_cols = self.mesh_cols - 1  # Exclude edge column
        x = (node_id % compute_cols) + 1   # +1 to skip edge column
        y = node_id // compute_cols
        return (x, y)

    def _coord_to_node_id(self, coord: Tuple[int, int]) -> int:
        """Convert (x, y) coordinate to node ID."""
        x, y = coord
        compute_cols = self.mesh_cols - 1
        return y * compute_cols + (x - 1)

    def connect_to_slave_ni(self, slave_ni: "SlaveNI") -> None:
        """Connect to local SlaveNI."""
        self._slave_ni = slave_ni

    def configure_transfer(self, config: LocalTransferConfig) -> None:
        """
        Configure transfer parameters.

        Args:
            config: Transfer configuration.
        """
        self._transfer_config = config

    def reset(self) -> None:
        """Reset master to IDLE state."""
        self._state = LocalAXIMasterState.IDLE
        self._current_cycle = 0
        self._burst_queue.clear()
        self._current_burst = None
        self.stats = LocalAXIMasterStats()

    def _split_into_bursts(
        self,
        dst_addr: int,
        data: bytes,
        user_signal: int,
    ) -> List[PendingBurst]:
        """
        Split data into AXI-compliant bursts.

        Handles:
        - Max burst length (max_burst_len * beat_size)
        - 4KB boundary crossing

        Args:
            dst_addr: Destination address.
            data: Data to transfer.
            user_signal: AXI user signal for destination routing.

        Returns:
            List of PendingBurst objects.
        """
        bursts = []
        max_burst_bytes = self.max_burst_len * self.beat_size
        offset = 0

        while offset < len(data):
            current_addr = dst_addr + offset
            remaining = len(data) - offset

            # Calculate max bytes before 4KB boundary
            boundary_offset = self.AXI_4KB_BOUNDARY - (current_addr % self.AXI_4KB_BOUNDARY)

            # Take minimum of: remaining data, max burst, boundary
            burst_size = min(remaining, max_burst_bytes, boundary_offset)

            # Align to beat size (round down), but ensure at least one beat
            aligned_size = (burst_size // self.beat_size) * self.beat_size
            if aligned_size == 0:
                aligned_size = min(remaining, self.beat_size)

            # Create burst
            burst_data = data[offset:offset + aligned_size]
            num_beats = (len(burst_data) + self.beat_size - 1) // self.beat_size

            axi_id = self._next_axi_id
            self._next_axi_id = (self._next_axi_id + 1) % 16

            burst = PendingBurst(
                axi_id=axi_id,
                dst_addr=current_addr,
                data=burst_data,
                user_signal=user_signal,
                w_beats_total=num_beats,
            )
            bursts.append(burst)
            offset += aligned_size

        return bursts

    def start(self) -> None:
        """Start the configured transfer."""
        if self._state != LocalAXIMasterState.IDLE:
            return
        if self._transfer_config is None:
            return

        # Read data from local memory
        config = self._transfer_config
        data, _ = self.local_memory.read(config.local_src_addr, config.transfer_size)

        # Split into AXI-compliant bursts
        user_signal = config.encode_user_signal()
        bursts = self._split_into_bursts(config.local_dst_addr, data, user_signal)

        # Initialize burst queue
        self._burst_queue = deque(bursts)
        self._current_burst = None

        # Reset state
        self._state = LocalAXIMasterState.RUNNING
        self._current_cycle = 0
        self.stats = LocalAXIMasterStats()

    def process_cycle(self, cycle: int) -> None:
        """
        Process one simulation cycle.

        Cycle-accurate behavior:
        - At most ONE AXI beat (AW or W) sent per cycle
        - B responses can be received in parallel

        Args:
            cycle: Current simulation cycle.
        """
        if self._state != LocalAXIMasterState.RUNNING:
            return

        self._current_cycle = cycle
        self.stats.total_cycles = cycle + 1

        # Get current burst or start next one
        if self._current_burst is None:
            if self._burst_queue:
                self._current_burst = self._burst_queue.popleft()
            else:
                # All bursts complete
                self._state = LocalAXIMasterState.COMPLETE
                return

        burst = self._current_burst

        # IMPORTANT: Only ONE of AW or W can be sent per cycle
        beat_sent_this_cycle = False

        # Phase 1: Send AW (Write Address) - only if no beat sent yet
        if not burst.aw_sent and not beat_sent_this_cycle:
            if self._try_send_aw(burst, cycle):
                beat_sent_this_cycle = True

        # Phase 2: Send W (Write Data) - only if no beat sent yet this cycle
        if burst.aw_sent and not beat_sent_this_cycle:
            if burst.w_beats_sent < burst.w_beats_total:
                if self._try_send_w(burst, cycle):
                    beat_sent_this_cycle = True

        # Phase 3: Receive B (Write Response) - can happen in parallel
        if burst.w_beats_sent >= burst.w_beats_total and not burst.b_received:
            self._try_receive_b(burst, cycle)

        # Check if current burst is complete
        if burst.b_received:
            self.stats.transactions_completed += 1
            self._current_burst = None

            # Check if all transfers done
            if not self._burst_queue:
                self._state = LocalAXIMasterState.COMPLETE

    def _try_send_aw(self, burst: PendingBurst, cycle: int) -> bool:
        """
        Try to send AW channel for a burst.

        Returns:
            True if AW was sent this cycle.
        """
        if self._slave_ni is None:
            return False

        # awlen = burst_length - 1 (number of W beats minus 1)
        aw = AXI_AW(
            awid=burst.axi_id,
            awaddr=burst.dst_addr,
            awlen=burst.w_beats_total - 1,
            awsize=AXISize.SIZE_32,  # 32 bytes per beat
            awburst=AXIBurst.INCR,
            awuser=burst.user_signal,
        )

        if self._slave_ni.process_aw(aw, cycle):
            burst.aw_sent = True
            self.stats.aw_sent += 1
            if self.stats.first_aw_cycle == 0:
                self.stats.first_aw_cycle = cycle
            return True
        return False

    def _try_send_w(self, burst: PendingBurst, cycle: int) -> bool:
        """
        Try to send one W beat for a burst.

        Returns:
            True if W beat was sent this cycle.
        """
        if self._slave_ni is None:
            return False

        # Calculate offset and chunk for this beat
        offset = burst.w_beats_sent * self.beat_size
        chunk = burst.data[offset:offset + self.beat_size]

        # This is the last beat if it's the final chunk
        is_last = (burst.w_beats_sent == burst.w_beats_total - 1)

        # Calculate strb based on valid bytes in this chunk
        valid_bytes = len(chunk)
        if valid_bytes >= self.beat_size:
            strb = (1 << self.beat_size) - 1  # All bytes valid
        else:
            strb = (1 << valid_bytes) - 1  # Only valid bytes

        # Create W beat for this chunk
        w = AXI_W(
            wdata=chunk,
            wstrb=strb,
            wlast=is_last,
        )

        if self._slave_ni.process_w(w, burst.axi_id, cycle):
            burst.w_beats_sent += 1
            self.stats.w_sent += 1
            return True
        return False

    def _try_receive_b(self, burst: PendingBurst, cycle: int) -> None:
        """Try to receive B response for a burst."""
        if self._slave_ni is None:
            return

        b_resp = self._slave_ni.get_b_response()
        if b_resp is not None and b_resp.bid == burst.axi_id:
            burst.b_received = True
            self.stats.b_received += 1
            self.stats.last_b_cycle = cycle

    @property
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self._state == LocalAXIMasterState.COMPLETE

    @property
    def is_idle(self) -> bool:
        """Check if master is idle."""
        return self._state == LocalAXIMasterState.IDLE

    @property
    def is_running(self) -> bool:
        """Check if master is running."""
        return self._state == LocalAXIMasterState.RUNNING

    def get_summary(self) -> Dict:
        """Get transfer summary."""
        return {
            "node_id": self.node_id,
            "src_coord": self.src_coord,
            "state": self._state.value,
            "dest_coord": self._transfer_config.dest_coord if self._transfer_config else None,
            "timing": {
                "total_cycles": self.stats.total_cycles,
                "first_aw_cycle": self.stats.first_aw_cycle,
                "last_b_cycle": self.stats.last_b_cycle,
            },
            "stats": {
                "aw_sent": self.stats.aw_sent,
                "w_sent": self.stats.w_sent,
                "b_received": self.stats.b_received,
                "transactions_completed": self.stats.transactions_completed,
            },
        }
