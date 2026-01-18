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
from typing import Optional, List, Dict, Tuple, Deque, Set, TYPE_CHECKING
from collections import deque
from enum import Enum

from ..axi.interface import (
    AXI_AW, AXI_W, AXI_B, AXI_AR, AXI_R,
    AXISize, AXIBurst, AXIResp,
)
from ..core.router import ChannelMode

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

    # Timing (for latency tracking)
    aw_sent_cycle: int = 0
    first_w_cycle: int = 0
    last_w_cycle: int = 0
    b_received_cycle: int = 0


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

        # True AXI Outstanding support
        # 按狀態追蹤 bursts
        self._aw_pending: Deque[PendingBurst] = deque()   # AW 尚未送出
        self._w_active: Dict[int, PendingBurst] = {}      # W 進行中 (by axi_id)
        self._b_pending: Dict[int, PendingBurst] = {}     # 等待 B response

        # W 發送順序 (FIFO - 按 AW 發送順序)
        self._w_send_order: Deque[int] = deque()

        # AXI ID counter
        self._next_axi_id = 0

        # Outstanding limit
        self._max_outstanding = 16

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
        self._aw_pending.clear()
        self._w_active.clear()
        self._b_pending.clear()
        self._w_send_order.clear()
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

        # Initialize state - all bursts start in aw_pending
        self._aw_pending = deque(bursts)
        self._w_active.clear()
        self._b_pending.clear()
        self._w_send_order.clear()

        # Reset state
        self._state = LocalAXIMasterState.RUNNING
        self._current_cycle = 0
        self.stats = LocalAXIMasterStats()

    def process_cycle(self, cycle: int) -> None:
        """
        Process one simulation cycle with true AXI outstanding pipelining.

        Pipelined behavior (AW first, then W):
        - Phase 1 (AW): Send if outstanding < max_outstanding (不等 W 完成)
        - Phase 2 (W): FIFO 順序發送 (按 AW 發送順序，AXI4 不支援 W interleaving)
        - Phase 3 (B): Receive B responses

        Channel Mode 差異:
        - General Mode: AW/W 互斥 (共用 Request channel)
        - AXI Mode: AW + W 可並行 (獨立 channel)

        Note: AW pipelining is safe in General Mode because Router's wormhole
        locking only triggers on W/R flits, not on AW/AR flits.

        Args:
            cycle: Current simulation cycle.
        """
        if self._state != LocalAXIMasterState.RUNNING:
            return

        self._current_cycle = cycle
        self.stats.total_cycles = cycle + 1

        # 判斷是否為 AXI Mode
        is_axi_mode = (self._slave_ni is not None and
                       self._slave_ni.config.channel_mode == ChannelMode.AXI)
        aw_sent_this_cycle = False
        w_sent_this_cycle = False

        # === Phase 1: Send AW (pipelined - 不等 W 完成) ===
        outstanding_count = len(self._w_active) + len(self._b_pending)

        if self._aw_pending and outstanding_count < self._max_outstanding:
            burst = self._aw_pending[0]
            if self._try_send_aw(burst, cycle):
                self._aw_pending.popleft()
                burst.aw_sent_cycle = cycle
                self._w_active[burst.axi_id] = burst
                self._w_send_order.append(burst.axi_id)
                aw_sent_this_cycle = True

        # === Phase 2: Send W beats (FIFO - 按 AW 順序) ===
        # General Mode: 若已送 AW 則跳過 W (互斥)
        # AXI Mode: 可同時送 AW + W (並行)
        can_send_w = is_axi_mode or not aw_sent_this_cycle
        if self._w_send_order and can_send_w:
            axi_id = self._w_send_order[0]
            if axi_id in self._w_active:
                burst = self._w_active[axi_id]
                if burst.w_beats_sent < burst.w_beats_total:
                    if self._try_send_w(burst, cycle):
                        if burst.first_w_cycle == 0:
                            burst.first_w_cycle = cycle
                        w_sent_this_cycle = True
                        if burst.w_beats_sent >= burst.w_beats_total:
                            burst.last_w_cycle = cycle
                            del self._w_active[axi_id]
                            self._w_send_order.popleft()
                            self._b_pending[axi_id] = burst

        # === Phase 3: Receive B responses ===
        self._try_receive_b_responses(cycle)

        # === Phase 4: Check completion ===
        if not self._aw_pending and not self._w_active and not self._b_pending:
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

    def _try_receive_b_responses(self, cycle: int) -> None:
        """Try to receive B responses for pending bursts."""
        if self._slave_ni is None:
            return

        # Try to receive B responses for any pending burst
        b_resp = self._slave_ni.get_b_response()
        if b_resp is not None:
            axi_id = b_resp.bid
            if axi_id in self._b_pending:
                burst = self._b_pending[axi_id]
                burst.b_received = True
                burst.b_received_cycle = cycle
                del self._b_pending[axi_id]

                self.stats.b_received += 1
                self.stats.last_b_cycle = cycle
                self.stats.transactions_completed += 1

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
