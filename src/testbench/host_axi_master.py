"""
Host AXI Master with AXI Channel Interface.

Provides a DMA-like controller that reads from Host Memory and sends
AXI transactions to SlaveNI. Implements valid/ready handshake for
each AXI channel.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable, Deque, Set, TYPE_CHECKING
from enum import Enum

from ..config import TransferConfig
from ..axi.interface import (
    AXI_AW, AXI_W, AXI_B, AXI_AR, AXI_R,
    AXIWriteTransaction,
)
from .axi_master import (
    AXIMasterController,
    AXIIdConfig,
    AXIMasterStats,
    PendingReadTransaction,
)

from ..core.router import ChannelMode

if TYPE_CHECKING:
    from .memory import Memory
    from .ni import SlaveNI


class HostAXIMasterState(Enum):
    """Host AXI Master state."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass
class AXIChannelPort:
    """
    AXI Channel Port with valid/ready handshake.

    Implements standard AXI valid/ready protocol for a single channel.
    Used for AW, W, AR channels (master to slave direction).
    """
    # Output signals (set by this port)
    out_valid: bool = False
    out_payload: Any = None  # AXI_AW, AXI_W, or AXI_AR

    # Input signal (set by connected port)
    in_ready: bool = True  # Default ready

    def can_send(self) -> bool:
        """Check if can send (ready asserted and not already valid)."""
        return self.in_ready and not self.out_valid

    def set_output(self, payload: Any) -> None:
        """Set output valid and payload."""
        self.out_valid = True
        self.out_payload = payload

    def try_handshake(self) -> bool:
        """
        Try to complete handshake.

        Returns:
            True if handshake completed (valid && ready).
        """
        if self.out_valid and self.in_ready:
            return True
        return False

    def clear_output(self) -> None:
        """Clear output after successful handshake."""
        self.out_valid = False
        self.out_payload = None


@dataclass
class AXIResponsePort:
    """
    AXI Response Channel Port (slave to master direction).

    For B and R channels where master receives responses.
    """
    # Input signals (from slave)
    in_valid: bool = False
    in_payload: Any = None  # AXI_B or AXI_R

    # Output signal (to slave)
    out_ready: bool = True  # Always ready to receive

    def has_response(self) -> bool:
        """Check if response is available."""
        return self.in_valid and self.in_payload is not None

    def get_response(self) -> Any:
        """Get and consume response."""
        if self.in_valid:
            payload = self.in_payload
            self.in_valid = False
            self.in_payload = None
            return payload
        return None


@dataclass
class HostAXIMasterStats:
    """Statistics for Host AXI Master."""
    # Write transaction counts
    aw_sent: int = 0
    w_sent: int = 0
    b_received: int = 0

    # Read transaction counts
    ar_sent: int = 0
    r_received: int = 0
    r_beats_received: int = 0

    # Backpressure counts
    aw_blocked: int = 0
    w_blocked: int = 0
    ar_blocked: int = 0

    # Timing
    total_cycles: int = 0
    first_aw_cycle: int = 0
    first_ar_cycle: int = 0
    last_b_cycle: int = 0
    last_r_cycle: int = 0

    @property
    def effective_write_throughput(self) -> float:
        """Calculate effective write throughput (transactions per cycle)."""
        if self.total_cycles == 0:
            return 0.0
        return self.b_received / self.total_cycles

    @property
    def effective_read_throughput(self) -> float:
        """Calculate effective read throughput (transactions per cycle)."""
        if self.total_cycles == 0:
            return 0.0
        return self.r_received / self.total_cycles


class HostAXIMaster:
    """
    Host AXI Master with AXI Channel Interface.

    Connects Host Memory to SlaveNI via AXI channel ports.
    Uses AXIMasterController internally for transaction generation
    and burst splitting.

    Signal Flow:
        HostMemory → HostAXIMaster → SlaveNI → Selector → Mesh → NI → Memory
                                         │                           │
                                         └───────── Response ────────┘
    """

    def __init__(
        self,
        host_memory: "Memory",
        transfer_config: TransferConfig,
        axi_id_config: Optional[AXIIdConfig] = None,
        mesh_cols: int = 5,
        mesh_rows: int = 4,
        on_transfer_start: Optional[Callable[[int, TransferConfig], None]] = None,
        on_transfer_complete: Optional[Callable[[int, TransferConfig], None]] = None,
    ):
        """
        Initialize Host AXI Master.

        Args:
            host_memory: Host Memory to read from.
            transfer_config: DMA transfer configuration.
            axi_id_config: AXI ID generation config.
            mesh_cols: Mesh columns for address map.
            mesh_rows: Mesh rows for address map.
            on_transfer_start: Optional callback called before each transfer starts.
                               Signature: (transfer_index, config) -> None.
                               Use for dynamic memory initialization.
            on_transfer_complete: Optional callback called after each transfer completes.
                               Signature: (transfer_index, config) -> None.
                               Use for per-transfer verification.
        """
        self.host_memory = host_memory
        self.transfer_config = transfer_config
        self.axi_id_config = axi_id_config or AXIIdConfig()

        # Internal controller for transaction generation
        self._controller = AXIMasterController(
            config=transfer_config,
            host_memory=host_memory,
            mesh_cols=mesh_cols,
            mesh_rows=mesh_rows,
            axi_id_config=axi_id_config,
        )

        # AXI Channel Ports (to SlaveNI)
        self._aw_port = AXIChannelPort()  # Write Address
        self._w_port = AXIChannelPort()   # Write Data
        self._ar_port = AXIChannelPort()  # Read Address

        # Response Ports (from SlaveNI)
        self._b_port = AXIResponsePort()  # Write Response
        self._r_port = AXIResponsePort()  # Read Response

        # Connected SlaveNI
        self._slave_ni: Optional["SlaveNI"] = None

        # State
        self._state = HostAXIMasterState.IDLE
        self._current_cycle = 0

        # Pending write data (axi_id -> list of W beats)
        self._pending_w_beats: Dict[int, List[AXI_W]] = {}

        # True AXI Outstanding support
        # W 發送順序 (FIFO - 按 AW 發送順序)
        self._w_send_order: Deque[int] = deque()
        # 追蹤已送 AW 但等待 B response 的 transactions
        self._awaiting_b: Set[int] = set()

        # Pending AW queue (transactions waiting to be sent)
        self._pending_aw_queue: List[AXI_AW] = []

        # Read operation state
        self._pending_ar_queue: List[AXI_AR] = []
        self._read_data: Dict[Tuple[int, int], bytes] = {}  # (node_id, addr) -> data
        self._is_read_mode = False

        # Statistics
        self.stats = HostAXIMasterStats()

        # === Multi-Transfer Queue ===
        # Queue of pending transfer configs
        self._transfer_queue: List[TransferConfig] = []
        # Index of current transfer being processed
        self._current_transfer_index: int = -1
        # Total transfers completed across all configs
        self._queue_transfers_completed: int = 0
        # Flag indicating queue mode is active
        self._queue_mode: bool = False
        # Store mesh dimensions for reconfiguring controller
        self._mesh_cols = mesh_cols
        self._mesh_rows = mesh_rows
        # Callback for dynamic initialization before each transfer
        self._on_transfer_start = on_transfer_start
        # Callback for per-transfer verification after each transfer completes
        self._on_transfer_complete = on_transfer_complete

    def connect_to_slave_ni(self, slave_ni: "SlaveNI") -> None:
        """
        Connect to SlaveNI.

        Args:
            slave_ni: SlaveNI to connect to.
        """
        self._slave_ni = slave_ni

    def reset(self) -> None:
        """Reset master to IDLE state for reuse."""
        self._state = HostAXIMasterState.IDLE
        self._current_cycle = 0
        self._pending_w_beats.clear()
        self._w_send_order.clear()
        self._awaiting_b.clear()
        self._pending_aw_queue.clear()
        self._pending_ar_queue.clear()
        self._read_data.clear()
        self._is_read_mode = False
        self.stats = HostAXIMasterStats()
        # Reset queue state
        self._transfer_queue.clear()
        self._current_transfer_index = -1
        self._queue_transfers_completed = 0
        self._queue_mode = False

    # === Multi-Transfer Queue Methods ===

    def queue_transfer(self, config: TransferConfig) -> None:
        """
        Add a single transfer to the queue.

        Args:
            config: Transfer configuration to queue.
        """
        self._transfer_queue.append(config)

    def queue_transfers(self, configs: List[TransferConfig]) -> None:
        """
        Add multiple transfers to the queue.

        Args:
            configs: List of transfer configurations to queue.
        """
        self._transfer_queue.extend(configs)

    def start_queue(self) -> None:
        """
        Start processing the transfer queue.

        Configures the first transfer and begins execution.
        Subsequent transfers start automatically when each completes.
        """
        if not self._transfer_queue:
            return

        self._queue_mode = True
        self._current_transfer_index = 0
        self._queue_transfers_completed = 0
        
        # Call callback before first transfer
        if self._on_transfer_start:
            self._on_transfer_start(0, self._transfer_queue[0])
        
        self._configure_current_transfer()
        self.start()

    def _configure_current_transfer(self) -> None:
        """Configure the controller with the current queued transfer."""
        if self._current_transfer_index >= len(self._transfer_queue):
            return

        config = self._transfer_queue[self._current_transfer_index]

        # Detect read mode from config
        self._is_read_mode = config.is_read

        # Create new controller for this transfer
        self._controller = AXIMasterController(
            config=config,
            host_memory=self.host_memory,
            mesh_cols=self._mesh_cols,
            mesh_rows=self._mesh_rows,
            axi_id_config=self.axi_id_config,
        )

        # Update our local config reference
        self.transfer_config = config

    def _advance_queue(self) -> bool:
        """
        Advance to the next transfer in queue.

        Returns:
            True if there is another transfer to process, False if queue complete.
        """
        # Call completion callback for the just-finished transfer
        completed_idx = self._current_transfer_index
        if self._on_transfer_complete and completed_idx >= 0:
            self._on_transfer_complete(
                completed_idx,
                self._transfer_queue[completed_idx]
            )
        
        self._queue_transfers_completed += 1
        self._current_transfer_index += 1

        if self._current_transfer_index < len(self._transfer_queue):
            # More transfers - reset state
            self._state = HostAXIMasterState.IDLE
            self._pending_w_beats.clear()
            self._w_send_order.clear()
            self._awaiting_b.clear()
            self._pending_aw_queue.clear()
            self._pending_ar_queue.clear()
            
            # Call callback before next transfer
            if self._on_transfer_start:
                self._on_transfer_start(
                    self._current_transfer_index,
                    self._transfer_queue[self._current_transfer_index]
                )

            self._configure_current_transfer()
            self.start()
            return True
        else:
            # Queue complete
            self._queue_mode = False
            return False

    @property
    def queue_progress(self) -> Tuple[int, int]:
        """
        Get queue progress.

        Returns:
            Tuple of (completed, total).
        """
        return (self._queue_transfers_completed, len(self._transfer_queue))

    @property
    def all_queue_transfers_complete(self) -> bool:
        """Check if all queued transfers are complete."""
        if not self._queue_mode:
            return self._state == HostAXIMasterState.COMPLETE or self._state == HostAXIMasterState.IDLE
        return self._current_transfer_index >= len(self._transfer_queue)

    def configure_read(
        self,
        golden_store: Optional[Dict[Tuple[int, int], bytes]] = None
    ) -> None:
        """
        Configure for read operation.

        Args:
            golden_store: Optional golden data for verification.
                          Dict of (node_id, local_addr) -> expected_data.
        """
        self._controller.set_golden_store(golden_store or {})
        self._is_read_mode = True

    def start(self) -> None:
        """Start the transfer."""
        if self._state != HostAXIMasterState.IDLE:
            return

        # Detect read mode from config (for both queue and non-queue modes)
        self._is_read_mode = self.transfer_config.is_read

        # Initialize controller based on mode
        if self._is_read_mode:
            self._controller.initialize_read()
        else:
            self._controller.initialize()

        self._state = HostAXIMasterState.RUNNING
        self._current_cycle = 0
        # Only reset stats on first transfer, not on queue advances
        if not self._queue_mode or self._current_transfer_index <= 0:
            self.stats = HostAXIMasterStats()
        self._read_data.clear()

    def process_cycle(self, cycle: int) -> None:
        """
        Process one simulation cycle.

        Args:
            cycle: Current simulation cycle.
        """
        if self._state != HostAXIMasterState.RUNNING:
            return

        self._current_cycle = cycle
        self.stats.total_cycles = cycle + 1

        # Phase 1: Generate new transactions from controller
        self._generate_transactions(cycle)

        # Phase 2: Send AXI requests to SlaveNI
        self._send_axi_requests(cycle)

        # Phase 3: Receive AXI responses from SlaveNI
        self._receive_axi_responses(cycle)

        # Phase 4: Check completion
        if self._is_read_mode:
            # Read mode: check read completion
            if (self._controller.read_is_complete and
                not self._pending_ar_queue):
                self._state = HostAXIMasterState.COMPLETE
                self.stats.last_r_cycle = cycle
                # Queue mode: advance to next transfer
                if self._queue_mode:
                    self._advance_queue()
        else:
            # Write mode: check write completion
            # 完成條件: 所有 AW 已送出 + 所有 W 已送出 + 所有 B 已收到
            if (self._controller.is_complete and
                not self._pending_aw_queue and
                not self._pending_w_beats and
                not self._w_send_order and
                not self._awaiting_b):
                self._state = HostAXIMasterState.COMPLETE
                self.stats.last_b_cycle = cycle
                # Queue mode: advance to next transfer
                if self._queue_mode:
                    self._advance_queue()

    def _generate_transactions(self, cycle: int) -> None:
        """Generate new AXI transactions from controller."""
        if self._is_read_mode:
            # Generate read transactions (AR only)
            for read_txn in self._controller.generate_read(cycle):
                self._pending_ar_queue.append(read_txn.ar)

                if self.stats.first_ar_cycle == 0:
                    self.stats.first_ar_cycle = cycle
        else:
            # Generate write transactions (AW + W)
            for txn in self._controller.generate(cycle):
                # Queue AW for sending
                self._pending_aw_queue.append(txn.aw)

                # Store W beats for later sending
                self._pending_w_beats[txn.aw.awid] = list(txn.w_beats)

                if self.stats.first_aw_cycle == 0:
                    self.stats.first_aw_cycle = cycle

    def _send_axi_requests(self, cycle: int) -> None:
        """
        Send AXI requests with true AXI outstanding pipelining.

        Pipelined behavior (AW first, then W):
        - Phase 1 (AW): Send if outstanding < max_outstanding (不等 W 完成)
        - Phase 2 (W): FIFO 順序發送 (按 AW 發送順序，AXI4 不支援 W interleaving)
        - Phase 3 (AR): Send read address

        Channel Mode 差異:
        - General Mode: AW/W 互斥 (共用 Request channel)
        - AXI Mode: AW + W 可並行 (獨立 channel)

        Note: AW pipelining is safe in General Mode because Router's wormhole
        locking only triggers on W/R flits, not on AW/AR flits.
        """
        if self._slave_ni is None:
            return

        # 判斷是否為 AXI Mode (可並行送 AW+W)
        is_axi_mode = self._slave_ni.config.channel_mode == ChannelMode.AXI
        aw_sent_this_cycle = False
        w_sent_this_cycle = False

        # === Phase 1: Send AW (pipelined - 不等 W 完成) ===
        current_outstanding = len(self._awaiting_b)
        max_outstanding = self.transfer_config.max_outstanding

        if self._pending_aw_queue and current_outstanding < max_outstanding:
            aw = self._pending_aw_queue[0]
            if self._slave_ni.process_aw(aw, cycle):
                self._pending_aw_queue.pop(0)
                axi_id = aw.awid
                self._w_send_order.append(axi_id)
                self._awaiting_b.add(axi_id)
                self.stats.aw_sent += 1
                aw_sent_this_cycle = True
            else:
                self.stats.aw_blocked += 1

        # === Phase 2: Send W beats (FIFO - 按 AW 順序) ===
        # General Mode: 若已送 AW 則跳過 W (互斥)
        # AXI Mode: 可同時送 AW + W (並行)
        can_send_w = is_axi_mode or not aw_sent_this_cycle
        if self._w_send_order and can_send_w:
            axi_id = self._w_send_order[0]
            if axi_id in self._pending_w_beats:
                w_beats = self._pending_w_beats[axi_id]
                if w_beats:
                    w = w_beats[0]
                    if self._slave_ni.process_w(w, axi_id, cycle):
                        w_beats.pop(0)
                        self.stats.w_sent += 1
                        w_sent_this_cycle = True
                        if w.wlast:
                            del self._pending_w_beats[axi_id]
                            self._w_send_order.popleft()
                    else:
                        self.stats.w_blocked += 1

        # === Phase 3: Send AR (Read Address) ===
        # General Mode: AR 也與 AW/W 互斥
        # AXI Mode: AR 獨立 channel
        req_sent = aw_sent_this_cycle or w_sent_this_cycle
        ar_can_send = is_axi_mode or not req_sent
        if self._pending_ar_queue and ar_can_send:
            ar = self._pending_ar_queue[0]
            if self._slave_ni.process_ar(ar, cycle):
                self._pending_ar_queue.pop(0)
                self.stats.ar_sent += 1
            else:
                self.stats.ar_blocked += 1

    def _receive_axi_responses(self, cycle: int) -> None:
        """Receive AXI responses from SlaveNI."""
        if self._slave_ni is None:
            return

        # Receive B (Write Response)
        b_resp = self._slave_ni.get_b_response()
        if b_resp is not None:
            axi_id = b_resp.bid
            # 釋放 outstanding slot
            self._awaiting_b.discard(axi_id)

            self._controller.handle_response(b_resp, cycle)
            self.stats.b_received += 1

        # Receive R (Read Response)
        while True:
            r_resp = self._slave_ni.get_r_response()
            if r_resp is None:
                break

            self.stats.r_beats_received += 1

            # Pass to controller for tracking
            completed = self._controller.handle_read_response(r_resp, cycle)
            if completed is not None:
                # Store the reconstructed read data for this node
                # completed.received_data now returns the full buffer from AXIMasterController
                base_addr = self.transfer_config.read_src_addr
                key = (completed.node_id, base_addr)
                self._read_data[key] = completed.received_data
                
                self.stats.r_received += 1
                self.stats.last_r_cycle = cycle

    @property
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self._state == HostAXIMasterState.COMPLETE

    @property
    def is_idle(self) -> bool:
        """Check if master is idle."""
        return self._state == HostAXIMasterState.IDLE

    @property
    def is_running(self) -> bool:
        """Check if master is running."""
        return self._state == HostAXIMasterState.RUNNING

    @property
    def progress(self) -> float:
        """Get transfer progress (0.0 - 1.0)."""
        if self._is_read_mode:
            return self._controller.get_read_progress()
        return self._controller.get_progress()

    @property
    def controller_stats(self) -> AXIMasterStats:
        """Get internal controller statistics."""
        return self._controller.stats

    @property
    def read_data(self) -> Dict[Tuple[int, int], bytes]:
        """Get read data collected from nodes.

        Returns:
            Dict of (node_id, local_addr) -> data bytes.
        """
        return self._read_data.copy()

    @property
    def completed_reads(self) -> List[PendingReadTransaction]:
        """Get list of completed read transactions."""
        return self._controller.completed_reads

    def get_summary(self) -> Dict:
        """Get transfer summary."""
        summary = {
            "state": self._state.value,
            "mode": "read" if self._is_read_mode else "write",
            "progress": self.progress,
            "controller": self._controller.get_summary(),
            "timing": {
                "total_cycles": self.stats.total_cycles,
            },
        }

        if self._is_read_mode:
            summary["axi_channels"] = {
                "ar_sent": self.stats.ar_sent,
                "r_received": self.stats.r_received,
                "r_beats_received": self.stats.r_beats_received,
                "ar_blocked": self.stats.ar_blocked,
            }
            summary["timing"]["first_ar_cycle"] = self.stats.first_ar_cycle
            summary["timing"]["last_r_cycle"] = self.stats.last_r_cycle
            summary["read_data_count"] = len(self._read_data)
        else:
            summary["axi_channels"] = {
                "aw_sent": self.stats.aw_sent,
                "w_sent": self.stats.w_sent,
                "b_received": self.stats.b_received,
                "aw_blocked": self.stats.aw_blocked,
                "w_blocked": self.stats.w_blocked,
            }
            summary["timing"]["first_aw_cycle"] = self.stats.first_aw_cycle
            summary["timing"]["last_b_cycle"] = self.stats.last_b_cycle

        return summary

    def print_summary(self) -> None:
        """Print transfer summary."""
        print("=" * 60)
        mode_str = "READ" if self._is_read_mode else "WRITE"
        print(f"Host AXI Master Summary ({mode_str} Mode)")
        print("=" * 60)
        print(f"State: {self._state.value}")
        print(f"Progress: {self.progress * 100:.1f}%")
        print()

        if self._is_read_mode:
            print("AXI Read Channel Statistics:")
            print(f"  AR Sent: {self.stats.ar_sent} (blocked: {self.stats.ar_blocked})")
            print(f"  R Received: {self.stats.r_received} transactions")
            print(f"  R Beats: {self.stats.r_beats_received}")
            print()
            print("Timing:")
            print(f"  Total Cycles: {self.stats.total_cycles}")
            print(f"  First AR: cycle {self.stats.first_ar_cycle}")
            print(f"  Last R: cycle {self.stats.last_r_cycle}")
            print()
            print(f"Read Data Collected: {len(self._read_data)} entries")
        else:
            print("AXI Write Channel Statistics:")
            print(f"  AW Sent: {self.stats.aw_sent} (blocked: {self.stats.aw_blocked})")
            print(f"  W Sent: {self.stats.w_sent} (blocked: {self.stats.w_blocked})")
            print(f"  B Received: {self.stats.b_received}")
            print()
            print("Timing:")
            print(f"  Total Cycles: {self.stats.total_cycles}")
            print(f"  First AW: cycle {self.stats.first_aw_cycle}")
            print(f"  Last B: cycle {self.stats.last_b_cycle}")

        print("=" * 60)

        # Also print controller summary
        self._controller.print_summary()
