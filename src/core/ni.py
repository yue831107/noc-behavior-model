"""
Network Interface (NI) implementation.

NI handles AXI to NoC protocol conversion with Req/Resp separation.

Architecture per spec.md 2.2.2:
- SlaveNI: AXI Slave interface, receives AXI Master requests, converts to Req Flit
- MasterNI: AXI Master interface, receives Req Flit, sends AXI requests to Memory

Components:
- SlaveNI: Complete Slave NI (AXI Slave side)
  - _SlaveNI_ReqPath: AXI AW/W/AR → Request Flits
  - _SlaveNI_RspPath: Response Flits → AXI B/R
- MasterNI: Complete Master NI (AXI Master side)
  - Per-ID FIFO: Track outstanding requests by AXI ID
  - AXI Master interface: Send requests to Memory
  - Routing Logic: Route responses back to source
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Deque, Callable, Any
from collections import deque
from enum import Enum, auto

from .flit import (
    Flit, FlitFactory, FlitHeader, AxiChannel,
    AxiAwPayload, AxiWPayload, AxiArPayload, AxiBPayload, AxiRPayload,
    encode_node_id, decode_node_id,
)
from .buffer import FlitBuffer, Buffer
from .packet import (
    Packet, PacketType, PacketFactory,
    PacketAssembler, PacketDisassembler
)
from .router import RouterPort, Direction, ChannelMode
from .channel_mode_strategy import get_channel_mode_strategy, ChannelModeStrategy

from ..axi.interface import (
    AXI_AW, AXI_W, AXI_B, AXI_AR, AXI_R,
    AXIResp, AXISize,
    AXIWriteTransaction, AXIReadTransaction,
)
from ..address.address_map import SystemAddressMap, AddressTranslator


# Constants for AXI transfer sizing
DEFAULT_BEAT_SIZE_BYTES = 8  # Default beat size for AXI transfers (corresponds to AXISize.SIZE_8)


@dataclass
class NIConfig:
    """Network Interface configuration."""
    # Address translation
    axi_addr_width: int = 64        # AXI address width (bits)
    local_addr_width: int = 32      # NoC local address width (bits)
    node_id_bits: int = 8           # Node ID field width (bits)

    # Transaction handling
    max_outstanding: int = 16       # Max outstanding transactions
    reorder_buffer_size: int = 32   # Reorder buffer entries

    # AXI parameters
    axi_data_width: int = 64        # AXI data width (bits)
    axi_id_width: int = 4           # AXI ID width (bits)
    burst_support: bool = True      # Support AXI burst
    max_burst_len: int = 256        # Max burst length

    # Buffer depths
    req_buffer_depth: int = 32      # Request output buffer depth
    resp_buffer_depth: int = 32     # Response input buffer depth
    r_queue_depth: int = 128        # Max R beats in AXISlave response queue (backpressure)

    # AXI input FIFO depths (AW/AR Spill Reg + W Payload Store per spec.md)
    aw_input_depth: int = 32        # AW channel input FIFO depth
    w_input_depth: int = 64         # W channel input FIFO depth (larger for burst data)
    ar_input_depth: int = 32        # AR channel input FIFO depth

    # Flit parameters
    # Must be >= PACKET_HEADER_SIZE (12 bytes) + reasonable data
    flit_payload_size: int = 32     # Flit payload size in bytes

    # NoC-to-NoC routing mode
    # If True, destination is from AXI user signal: awuser[7:0]=x, awuser[15:8]=y
    # If False (default), destination is from address encoding: addr[39:32]=node_id
    use_user_signal_routing: bool = False

    # Physical channel mode
    # GENERAL: Single req/resp buffer (AW/W/AR share, B/R share)
    # AXI: Per-channel buffers (AW, W, AR, B, R independent)
    channel_mode: ChannelMode = ChannelMode.GENERAL


@dataclass
class NIStats:
    """NI statistics."""
    # Request side
    aw_received: int = 0
    w_received: int = 0
    ar_received: int = 0
    req_flits_sent: int = 0
    write_requests: int = 0
    read_requests: int = 0

    # Response side
    resp_flits_received: int = 0
    resp_flits_dropped: int = 0
    b_responses_sent: int = 0
    r_responses_sent: int = 0

    # Latency tracking
    total_write_latency: int = 0
    total_read_latency: int = 0

    @property
    def avg_write_latency(self) -> float:
        if self.b_responses_sent == 0:
            return 0.0
        return self.total_write_latency / self.b_responses_sent

    @property
    def avg_read_latency(self) -> float:
        if self.r_responses_sent == 0:
            return 0.0
        return self.total_read_latency / self.r_responses_sent


class TransactionState(Enum):
    """State of an AXI transaction."""
    PENDING_W = auto()      # Write: waiting for W beats
    PENDING_SEND = auto()   # Waiting to send to NoC
    IN_FLIGHT = auto()      # Sent to NoC, waiting response
    COMPLETED = auto()      # Response received


@dataclass
class PendingTransaction:
    """Tracking data for an in-flight transaction."""
    axi_id: int
    rob_idx: int  # RoB index for response matching (FlooNoC style)
    is_write: bool
    state: TransactionState
    timestamp_start: int
    timestamp_end: int = 0
    src_coord: Tuple[int, int] = (0, 0)
    dest_coord: Tuple[int, int] = (0, 0)
    local_addr: int = 0

    # For write transactions
    aw: Optional[AXI_AW] = None
    w_beats: List[AXI_W] = field(default_factory=list)
    w_beats_expected: int = 0
    w_beats_received: int = 0

    # For read transactions
    ar: Optional[AXI_AR] = None
    r_beats_expected: int = 0
    r_beats_received: int = 0


# =============================================================================
# Slave NI Internal Components
# =============================================================================

class _SlaveNI_ReqPath:
    """
    Slave NI Request Path (internal component).

    Handles AXI request channels (AW, W, AR) and converts them
    to NoC request flits.

    This is the "Flit Packing" part of Slave NI per spec.md.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        address_map: SystemAddressMap,
        config: Optional[NIConfig] = None,
        ni_id: int = 0
    ):
        self.coord = coord
        self.config = config or NIConfig()
        self.ni_id = ni_id

        # Address translator (Coord Trans in spec.md)
        self.addr_translator = AddressTranslator(address_map)

        # Packet assembler (Pack AW/AR/W in spec.md)
        self.packet_assembler = PacketAssembler(self.config.flit_payload_size)

        # AXI Input FIFOs (AW/AR Spill Reg + W Payload Store per spec.md)
        # These buffer incoming AXI requests before conversion to flits
        self._aw_input_fifo: Deque[Tuple[AXI_AW, int]] = deque()  # (aw, timestamp)
        self._w_input_fifo: Deque[Tuple[AXI_W, int, int]] = deque()  # (w, axi_id, timestamp)
        self._ar_input_fifo: Deque[Tuple[AXI_AR, int]] = deque()  # (ar, timestamp)

        # Transaction tracking
        self._pending_writes: Dict[int, PendingTransaction] = {}

        # Channel mode strategy for polymorphic buffer management
        self._strategy: ChannelModeStrategy = get_channel_mode_strategy(
            self.config.channel_mode
        )

        # Output buffer for request flits
        # General Mode: Single buffer for all channels
        self.output_buffer = FlitBuffer(
            self.config.req_buffer_depth,
            f"SlaveNI({coord})_req_out"
        )

        # Per-channel output buffers (AXI Mode only)
        # Uses strategy to determine which channels need separate buffers
        self._per_channel_buffers: Dict[AxiChannel, FlitBuffer] = {}
        for ch in self._strategy.get_buffer_channels_for_request():
            self._per_channel_buffers[ch] = FlitBuffer(
                self.config.req_buffer_depth,
                f"SlaveNI({coord})_{ch.name}_out"
            )

        # Pending W flits queue (for backpressure handling)
        # Key: axi_id, Value: deque of W flits waiting to be sent
        self._pending_w_flits: Dict[int, Deque[Flit]] = {}

        # Cycle-accurate flit sending tracking for W flits
        # W flits (data) are rate-limited to 1 per cycle
        # AW/AR flits (address) are NOT rate-limited as they're small
        self._last_w_flit_cycle: int = -1

        # B RoB / R RoB: Track outstanding transactions for response matching
        self._active_transactions: Dict[int, PendingTransaction] = {}
        self._rob_to_axi: Dict[int, int] = {}  # rob_idx -> axi_id

        # ROB index counter (for response matching)
        self._next_rob_idx: int = 0

        # Statistics
        self.stats = NIStats()

        # Output port connection
        self.output_port: Optional[RouterPort] = None

    def connect_output(self, port: RouterPort) -> None:
        """Connect output to router request port."""
        self.output_port = port

    def _get_target_buffer(self, channel: AxiChannel) -> FlitBuffer:
        """Get the appropriate output buffer for a channel."""
        if self._strategy.uses_per_channel_buffers and channel in self._per_channel_buffers:
            return self._per_channel_buffers[channel]
        return self.output_buffer

    def _push_flit(self, flit: Flit) -> bool:
        """Push flit to appropriate buffer based on channel mode."""
        target = self._get_target_buffer(flit.hdr.axi_ch)
        if target.free_space > 0:
            target.push(flit)
            return True
        return False

    def _has_buffer_space(self, channel: AxiChannel) -> bool:
        """Check if buffer has space for the given channel."""
        return self._get_target_buffer(channel).free_space > 0

    def _allocate_rob_idx(self) -> int:
        """Allocate a ROB index for tracking."""
        idx = self._next_rob_idx
        self._next_rob_idx = (self._next_rob_idx + 1) % self.config.reorder_buffer_size
        return idx

    def process_aw(self, aw: AXI_AW, timestamp: int = 0) -> bool:
        """
        Process AXI Write Address channel.

        Accepts AW into input FIFO for later processing.
        Allocates rob_idx immediately so W beats can reference it.

        Args:
            aw: Write address request.
            timestamp: Current simulation time.

        Returns:
            True if accepted into FIFO.
        """
        # Check AW input FIFO space
        if len(self._aw_input_fifo) >= self.config.aw_input_depth:
            return False

        # Check outstanding limit
        if len(self._active_transactions) + len(self._pending_writes) >= self.config.max_outstanding:
            return False

        self.stats.aw_received += 1

        # Extract destination based on routing mode
        if self.config.use_user_signal_routing:
            # NoC-to-NoC mode: destination from AXI user signal
            awuser = getattr(aw, 'awuser', 0) or 0
            dest_x = awuser & 0xFF
            dest_y = (awuser >> 8) & 0xFF
            dest_coord = (dest_x, dest_y)
            local_addr = aw.awaddr & 0xFFFFFFFF
        else:
            # Host-to-NoC mode: destination from address encoding
            dest_coord, local_addr = self.addr_translator.translate(aw.awaddr)

        # Allocate ROB index now (shared with W flits)
        rob_idx = self._allocate_rob_idx()

        # Create pending transaction for tracking W beats
        txn = PendingTransaction(
            axi_id=aw.awid,
            rob_idx=rob_idx,
            is_write=True,
            state=TransactionState.PENDING_W,
            timestamp_start=timestamp,
            src_coord=self.coord,
            dest_coord=dest_coord,
            local_addr=local_addr,
            aw=aw,
            w_beats_expected=aw.burst_length,
        )

        self._pending_writes[aw.awid] = txn
        if aw.awid not in self._pending_w_flits or not self._pending_w_flits[aw.awid]:
            self._pending_w_flits[aw.awid] = deque()

        # Store in input FIFO for later flit creation
        self._aw_input_fifo.append((aw, timestamp))
        return True

    def _process_aw_fifo(self, timestamp: int) -> None:
        """Process AW requests from input FIFO, creating and sending flits."""
        while self._aw_input_fifo and self._has_buffer_space(AxiChannel.AW):
            aw, orig_timestamp = self._aw_input_fifo[0]

            # Get transaction info (may be in _pending_writes or _active_transactions)
            # Transaction may have moved to _active_transactions if wlast arrived early
            txn = self._pending_writes.get(aw.awid)
            if txn is None:
                # Check active_transactions (wlast may have moved it there)
                for rob_idx, active_txn in self._active_transactions.items():
                    if active_txn.axi_id == aw.awid:
                        txn = active_txn
                        break

            if txn is None:
                # Transaction was cancelled, skip
                self._aw_input_fifo.popleft()
                continue

            # Create AW flit
            aw_flit = FlitFactory.create_aw(
                src=txn.src_coord,
                dest=txn.dest_coord,
                addr=txn.local_addr,
                axi_id=aw.awid,
                length=aw.burst_length - 1,
                rob_idx=txn.rob_idx,
                rob_req=True,
                last=True,  # AW is single-flit packet (FlooNoC spec)
            )

            if self._push_flit(aw_flit):
                self._aw_input_fifo.popleft()
                self.stats.req_flits_sent += 1
            else:
                break  # Buffer full, try again next cycle

    def process_w(self, w: AXI_W, axi_id: int, timestamp: int = 0) -> bool:
        """
        Process AXI Write Data channel.

        Accepts W beat into input FIFO for later processing.
        Updates transaction tracking immediately for correct wlast handling.

        Args:
            w: Write data beat.
            axi_id: Associated AXI ID.
            timestamp: Current simulation time.

        Returns:
            True if accepted into FIFO.
        """
        if axi_id not in self._pending_writes:
            return False

        # Check W input FIFO space
        if len(self._w_input_fifo) >= self.config.w_input_depth:
            return False

        txn = self._pending_writes[axi_id]
        txn.w_beats.append(w)  # W Payload Store (for data tracking)
        txn.w_beats_received += 1
        self.stats.w_received += 1

        # Determine if this is the last W beat
        is_last = w.wlast or txn.w_beats_received >= txn.w_beats_expected

        # Store in input FIFO for later flit creation
        # Include seq_num for correct ordering
        seq_num = txn.w_beats_received - 1
        self._w_input_fifo.append((w, axi_id, timestamp, seq_num, is_last))

        # On wlast: complete transaction tracking
        if is_last:
            self._active_transactions[txn.rob_idx] = txn
            self._rob_to_axi[txn.rob_idx] = txn.axi_id
            self.stats.write_requests += 1
            del self._pending_writes[axi_id]

        return True

    def _process_w_fifo(self, timestamp: int) -> None:
        """Process W beats from input FIFO, creating and sending flits.

        Rate-limited to 1 W flit per cycle for cycle-accurate timing.
        """
        # Skip if W flit already sent this cycle
        if self._last_w_flit_cycle == timestamp:
            return

        if not self._w_input_fifo:
            return

        if not self._has_buffer_space(AxiChannel.W):
            return

        w, axi_id, orig_timestamp, seq_num, is_last = self._w_input_fifo[0]

        # Get transaction info (may be in _pending_writes or _active_transactions)
        txn = self._pending_writes.get(axi_id)
        if txn is None:
            # Check active transactions (for wlast case)
            for rob_idx, active_txn in self._active_transactions.items():
                if active_txn.axi_id == axi_id:
                    txn = active_txn
                    break

        if txn is None:
            # Transaction not found, skip this W beat
            self._w_input_fifo.popleft()
            return

        # Calculate strb based on actual data length
        data = w.wdata
        valid_bytes = len(data)
        flit_payload_size = self.config.flit_payload_size
        if valid_bytes >= flit_payload_size:
            strb = (1 << flit_payload_size) - 1
        else:
            strb = (1 << valid_bytes) - 1

        # Create W flit
        w_flit = FlitFactory.create_w(
            src=txn.src_coord,
            dest=txn.dest_coord,
            data=data,
            strb=strb,
            last=is_last,
            rob_idx=txn.rob_idx,
            seq_num=seq_num,
        )

        if self._push_flit(w_flit):
            self._w_input_fifo.popleft()
            self._last_w_flit_cycle = timestamp
            self.stats.req_flits_sent += 1

    def _try_send_pending_w_flits(self, timestamp: int = 0) -> None:
        """
        Try to send pending W flits from backpressure queue.

        Sends at most 1 flit per cycle for cycle-accurate timing.
        Skips if a W flit was already sent this cycle.

        Args:
            timestamp: Current simulation cycle.
        """
        # Cycle-accurate: Skip if W flit already sent this cycle
        if self._last_w_flit_cycle == timestamp:
            return

        if not self._has_buffer_space(AxiChannel.W):
            return

        # Try each transaction with pending W flits
        for axi_id in list(self._pending_w_flits.keys()):
            w_queue = self._pending_w_flits[axi_id]
            if w_queue:
                flit = w_queue.popleft()
                self._push_flit(flit)
                self._last_w_flit_cycle = timestamp  # Track for cycle-accurate
                self.stats.req_flits_sent += 1
                break  # Only 1 flit per cycle

        # Cleanup empty queues for completed transactions
        for axi_id in list(self._pending_w_flits.keys()):
            if not self._pending_w_flits[axi_id]:
                # Check if transaction is complete (all W beats received)
                if axi_id not in self._pending_writes:
                    del self._pending_w_flits[axi_id]

    def process_ar(self, ar: AXI_AR, timestamp: int = 0) -> bool:
        """
        Process AXI Read Address channel.

        Accepts AR into input FIFO for later processing.

        Args:
            ar: Read address request.
            timestamp: Current simulation time.

        Returns:
            True if accepted into FIFO.
        """
        # Check AR input FIFO space
        if len(self._ar_input_fifo) >= self.config.ar_input_depth:
            return False

        # Check outstanding limit
        if len(self._active_transactions) >= self.config.max_outstanding:
            return False

        self.stats.ar_received += 1

        # Store in input FIFO for later processing
        self._ar_input_fifo.append((ar, timestamp))
        return True

    def _process_ar_fifo(self, timestamp: int) -> None:
        """Process AR requests from input FIFO, creating and sending flits."""
        while self._ar_input_fifo and self._has_buffer_space(AxiChannel.AR):
            ar, orig_timestamp = self._ar_input_fifo[0]

            # Extract destination based on routing mode
            if self.config.use_user_signal_routing:
                aruser = getattr(ar, 'aruser', 0) or 0
                dest_x = aruser & 0xFF
                dest_y = (aruser >> 8) & 0xFF
                dest_coord = (dest_x, dest_y)
                local_addr = ar.araddr & 0xFFFFFFFF
            else:
                dest_coord, local_addr = self.addr_translator.translate(ar.araddr)

            # Allocate ROB index
            rob_idx = self._allocate_rob_idx()

            # Create packet directly (read request has no data)
            packet = PacketFactory.create_read_request(
                src=self.coord,
                dest=dest_coord,
                local_addr=local_addr,
                read_size=ar.transfer_size,
                axi_id=ar.arid,
            )

            # Create transaction tracking (R RoB entry)
            txn = PendingTransaction(
                axi_id=ar.arid,
                rob_idx=packet.rob_idx,
                is_write=False,
                state=TransactionState.PENDING_SEND,
                timestamp_start=orig_timestamp,
                src_coord=self.coord,
                dest_coord=dest_coord,
                local_addr=local_addr,
                ar=ar,
                r_beats_expected=ar.burst_length,
            )

            # Assemble into flits (Pack AR)
            flits = self.packet_assembler.assemble(packet)

            # Try to queue all flits
            all_pushed = True
            for flit in flits:
                if not self._push_flit(flit):
                    all_pushed = False
                    break

            if all_pushed:
                self._ar_input_fifo.popleft()
                self._active_transactions[packet.rob_idx] = txn
                self._rob_to_axi[packet.rob_idx] = ar.arid
                self.stats.read_requests += 1
            else:
                break  # Buffer full, try again next cycle

    def get_output_flit(self, current_cycle: int = 0) -> Optional[Flit]:
        """
        Get next flit to send to NoC.

        Args:
            current_cycle: Current simulation cycle (for injection tracking).

        Returns:
            Flit if available, None otherwise.
        """
        if self._strategy.uses_per_channel_buffers:
            # AXI Mode: check per-channel buffers in priority order
            for channel in self._strategy.request_channels:
                buffer = self._per_channel_buffers.get(channel)
                if buffer and not buffer.is_empty():
                    flit = buffer.pop()
                    if flit is not None:
                        self._finalize_flit_send(flit, current_cycle)
                        return flit
            return None
        else:
            # General Mode: use single shared buffer
            flit = self.output_buffer.pop()
            if flit is not None:
                self._finalize_flit_send(flit, current_cycle)
            return flit

    def get_channel_flit(
        self,
        channel: AxiChannel,
        current_cycle: int = 0
    ) -> Optional[Flit]:
        """
        Get next flit for a specific AXI channel (AXI Mode).

        Args:
            channel: AXI channel to get flit from.
            current_cycle: Current simulation cycle.

        Returns:
            Flit if available, None otherwise.
        """
        if not self._strategy.uses_per_channel_buffers:
            # General Mode: fall back to single buffer
            return self.get_output_flit(current_cycle)

        if channel not in self._per_channel_buffers:
            return None

        flit = self._per_channel_buffers[channel].pop()
        if flit is not None:
            self._finalize_flit_send(flit, current_cycle)
        return flit

    def _finalize_flit_send(self, flit: Flit, current_cycle: int) -> None:
        """Common finalization after sending a flit."""
        self.stats.req_flits_sent += 1

        # Update transaction state
        if flit.hdr.rob_idx in self._active_transactions:
            txn = self._active_transactions[flit.hdr.rob_idx]
            txn.state = TransactionState.IN_FLIGHT

        # BookSim2-style: Store injection cycle on flit for latency calculation
        flit.injection_cycle = current_cycle

    def mark_transaction_complete(self, rob_idx: int, timestamp: int) -> Optional[int]:
        """Mark a transaction as complete (response received)."""
        if rob_idx not in self._active_transactions:
            return None

        txn = self._active_transactions[rob_idx]
        txn.state = TransactionState.COMPLETED
        txn.timestamp_end = timestamp

        # Calculate latency
        latency = timestamp - txn.timestamp_start
        if txn.is_write:
            self.stats.total_write_latency += latency
        else:
            self.stats.total_read_latency += latency

        axi_id = txn.axi_id
        del self._active_transactions[rob_idx]
        del self._rob_to_axi[rob_idx]

        return axi_id

    def mark_transaction_complete_by_axi_id(self, axi_id: int, timestamp: int) -> bool:
        """Mark a transaction as complete by AXI ID."""
        rob_idx = None
        for rid, aid in self._rob_to_axi.items():
            if aid == axi_id:
                rob_idx = rid
                break

        if rob_idx is None:
            return False

        self.mark_transaction_complete(rob_idx, timestamp)
        return True

    def has_pending_input(self) -> bool:
        """Check if there are pending requests in AXI input FIFOs."""
        return bool(self._aw_input_fifo or self._w_input_fifo or self._ar_input_fifo)

    def has_pending_output(self) -> bool:
        """Check if there are flits ready to be sent from output buffers.

        Note: This only checks output buffers, NOT input FIFOs.
        Call process_cycle() first to move requests from input FIFOs to output buffers.
        """
        if self._strategy.uses_per_channel_buffers:
            # AXI Mode: check all per-channel buffers
            for ch in self._strategy.request_channels:
                buffer = self._per_channel_buffers.get(ch)
                if buffer and not buffer.is_empty():
                    return True
            return False
        else:
            # General Mode: check single shared buffer
            return not self.output_buffer.is_empty()

    def has_pending_work(self) -> bool:
        """Check if there is any pending work (input FIFOs or output buffers)."""
        return self.has_pending_input() or self.has_pending_output()

    def process_cycle(self, timestamp: int = 0) -> None:
        """Process one cycle - process input FIFOs and send flits.

        Processing order:
        1. AW FIFO - Create and send AW flits
        2. W FIFO - Create and send W flits (rate-limited)
        3. AR FIFO - Create and send AR flits
        4. Pending W flits - Legacy backpressure queue (deprecated)
        """
        # Process AXI input FIFOs
        self._process_aw_fifo(timestamp)
        self._process_w_fifo(timestamp)
        self._process_ar_fifo(timestamp)

        # Legacy: Try to send pending W flits from backpressure queue
        # This is kept for backward compatibility but should not be used
        # with the new FIFO-based approach
        self._try_send_pending_w_flits(timestamp)

    @property
    def outstanding_count(self) -> int:
        """Number of outstanding transactions."""
        return len(self._active_transactions) + len(self._pending_writes)


class _SlaveNI_RspPath:
    """
    Slave NI Response Path (internal component).

    Handles NoC response flits and converts them to AXI
    response channels (B, R).

    This is the "Flit Unpacking" part of Slave NI per spec.md.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[NIConfig] = None,
        ni_id: int = 0
    ):
        self.coord = coord
        self.config = config or NIConfig()
        self.ni_id = ni_id

        # Packet disassembler (Unpack B/R in spec.md)
        self.packet_disassembler = PacketDisassembler()

        # Input buffer for response flits
        self.input_buffer = FlitBuffer(
            self.config.resp_buffer_depth,
            f"SlaveNI({coord})_rsp_in"
        )

        # B RoB / R RoB: Reorder buffer for response matching
        self._reorder_buffer: Dict[int, Deque[Packet]] = {}
        for i in range(1 << self.config.axi_id_width):
            self._reorder_buffer[i] = deque()

        # Output queues for AXI responses
        self._b_queue: Deque[AXI_B] = deque()
        self._r_queue: Deque[AXI_R] = deque()

        # Statistics
        self.stats = NIStats()

        # Input port connection
        self.input_port: Optional[RouterPort] = None

        # Callback for transaction completion notification
        self._on_transaction_complete: Optional[Callable[[int, int], Any]] = None

        # Flit latency callback for metrics collection
        # Called with (latency, axi_channel, payload_bytes) when each flit arrives
        self._flit_latency_callback: Optional[Callable[[int, "AxiChannel", int], None]] = None

    def set_flit_latency_callback(
        self, callback: Callable[[int, "AxiChannel", int], None]
    ) -> None:
        """
        Set callback for per-flit latency notification.

        Args:
            callback: Function that receives (latency, axi_channel, payload_bytes).
        """
        self._flit_latency_callback = callback

    def set_completion_callback(
        self,
        callback: Callable[[int, int], Any]
    ) -> None:
        """Set callback to be invoked when a transaction completes."""
        self._on_transaction_complete = callback

    def connect_input(self, port: RouterPort) -> None:
        """Connect input from router response port."""
        self.input_port = port

    def receive_flit(self, flit: Flit) -> bool:
        """Receive a response flit from NoC."""
        if self.input_buffer.is_full():
            return False

        self.input_buffer.push(flit)
        self.stats.resp_flits_received += 1
        return True

    def process_cycle(self, current_time: int = 0) -> None:
        """Process one cycle: reassemble packets and generate responses."""
        while not self.input_buffer.is_empty():
            flit = self.input_buffer.pop()
            if flit is None:
                break

            # BookSim2-style: Calculate per-flit latency
            if self._flit_latency_callback is not None:
                injection_cycle = getattr(flit, 'injection_cycle', None)
                if injection_cycle is not None:
                    latency = current_time - injection_cycle
                    axi_ch = flit.hdr.axi_ch
                    # Calculate actual payload bytes (R flit only)
                    # B is response-only, no user data
                    payload_bytes = 0
                    if axi_ch == AxiChannel.R and flit.payload is not None:
                        # R flit data length (typically 32 bytes)
                        data = getattr(flit.payload, 'data', b'')
                        payload_bytes = len(data) if data else 32
                    self._flit_latency_callback(latency, axi_ch, payload_bytes)

            # Try to reconstruct packet
            packet = self.packet_disassembler.receive_flit(flit)
            if packet is not None:
                self._process_completed_packet(packet, current_time)

    def _process_completed_packet(self, packet: Packet, current_time: int) -> None:
        """Process a fully reconstructed packet."""
        # Notify ReqPath that this transaction is complete
        if self._on_transaction_complete is not None:
            self._on_transaction_complete(packet.axi_id, current_time)

        # Add to reorder buffer (B RoB / R RoB matching)
        axi_id = packet.axi_id
        if axi_id in self._reorder_buffer:
            self._reorder_buffer[axi_id].append(packet)
        else:
            self._reorder_buffer[axi_id] = deque([packet])

        # Generate AXI responses
        self._generate_responses(axi_id, current_time)

    def _generate_responses(self, axi_id: int, current_time: int) -> None:
        """Generate AXI B/R responses from completed packets."""
        if axi_id not in self._reorder_buffer:
            return

        while self._reorder_buffer[axi_id]:
            packet = self._reorder_buffer[axi_id].popleft()

            if packet.packet_type == PacketType.WRITE_RESP:
                # Generate B response
                b_resp = AXI_B(
                    bid=axi_id,
                    bresp=AXIResp.OKAY,
                )
                self._b_queue.append(b_resp)
                self.stats.b_responses_sent += 1

            elif packet.packet_type == PacketType.READ_RESP:
                # Generate R response(s)
                payload = packet.payload
                beat_size = self.config.axi_data_width // 8

                if len(payload) == 0:
                    # Empty read response
                    r_resp = AXI_R(
                        rid=axi_id,
                        rdata=bytes(beat_size),
                        rresp=AXIResp.OKAY,
                        rlast=True,
                    )
                    self._r_queue.append(r_resp)
                    self.stats.r_responses_sent += 1
                else:
                    # Split into beats
                    num_beats = (len(payload) + beat_size - 1) // beat_size
                    for i in range(num_beats):
                        start = i * beat_size
                        end = min(start + beat_size, len(payload))
                        beat_data = payload[start:end]

                        # Pad if needed
                        if len(beat_data) < beat_size:
                            beat_data = beat_data + bytes(beat_size - len(beat_data))

                        r_resp = AXI_R(
                            rid=axi_id,
                            rdata=beat_data,
                            rresp=AXIResp.OKAY,
                            rlast=(i == num_beats - 1),
                        )
                        self._r_queue.append(r_resp)
                        self.stats.r_responses_sent += 1

    def get_b_response(self) -> Optional[AXI_B]:
        """Get next B response."""
        if self._b_queue:
            return self._b_queue.popleft()
        return None

    def get_r_response(self) -> Optional[AXI_R]:
        """Get next R response."""
        if self._r_queue:
            return self._r_queue.popleft()
        return None

    def has_pending_b(self) -> bool:
        """Check if B responses are pending."""
        return len(self._b_queue) > 0

    def has_pending_r(self) -> bool:
        """Check if R responses are pending."""
        return len(self._r_queue) > 0


# =============================================================================
# Slave NI (AXI Slave Interface)
# =============================================================================

class SlaveNI:
    """
    Slave Network Interface (AXI Slave side).

    Receives requests from local AXI Master (Host CPU/DMA or Node CPU/DMA),
    converts to NoC Request Flits, and returns AXI responses from Response Flits.

    Architecture per spec.md 2.2.2:
    - AXI Interface: AXI Slave (receives from local AXI Master)
    - Key Components:
      - AW/AR Spill Reg: Buffer AXI address channel
      - W Payload Store: Buffer Write Data
      - B RoB / R RoB: Reorder Buffer for tracking outstanding transactions
      - Coord Trans: Address translation (addr → dest_xy), generates rob_idx
      - Pack AW/AR/W: Assemble Request Flit
      - Unpack B/R: Disassemble Response Flit
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        address_map: SystemAddressMap,
        config: Optional[NIConfig] = None,
        ni_id: int = 0
    ):
        """
        Initialize Slave NI.

        Args:
            coord: NI coordinate (x, y).
            address_map: System address map for translation.
            config: NI configuration.
            ni_id: NI identifier.
        """
        self.coord = coord
        self.config = config or NIConfig()
        self.ni_id = ni_id

        # Create internal Request and Response paths
        self.req_path = _SlaveNI_ReqPath(coord, address_map, self.config, ni_id)
        self.rsp_path = _SlaveNI_RspPath(coord, self.config, ni_id)

        # Connect RspPath to ReqPath for transaction completion tracking
        self.rsp_path.set_completion_callback(
            self.req_path.mark_transaction_complete_by_axi_id
        )

    def set_flit_latency_callback(
        self, callback: Callable[[int, "AxiChannel", int], None]
    ) -> None:
        """
        Set callback for per-flit latency notification on response path.

        This callback is invoked when each response flit (B/R) arrives.
        Used for tracking Read throughput (R flits carry data).

        Args:
            callback: Function that receives (latency, axi_channel, payload_bytes).
        """
        self.rsp_path.set_flit_latency_callback(callback)

    def connect_to_router(
        self,
        req_port: RouterPort,
        resp_port: RouterPort
    ) -> None:
        """
        Connect NI to router's local ports.

        Args:
            req_port: Router's request local port.
            resp_port: Router's response local port.
        """
        self.req_path.connect_output(req_port)
        self.rsp_path.connect_input(resp_port)

    # === AXI Slave Request Interface ===
    def process_aw(self, aw: AXI_AW, timestamp: int = 0) -> bool:
        """Process AXI Write Address (from local AXI Master)."""
        return self.req_path.process_aw(aw, timestamp)

    def process_w(self, w: AXI_W, axi_id: int, timestamp: int = 0) -> bool:
        """Process AXI Write Data (from local AXI Master)."""
        return self.req_path.process_w(w, axi_id, timestamp)

    def process_ar(self, ar: AXI_AR, timestamp: int = 0) -> bool:
        """Process AXI Read Address (from local AXI Master)."""
        return self.req_path.process_ar(ar, timestamp)

    # === NoC Interface ===
    def get_req_flit(self, current_cycle: int = 0) -> Optional[Flit]:
        """
        Get request flit to send to NoC (Request Router).

        Args:
            current_cycle: Current simulation cycle (for injection tracking).

        Returns:
            Flit if available, None otherwise.
        """
        return self.req_path.get_output_flit(current_cycle)

    def get_channel_flit(
        self,
        channel: AxiChannel,
        current_cycle: int = 0
    ) -> Optional[Flit]:
        """
        Get request flit for specific AXI channel (AXI Mode).

        Args:
            channel: AXI channel (AW, W, or AR).
            current_cycle: Current simulation cycle.

        Returns:
            Flit if available, None otherwise.
        """
        return self.req_path.get_channel_flit(channel, current_cycle)

    def receive_resp_flit(self, flit: Flit) -> bool:
        """Receive response flit from NoC (Response Router)."""
        return self.rsp_path.receive_flit(flit)

    # === AXI Slave Response Interface ===
    def get_b_response(self) -> Optional[AXI_B]:
        """Get B response (to local AXI Master)."""
        return self.rsp_path.get_b_response()

    def get_r_response(self) -> Optional[AXI_R]:
        """Get R response (to local AXI Master)."""
        return self.rsp_path.get_r_response()

    def process_cycle(self, current_time: int = 0) -> None:
        """Process one simulation cycle."""
        self.req_path.process_cycle(current_time)  # Try to send ready packets
        self.rsp_path.process_cycle(current_time)

    def mark_transaction_complete(self, rob_idx: int, timestamp: int) -> Optional[int]:
        """Mark transaction as complete."""
        return self.req_path.mark_transaction_complete(rob_idx, timestamp)

    @property
    def stats(self) -> Tuple[NIStats, NIStats]:
        """Get statistics for both Request and Response paths."""
        return (self.req_path.stats, self.rsp_path.stats)

    @property
    def req_ni(self):
        """Backward compatibility: access request path."""
        return self.req_path

    @property
    def resp_ni(self):
        """Backward compatibility: access response path."""
        return self.rsp_path

    # === AXI Channel Ready Signals ===

    @property
    def aw_ready(self) -> bool:
        """Check if AW channel can accept new request."""
        if len(self.req_path._active_transactions) >= self.config.max_outstanding:
            return False
        # Check channel-specific buffer space
        return self.req_path._has_buffer_space(AxiChannel.AW)

    @property
    def w_ready(self) -> bool:
        """Check if W channel can accept data beat."""
        # W is ready if there's a pending write transaction AND buffer has space
        has_pending = len(self.req_path._pending_writes) > 0
        has_space = self.req_path._has_buffer_space(AxiChannel.W)
        return has_pending and has_space

    @property
    def ar_ready(self) -> bool:
        """Check if AR channel can accept new request."""
        if len(self.req_path._active_transactions) >= self.config.max_outstanding:
            return False
        # Check channel-specific buffer space
        return self.req_path._has_buffer_space(AxiChannel.AR)

    @property
    def has_b_response(self) -> bool:
        """Check if B response is available."""
        return self.rsp_path.has_pending_b()

    @property
    def has_r_response(self) -> bool:
        """Check if R response is available."""
        return self.rsp_path.has_pending_r()

    def __repr__(self) -> str:
        return (
            f"SlaveNI{self.coord}("
            f"req_out={self.req_path.output_buffer.occupancy}, "
            f"rsp_in={self.rsp_path.input_buffer.occupancy})"
        )


# =============================================================================
# Master NI Data Structures
# =============================================================================

@dataclass
class MasterNI_RequestInfo:
    """
    Request info stored in Per-ID FIFO for response routing.

    When Master NI receives a request from NoC, it stores this info
    to correctly route the response back to the source.
    """
    rob_idx: int                        # RoB index for response matching
    axi_id: int                         # AXI transaction ID
    src_coord: Tuple[int, int]          # Source NI coordinate (for response routing)
    is_write: bool                      # True for write, False for read
    timestamp: int                      # Request arrival time
    local_addr: int                     # Local memory address


# =============================================================================
# AXI Slave Memory Interface
# =============================================================================

class AXISlave:
    """
    AXI Slave interface wrapper.

    Receives AXI requests from Master NI, forwards to Memory model,
    and returns AXI responses.

    This wraps an external Memory instance with AXI protocol handling.
    The Memory is passed in, not owned by this class.
    """

    def __init__(
        self,
        memory,  # Memory or LocalMemory instance
        config: Optional[NIConfig] = None,
    ):
        """
        Initialize AXI Slave Memory wrapper.

        Args:
            memory: Memory instance to wrap.
            config: NI configuration for AXI parameters.
        """
        from src.testbench.memory import Memory, LocalMemory
        self.memory = memory
        self.config = config or NIConfig()

        # AXI Slave interface queues (input)
        self._aw_queue: Deque[AXI_AW] = deque()
        self._w_queue: Deque[Tuple[AXI_W, int]] = deque()  # (w_beat, axi_id)
        self._ar_queue: Deque[AXI_AR] = deque()

        # Pending write transactions (waiting for W data)
        self._pending_writes: Dict[int, Tuple[AXI_AW, List[AXI_W]]] = {}

        # Response queues (output)
        self._b_queue: Deque[AXI_B] = deque()
        self._r_queue: Deque[AXI_R] = deque()

        # Backpressure configuration
        self._max_r_queue_depth = self.config.r_queue_depth

        # Pending AR requests (deferred due to backpressure)
        self._pending_ar: Deque[AXI_AR] = deque()

        # Statistics
        self.stats = NIStats()

    # === AXI Slave Request Interface ===

    def accept_aw(self, aw: AXI_AW) -> bool:
        """
        Accept AXI Write Address.

        Args:
            aw: Write address request.

        Returns:
            True if accepted.
        """
        self._aw_queue.append(aw)
        self._pending_writes[aw.awid] = (aw, [])
        self.stats.aw_received += 1
        return True

    def accept_w(self, w: AXI_W, axi_id: int) -> bool:
        """
        Accept AXI Write Data beat.

        Args:
            w: Write data beat.
            axi_id: Associated AXI ID.

        Returns:
            True if accepted.
        """
        if axi_id not in self._pending_writes:
            return False

        aw, w_beats = self._pending_writes[axi_id]
        w_beats.append(w)
        self.stats.w_received += 1

        # Check if write is complete
        if w.wlast:
            self._process_write(axi_id, aw, w_beats)

        return True

    def _process_write(self, axi_id: int, aw: AXI_AW, w_beats: List[AXI_W]) -> None:
        """Process completed write transaction."""
        # Concatenate write data
        data = b"".join(w.wdata for w in w_beats)

        # Extract local address (lower 32 bits)
        local_addr = aw.awaddr & 0xFFFFFFFF

        # Write to memory
        self.memory.write(local_addr, data)

        # Generate B response
        b_resp = AXI_B(
            bid=axi_id,
            bresp=AXIResp.OKAY,
        )
        self._b_queue.append(b_resp)
        self.stats.write_requests += 1
        self.stats.b_responses_sent += 1

        # Clean up
        del self._pending_writes[axi_id]

    def accept_ar(self, ar: AXI_AR) -> bool:
        """
        Accept AXI Read Address.

        With backpressure: if _r_queue is full, the AR is queued
        in _pending_ar and will be processed when space is available.

        Args:
            ar: Read address request.

        Returns:
            True if accepted (always accepts, may defer processing).
        """
        self._ar_queue.append(ar)
        self.stats.ar_received += 1

        # Check backpressure: if R queue has space, process immediately
        if len(self._r_queue) < self._max_r_queue_depth:
            self._process_read(ar)
        else:
            # Backpressure: defer processing until space is available
            self._pending_ar.append(ar)

        return True

    def _process_read(self, ar: AXI_AR) -> None:
        """Process read transaction."""
        # Extract local address
        local_addr = ar.araddr & 0xFFFFFFFF

        # Read from memory
        read_size = ar.transfer_size
        data, _ = self.memory.read(local_addr, read_size)

        # Generate R response(s)
        beat_size = self.config.axi_data_width // 8
        num_beats = (len(data) + beat_size - 1) // beat_size if data else 1

        if len(data) == 0:
            # Empty read
            r_resp = AXI_R(
                rid=ar.arid,
                rdata=bytes(beat_size),
                rresp=AXIResp.OKAY,
                rlast=True,
            )
            self._r_queue.append(r_resp)
        else:
            for i in range(num_beats):
                start = i * beat_size
                end = min(start + beat_size, len(data))
                beat_data = data[start:end]

                # Pad if needed
                if len(beat_data) < beat_size:
                    beat_data = beat_data + bytes(beat_size - len(beat_data))

                r_resp = AXI_R(
                    rid=ar.arid,
                    rdata=beat_data,
                    rresp=AXIResp.OKAY,
                    rlast=(i == num_beats - 1),
                )
                self._r_queue.append(r_resp)

        self.stats.read_requests += 1
        self.stats.r_responses_sent += 1

    # === AXI Slave Response Interface ===

    def get_b_response(self) -> Optional[AXI_B]:
        """Get next B response."""
        if self._b_queue:
            return self._b_queue.popleft()
        return None

    def get_r_response(self) -> Optional[AXI_R]:
        """Get next R response."""
        if self._r_queue:
            return self._r_queue.popleft()
        return None

    def has_pending_b(self) -> bool:
        """Check if B responses are pending."""
        return len(self._b_queue) > 0

    def has_pending_r(self) -> bool:
        """Check if R responses are pending."""
        return len(self._r_queue) > 0

    def is_r_queue_full(self) -> bool:
        """Check if R queue is at capacity (backpressured)."""
        return len(self._r_queue) >= self._max_r_queue_depth

    @property
    def r_queue_occupancy(self) -> int:
        """Current R queue occupancy."""
        return len(self._r_queue)

    @property
    def pending_ar_count(self) -> int:
        """Number of AR requests waiting due to backpressure."""
        return len(self._pending_ar)

    def process_cycle(self, current_time: int = 0) -> None:
        """
        Process one cycle.

        Handles deferred AR requests when R queue has space (backpressure release).
        """
        # Process pending AR requests if R queue has space
        while self._pending_ar and len(self._r_queue) < self._max_r_queue_depth:
            ar = self._pending_ar.popleft()
            self._process_read(ar)

    # === Direct Memory Access (for initialization/debug) ===

    def write_local(self, addr: int, data: bytes) -> None:
        """Write to memory directly (bypass AXI)."""
        self.memory.write(addr, data)

    def read_local(self, addr: int, size: int = 8) -> bytes:
        """Read from memory directly (bypass AXI)."""
        data, _ = self.memory.read(addr, size)
        return data

    def verify_local(self, addr: int, expected: bytes) -> bool:
        """Verify memory contents."""
        return self.memory.verify(addr, expected)

# =============================================================================
# Local Memory Unit (AXI Slave + LocalMemory Bundle)
# =============================================================================

class LocalMemoryUnit:
    """
    Local Memory Unit bundling AXI Slave interface with LocalMemory.
    
    This represents the memory subsystem at each compute node:
    - LocalMemory: Actual storage (sparse, 4GB address space)
    - AXI Slave: AXI protocol interface to the memory
    
    Architecture:
        MasterNI --> AXI Slave --> LocalMemory
        (separate)   (bundled together in LocalMemoryUnit)
    """
    
    def __init__(
        self,
        node_id: int = 0,
        memory_size: int = 0x100000000,
        config: Optional[NIConfig] = None,
    ):
        """
        Initialize Local Memory Unit.
        
        Args:
            node_id: Node identifier.
            memory_size: Memory size in bytes (default 4GB).
            config: NI configuration for AXI parameters.
        """
        from src.testbench.memory import LocalMemory
        
        self.node_id = node_id
        self.config = config or NIConfig()
        
        # Create LocalMemory
        self.memory = LocalMemory(
            node_id=node_id,
            size=memory_size,
        )
        
        # Create AXI Slave wrapping the memory
        self.axi_slave = AXISlave(
            memory=self.memory,
            config=self.config,
        )
    
    # === Convenience Methods (delegate to memory) ===
    
    def write(self, addr: int, data: bytes) -> None:
        """Write to memory directly (bypass AXI)."""
        self.memory.write(addr, data)
    
    def read(self, addr: int, size: int) -> bytes:
        """Read from memory directly (bypass AXI)."""
        data, _ = self.memory.read(addr, size)
        return data
    
    def get_contents(self, addr: int, size: int) -> bytes:
        """Get memory contents without updating stats."""
        return self.memory.get_contents(addr, size)
    
    def verify(self, addr: int, expected: bytes) -> bool:
        """Verify memory contents."""
        return self.memory.verify(addr, expected)
    
    def clear(self) -> None:
        """Clear memory contents."""
        self.memory.clear()


# =============================================================================
# Master NI (AXI Master Interface)
# =============================================================================

class MasterNI:
    """
    Master Network Interface (AXI Master side).

    Receives NoC Request flits, converts to AXI transactions,
    sends to external AXI Slave, and returns Response flits.

    Architecture per spec.md 2.2.2:
    - AXI Interface: AXI Master (sends requests to external AXI Slave)
    - Key Components:
      - Per-ID FIFO: Store incoming request info by AXI ID
      - Store Req Info: Save request info for response routing
      - Routing Logic: Extract rob_idx, dest_id from header for response routing
      - Unpack AW/AR/W: Disassemble Request Flit
      - Pack B/R: Assemble Response Flit
    
    Note: MasterNI does NOT own memory. It connects to an external
    AXI Slave (provided via dependency injection).
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[NIConfig] = None,
        ni_id: int = 0,
        node_id: int = 0,
        axi_slave: Optional[AXISlave] = None,
        memory_size: int = 0x100000000,  # Deprecated, for backward compatibility
    ):
        """
        Initialize Master NI.

        Args:
            coord: NI coordinate (x, y).
            config: NI configuration.
            ni_id: NI identifier.
            node_id: Node ID for this NI.
            axi_slave: External AXI Slave to connect to.
                       If None, creates internal LocalMemoryUnit (backward compatible).
            memory_size: Deprecated. Use axi_slave parameter instead.
        """
        self.coord = coord
        self.config = config or NIConfig()
        self.ni_id = ni_id
        self.node_id = node_id

        # === NoC Side ===
        # Packet assembler/disassembler
        self.packet_assembler = PacketAssembler(self.config.flit_payload_size)
        self.packet_disassembler = PacketDisassembler()

        # Input buffer for request flits (from Request Router)
        self.req_input = FlitBuffer(
            self.config.req_buffer_depth,
            f"MasterNI({coord})_req_in"
        )

        # Channel mode strategy for polymorphic buffer management
        self._strategy: ChannelModeStrategy = get_channel_mode_strategy(
            self.config.channel_mode
        )

        # Output buffer for response flits (to Response Router)
        # General Mode: Single buffer for all response channels
        self.resp_output = FlitBuffer(
            self.config.resp_buffer_depth,
            f"MasterNI({coord})_rsp_out"
        )

        # Per-channel output buffers for response (AXI Mode only)
        # Uses strategy to determine which channels need separate buffers
        self._resp_per_channel: Dict[AxiChannel, FlitBuffer] = {}
        for ch in self._strategy.get_buffer_channels_for_response():
            self._resp_per_channel[ch] = FlitBuffer(
                self.config.resp_buffer_depth,
                f"MasterNI({coord})_{ch.name}_out"
            )

        # === Per-ID FIFO (spec.md key component) ===
        # Store incoming request info by AXI ID for response routing
        self._per_id_fifo: Dict[int, Deque[MasterNI_RequestInfo]] = {}
        for i in range(1 << self.config.axi_id_width):
            self._per_id_fifo[i] = deque()

        # === AXI Master Side ===
        # Connect to external AXI Slave or create internal one (backward compat)
        if axi_slave is not None:
            self.axi_slave = axi_slave
            self.local_memory = axi_slave.memory  # Reference for convenience
            self._owns_memory = False
        else:
            # Backward compatibility: create internal LocalMemoryUnit
            from src.testbench.memory import LocalMemory
            self._local_memory_unit = LocalMemoryUnit(
                node_id=node_id,
                memory_size=memory_size,
                config=self.config,
            )
            self.axi_slave = self._local_memory_unit.axi_slave
            self.local_memory = self._local_memory_unit.memory
            self._owns_memory = True

        # Statistics
        self.stats = NIStats()

        # R beat sequence number tracking for streaming R flits
        # Keyed by AXI ID -> current seq_num
        self._r_seq_num: Dict[int, int] = {}

        # Pending R flits queue for backpressure handling
        self._pending_r_flits: Deque[Flit] = deque()

        # Accumulate R beats per transaction before creating Packet
        # Key: axi_id, Value: (req_info, accumulated_data)
        self._pending_r_data: Dict[int, Tuple[MasterNI_RequestInfo, bytearray]] = {}

        # Pending AW info for matching with W packets (FlooNoC: AW and W are separate packets)
        # Key: (dst_id, rob_idx), Value: AW packet info dict
        self._pending_aw_info: Dict[Tuple[int, int], Dict] = {}

        # Pending W packets for matching with AW (W may arrive before AW in pipelined mode)
        # Key: (dst_id, rob_idx), Value: (Packet, Flit, timestamp)
        self._pending_w_packets: Dict[Tuple[int, int], Tuple] = {}

        # Flit latency callback for metrics collection (BookSim2-style)
        # Called with (latency, axi_channel, payload_bytes) when each flit arrives
        # - latency: cycles from injection to arrival
        # - axi_channel: AxiChannel enum (AW, W, AR, B, R)
        # - payload_bytes: actual data bytes (W/R have data, AW/AR/B are 0)
        self._flit_latency_callback: Optional[Callable[[int, "AxiChannel", int], None]] = None

        # === Valid/Ready Interface Signals ===
        # Request input (Router LOCAL → NI)
        self.req_in_valid: bool = False
        self.req_in_flit: Optional[Flit] = None
        self.req_in_ready: bool = True  # Can accept if buffer not full

        # Response output (NI → Router LOCAL)
        self.resp_out_valid: bool = False
        self.resp_out_flit: Optional[Flit] = None
        self.resp_out_ready: bool = False  # Set by downstream (router)

        # Virtual ports for mesh connection (set by Mesh._connect_nis)
        self._router_req_port: Optional[RouterPort] = None
        self._router_resp_port: Optional[RouterPort] = None

    # === Per-Channel Buffer Helpers ===

    def _get_resp_buffer(self, channel: AxiChannel) -> FlitBuffer:
        """Get the appropriate response output buffer for a channel."""
        if self._strategy.uses_per_channel_buffers and channel in self._resp_per_channel:
            return self._resp_per_channel[channel]
        return self.resp_output

    def _push_resp_flit(self, flit: Flit) -> bool:
        """Push response flit to appropriate buffer based on channel mode."""
        target = self._get_resp_buffer(flit.hdr.axi_ch)
        return target.push(flit)

    def _has_resp_buffer_space(self, channel: AxiChannel) -> bool:
        """Check if response buffer has space for the given channel."""
        return self._get_resp_buffer(channel).free_space > 0

    # === Valid/Ready Interface Methods ===

    def update_ready_signals(self) -> None:
        """
        Update ready signals based on buffer state.

        Called at the beginning of each cycle.
        """
        self.req_in_ready = not self.req_input.is_full()

    def sample_req_input(self) -> bool:
        """
        Sample request input and perform handshake if valid && ready.

        Returns:
            True if a flit was received.
        """
        if self.req_in_valid and self.req_in_ready and self.req_in_flit is not None:
            success = self.req_input.push(self.req_in_flit)
            if success:
                return True
        return False

    def update_resp_output(self) -> None:
        """
        Update response output signals.

        Sets out_valid if there's a response flit to send.
        """
        if not self.resp_out_valid and not self.resp_output.is_empty():
            flit = self.resp_output.peek()
            if flit is not None:
                self.resp_out_valid = True
                self.resp_out_flit = flit

    def clear_resp_output_if_accepted(self) -> bool:
        """
        Clear response output if accepted by downstream.

        Returns:
            True if output was accepted.
        """
        if self.resp_out_valid and self.resp_out_ready:
            # Handshake successful - pop from buffer
            self.resp_output.pop()
            self.resp_out_valid = False
            self.resp_out_flit = None
            return True
        return False

    def clear_input_signals(self) -> None:
        """Clear input signals after sampling."""
        self.req_in_valid = False
        self.req_in_flit = None

    # === Packet Arrival Callback ===

    def set_flit_latency_callback(
        self, callback: Callable[[int, "AxiChannel", int], None]
    ) -> None:
        """
        Set callback for per-flit latency notification.

        This callback is invoked when each flit arrives at the MasterNI.
        The latency is calculated as: arrival_cycle - injection_cycle.
        Used for BookSim2-style per-flit latency tracking.

        Args:
            callback: Function that receives (latency, axi_channel, payload_bytes).
                - latency: cycles from injection to arrival
                - axi_channel: AxiChannel enum (AW, W, AR, B, R)
                - payload_bytes: actual data bytes (W/R have data, AW/AR/B are 0)
        """
        self._flit_latency_callback = callback

    # === NoC Interface (Legacy) ===

    def receive_req_flit(self, flit: Flit) -> bool:
        """
        Receive a request flit from NoC (Request Router).

        Args:
            flit: Request flit.

        Returns:
            True if accepted.
        """
        if self.req_input.is_full():
            return False
        self.req_input.push(flit)
        return True

    def get_resp_flit(self) -> Optional[Flit]:
        """
        Get next response flit to send to NoC (Response Router).

        Returns:
            Response flit if available.
        """
        if self._strategy.uses_per_channel_buffers:
            # AXI Mode: check response channel buffers in priority order
            for channel in self._strategy.response_channels:
                buffer = self._resp_per_channel.get(channel)
                if buffer:
                    flit = buffer.pop()
                    if flit is not None:
                        return flit
            return None
        return self.resp_output.pop()

    def get_channel_resp_flit(self, channel: AxiChannel) -> Optional[Flit]:
        """
        Get next response flit for specific AXI channel (AXI Mode).

        Args:
            channel: AXI channel (B or R).

        Returns:
            Response flit if available, None otherwise.
        """
        if not self._strategy.uses_per_channel_buffers:
            return self.get_resp_flit()

        if channel not in self._resp_per_channel:
            return None

        return self._resp_per_channel[channel].pop()

    def peek_channel_resp_flit(self, channel: AxiChannel) -> Optional[Flit]:
        """
        Peek at next response flit for specific channel without removing.

        Args:
            channel: AXI channel (B or R).

        Returns:
            Response flit if available, None otherwise.
        """
        if not self._strategy.uses_per_channel_buffers:
            return self.resp_output.peek()

        if channel not in self._resp_per_channel:
            return None

        return self._resp_per_channel[channel].peek()

    def has_pending_channel_response(self, channel: AxiChannel) -> bool:
        """
        Check if responses are pending for specific channel.

        Args:
            channel: AXI channel (B or R).

        Returns:
            True if pending responses exist.
        """
        if not self._strategy.uses_per_channel_buffers:
            return not self.resp_output.is_empty()

        if channel not in self._resp_per_channel:
            return False

        return not self._resp_per_channel[channel].is_empty()

    # === Processing ===

    def process_cycle(self, current_time: int = 0) -> None:
        """
        Process one simulation cycle.

        Steps:
        1. Receive flits, reconstruct packets (Unpack AW/AR/W)
        2. Extract AXI requests, push to Per-ID FIFO
        3. Forward AXI requests to Memory (AXI Master → AXI Slave)
        4. Collect AXI responses from Memory
        5. Match responses using Per-ID FIFO (Routing Logic)
        6. Pack responses into flits (Pack B/R)
        """
        # Process Memory (for latency modeling)
        self.axi_slave.process_cycle(current_time)

        # Process incoming flits
        self._process_incoming_flits(current_time)

        # Collect AXI responses and generate response flits
        self._collect_axi_responses(current_time)

    def _process_incoming_flits(self, current_time: int) -> None:
        """Unpack flits → packets → AXI requests."""
        while not self.req_input.is_empty():
            flit = self.req_input.pop()
            if flit is None:
                break

            # BookSim2-style: Calculate per-flit latency
            # Flit carries its injection_cycle (set by SlaveNI)
            if self._flit_latency_callback is not None:
                injection_cycle = getattr(flit, 'injection_cycle', None)
                if injection_cycle is not None:
                    latency = current_time - injection_cycle
                    axi_ch = flit.hdr.axi_ch
                    # Calculate actual payload bytes from strb (W flit only)
                    # AW/AR are address-only, no user data
                    payload_bytes = 0
                    if axi_ch == AxiChannel.W and flit.payload is not None:
                        # Count set bits in strb to get valid bytes
                        strb = getattr(flit.payload, 'strb', 0xFFFFFFFF)
                        payload_bytes = bin(strb).count('1')
                    self._flit_latency_callback(latency, axi_ch, payload_bytes)

            # Try to reconstruct packet
            packet = self.packet_disassembler.receive_flit(flit)
            if packet is not None:
                self._process_request_packet(packet, current_time)

    def _process_request_packet(self, packet: Packet, current_time: int) -> None:
        """
        Process a complete request packet.

        FlooNoC spec: AW and W are separate packets.
        - AW packet: single-flit, contains address/ID info, no data
        - W packet: multi-flit, contains data, needs matching AW info

        Converts packet to AXI request, stores info in Per-ID FIFO,
        and forwards to Memory.
        """
        # Note: Per-flit latency tracking is done in _process_incoming_flits()

        # Check if this is AW-only or W-only packet (FlooNoC separate packets)
        head_flit = packet.flits[0] if packet.flits else None

        if packet.packet_type == PacketType.WRITE_REQ and head_flit:
            if head_flit.hdr.axi_ch == AxiChannel.AW:
                # AW-only packet: store info and wait for W packet
                self._handle_aw_only_packet(packet, head_flit, current_time)
                return
            elif head_flit.hdr.axi_ch == AxiChannel.W:
                # W-only packet: find matching AW info and process
                self._handle_w_only_packet(packet, head_flit, current_time)
                return

        # Handle other packet types (READ_REQ, WRITE_RESP, READ_RESP)
        self._store_request_info(packet, current_time)

        if packet.packet_type == PacketType.READ_REQ:
            self._forward_read_request(packet, current_time)

    def _handle_aw_only_packet(
        self, packet: Packet, aw_flit: Flit, current_time: int
    ) -> None:
        """
        Handle AW-only packet (FlooNoC: AW is single-flit packet).

        Checks if W packet already arrived (pipelined mode), otherwise stores
        AW info for later matching.

        Note: AW/W matching uses (dst_id, rob_idx).
        - dst_id is not modified during routing
        - rob_idx is shared between AW and W from same transaction
        - src_id is modified by RoutingSelector, so cannot be used
        """
        # Extract AW info - match by (dst_id, rob_idx)
        aw_payload = aw_flit.payload
        key = (aw_flit.hdr.dst_id, aw_flit.hdr.rob_idx)

        aw_info = {
            'rob_idx': packet.rob_idx,
            'axi_id': aw_payload.axi_id,
            'src_coord': packet.src,
            'local_addr': aw_payload.addr,
            'burst_length': aw_payload.length + 1,  # awlen + 1 = num beats
            'timestamp': current_time,
        }

        # Check if W packet already arrived (pipelined: W before AW)
        if key in self._pending_w_packets:
            w_packet, w_flit, w_timestamp = self._pending_w_packets.pop(key)
            self._complete_write_transaction(w_packet, aw_info, current_time)
        else:
            # Store AW info and wait for W
            self._pending_aw_info[key] = aw_info

    def _handle_w_only_packet(
        self, packet: Packet, w_flit: Flit, current_time: int
    ) -> None:
        """
        Handle W-only packet (FlooNoC: W is separate data packet).

        Checks if AW packet already arrived, otherwise stores W packet
        for later matching (pipelined mode: W may arrive before AW).
        """
        key = (w_flit.hdr.dst_id, w_flit.hdr.rob_idx)

        if key in self._pending_aw_info:
            # AW already arrived, complete the transaction
            aw_info = self._pending_aw_info.pop(key)
            self._complete_write_transaction(packet, aw_info, current_time)
        else:
            # AW hasn't arrived yet, store W packet for later
            self._pending_w_packets[key] = (packet, w_flit, current_time)

    def _complete_write_transaction(
        self, w_packet: Packet, aw_info: Dict, current_time: int
    ) -> None:
        """
        Complete a write transaction with matched AW info and W packet.

        Stores request info in Per-ID FIFO and forwards to Memory.
        """
        # Store request info in Per-ID FIFO for response routing
        req_info = MasterNI_RequestInfo(
            rob_idx=aw_info['rob_idx'],
            axi_id=aw_info['axi_id'],
            src_coord=aw_info['src_coord'],
            is_write=True,
            timestamp=aw_info['timestamp'],
            local_addr=aw_info['local_addr'],
        )

        axi_id = aw_info['axi_id']
        if axi_id not in self._per_id_fifo:
            self._per_id_fifo[axi_id] = deque()
        self._per_id_fifo[axi_id].append(req_info)

        # Forward complete write request to Memory
        self._forward_write_request_with_aw_info(w_packet, aw_info, current_time)

    def _store_request_info(self, packet: Packet, current_time: int) -> None:
        """Store request info in Per-ID FIFO for response routing."""
        req_info = MasterNI_RequestInfo(
            rob_idx=packet.rob_idx,
            axi_id=packet.axi_id,
            src_coord=packet.src,
            is_write=(packet.packet_type == PacketType.WRITE_REQ),
            timestamp=current_time,
            local_addr=packet.local_addr,
        )

        axi_id = packet.axi_id
        if axi_id not in self._per_id_fifo:
            self._per_id_fifo[axi_id] = deque()
        self._per_id_fifo[axi_id].append(req_info)

    def _forward_write_request(self, packet: Packet, current_time: int) -> None:
        """Forward write request to Memory via AXI interface."""
        # Create AXI AW
        aw = AXI_AW(
            awid=packet.axi_id,
            awaddr=packet.local_addr,
            awlen=0,  # Single beat
            awsize=AXISize.SIZE_8,
        )
        self.axi_slave.accept_aw(aw)

        # Create AXI W
        w = AXI_W(
            wdata=packet.payload,
            wstrb=0xFF,
            wlast=True,
        )
        self.axi_slave.accept_w(w, packet.axi_id)

        self.stats.write_requests += 1

    def _forward_write_request_with_aw_info(
        self, packet: Packet, aw_info: Dict, current_time: int
    ) -> None:
        """
        Forward write request to Memory using separate AW info and W packet.

        FlooNoC spec: AW and W are separate packets. This method combines
        the AW info (address, ID) with W packet data to form complete write.
        """
        # Create AXI AW from stored info
        aw = AXI_AW(
            awid=aw_info['axi_id'],
            awaddr=aw_info['local_addr'],
            awlen=aw_info['burst_length'] - 1,  # awlen = beats - 1
            awsize=AXISize.SIZE_8,
        )
        self.axi_slave.accept_aw(aw)

        # Create AXI W from packet payload
        w = AXI_W(
            wdata=packet.payload,
            wstrb=0xFF,
            wlast=True,
        )
        self.axi_slave.accept_w(w, aw_info['axi_id'])

        self.stats.write_requests += 1

    def _forward_read_request(self, packet: Packet, current_time: int) -> None:
        """Forward read request to Memory via AXI interface."""
        # Get read size from packet
        read_size = getattr(packet, 'read_size', 8)
        if read_size <= 0:
            read_size = 8

        # Calculate burst length from read_size
        beat_size = DEFAULT_BEAT_SIZE_BYTES
        num_beats = (read_size + beat_size - 1) // beat_size
        arlen = num_beats - 1  # AXI arlen is num_beats - 1

        # Create AXI AR with correct burst length
        ar = AXI_AR(
            arid=packet.axi_id,
            araddr=packet.local_addr,
            arlen=arlen,
            arsize=AXISize.SIZE_8,
        )
        self.axi_slave.accept_ar(ar)

        self.stats.read_requests += 1

    def _collect_axi_responses(self, current_time: int) -> None:
        """
        Collect AXI responses from Memory and pack into response flits.

        Uses Per-ID FIFO to match responses with original requests
        and extract routing info (Routing Logic).

        IMPORTANT: Cycle-accurate behavior:
        - AXI Mode: Each channel (B, R) can send at most 1 flit per cycle
        - General Mode: All channels share output, at most 1 flit per cycle total

        This ensures flit injection rate matches physical constraints.
        """
        # Track flits sent this cycle for cycle-accurate limiting
        b_flit_sent = False
        r_flit_sent = False

        # In General Mode, all responses share one output - only one flit total
        general_mode = not self._strategy.uses_per_channel_buffers

        # Collect B response (write complete) - at most 1 per cycle
        if self.axi_slave.has_pending_b():
            b_buffer = self._get_resp_buffer(AxiChannel.B)
            if b_buffer.free_space >= 1:
                b_resp = self.axi_slave.get_b_response()
                if b_resp is not None:
                    axi_id = b_resp.bid
                    if axi_id in self._per_id_fifo and self._per_id_fifo[axi_id]:
                        req_info = self._per_id_fifo[axi_id].popleft()

                        # Create response packet with routing info
                        resp_packet = PacketFactory.create_write_response_from_info(
                            src=self.coord,
                            dest=req_info.src_coord,
                            axi_id=axi_id,
                        )

                        # Pack into flits (B is single flit)
                        flits = self.packet_assembler.assemble(resp_packet)
                        for flit in flits:
                            flit.injection_cycle = current_time  # Set for latency tracking
                            if not self._push_resp_flit(flit):
                                self.stats.resp_flits_dropped += 1

                        self.stats.b_responses_sent += 1
                        b_flit_sent = True

        # In General Mode, skip R if we already sent a B flit
        if general_mode and b_flit_sent:
            return

        # Collect R response (read data) - at most 1 flit per cycle
        # First try to send pending R flits from backpressure queue
        if not r_flit_sent:
            r_flit_sent = self._try_send_one_pending_r_flit()

        # If no pending flit was sent, try to collect new R beat
        # Note: We collect ALL available R beats to accumulate data, but only send 1 flit
        while self.axi_slave.has_pending_r():
            r_resp = self.axi_slave.get_r_response()
            if r_resp is None:
                break

            axi_id = r_resp.rid

            # Get routing info from Per-ID FIFO (peek, don't pop yet)
            if axi_id not in self._per_id_fifo or not self._per_id_fifo[axi_id]:
                continue

            req_info = self._per_id_fifo[axi_id][0]  # Peek

            # Initialize accumulator for this transaction if needed
            if axi_id not in self._pending_r_data:
                self._pending_r_data[axi_id] = (req_info, bytearray())

            # Accumulate R beat data
            _, data_buffer = self._pending_r_data[axi_id]
            data_buffer.extend(r_resp.rdata)

            # On rlast: create READ_RESP Packet and assemble into flits
            if r_resp.rlast:
                req_info, accumulated_data = self._pending_r_data.pop(axi_id)

                # Create READ_RESP Packet with all accumulated data
                # Use original rob_idx for proper packet matching at destination
                resp_packet = PacketFactory.create_read_response_from_info(
                    src=self.coord,
                    dest=req_info.src_coord,
                    axi_id=axi_id,
                    data=bytes(accumulated_data),
                    rob_idx=req_info.rob_idx,
                )

                # Assemble into R flits using PacketAssembler
                r_flits = self.packet_assembler.assemble(resp_packet)
                for flit in r_flits:
                    flit.injection_cycle = current_time  # Set for latency tracking
                    self._pending_r_flits.append(flit)

                # Complete transaction
                self._per_id_fifo[axi_id].popleft()
                self.stats.r_responses_sent += 1

        # Try to send one pending R flit (if we haven't already sent one this cycle)
        if not r_flit_sent:
            r_flit_sent = self._try_send_one_pending_r_flit()

    def _try_send_one_pending_r_flit(self) -> bool:
        """
        Try to send one pending R flit from backpressure queue.

        Returns:
            True if a flit was sent, False otherwise.
        """
        if not self._pending_r_flits:
            return False

        if not self._has_resp_buffer_space(AxiChannel.R):
            return False

        flit = self._pending_r_flits.popleft()
        if self._push_resp_flit(flit):
            return True
        else:
            # Put back if failed
            self._pending_r_flits.appendleft(flit)
            return False

    def _process_deferred_responses(self, current_time: int) -> None:
        """
        Legacy method - no longer needed with streaming R flit mode.

        R flits are now sent immediately as each beat arrives,
        queued in _pending_r_flits if backpressure occurs.
        """
        pass  # No-op: streaming mode handles R flits differently

    # === Direct Memory Access (for initialization/debug) ===

    def write_local(self, addr: int, data: bytes) -> None:
        """Write to local memory directly (for initialization)."""
        self.axi_slave.write_local(addr, data)

    def read_local(self, addr: int, size: int = 8) -> bytes:
        """Read from local memory directly."""
        return self.axi_slave.read_local(addr, size)

    def verify_local(self, addr: int, expected: bytes) -> bool:
        """Verify local memory contents match expected data."""
        return self.axi_slave.verify_local(addr, expected)

    def has_pending_response(self) -> bool:
        """Check if responses are pending."""
        if self._strategy.uses_per_channel_buffers:
            # AXI Mode: check all response channel buffers
            for channel in self._strategy.response_channels:
                buffer = self._resp_per_channel.get(channel)
                if buffer and not buffer.is_empty():
                    return True
            return False
        return not self.resp_output.is_empty()

    def peek_resp_flit(self) -> Optional[Flit]:
        """
        Peek at next response flit without removing it.

        Used by AXI Mode to determine which channel router to use.

        Returns:
            Next response flit if available, None otherwise.
        """
        if self._strategy.uses_per_channel_buffers:
            # AXI Mode: check response channel buffers in priority order
            for channel in self._strategy.response_channels:
                buffer = self._resp_per_channel.get(channel)
                if buffer:
                    flit = buffer.peek()
                    if flit is not None:
                        return flit
            return None
        return self.resp_output.peek()

    def __repr__(self) -> str:
        return (
            f"MasterNI{self.coord}("
            f"req_in={self.req_input.occupancy}, "
            f"rsp_out={self.resp_output.occupancy})"
        )


# =============================================================================
# Backward Compatibility Aliases (Deprecated)
# =============================================================================

import warnings as _warnings


def _create_deprecated_alias(name: str, target, new_name: str):
    """
    Create a deprecated class alias that warns on first use.

    Uses a wrapper class to emit deprecation warning when instantiated.
    """
    class DeprecatedAlias(target):
        _warned = False

        def __new__(cls, *args, **kwargs):
            if not DeprecatedAlias._warned:
                _warnings.warn(
                    f"{name} is deprecated, use {new_name} instead",
                    DeprecationWarning,
                    stacklevel=2
                )
                DeprecatedAlias._warned = True
            return super().__new__(cls)

    DeprecatedAlias.__name__ = name
    DeprecatedAlias.__qualname__ = name
    return DeprecatedAlias


# For backward compatibility with existing code (deprecated)
NetworkInterface = _create_deprecated_alias("NetworkInterface", SlaveNI, "SlaveNI")

# Legacy internal class names (deprecated, use SlaveNI/MasterNI instead)
ReqNI = _create_deprecated_alias("ReqNI", _SlaveNI_ReqPath, "_SlaveNI_ReqPath")
RespNI = _create_deprecated_alias("RespNI", _SlaveNI_RspPath, "_SlaveNI_RspPath")
