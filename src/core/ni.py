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
from typing import Optional, Dict, List, Tuple, Deque, Callable
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
from .router import RouterPort, Direction

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
    req_buffer_depth: int = 8       # Request output buffer depth
    resp_buffer_depth: int = 8      # Response input buffer depth
    r_queue_depth: int = 64         # Max R beats in AXISlave response queue (backpressure)

    # Flit parameters
    # Must be >= PACKET_HEADER_SIZE (12 bytes) + reasonable data
    flit_payload_size: int = 32     # Flit payload size in bytes

    # NoC-to-NoC routing mode
    # If True, destination is from AXI user signal: awuser[7:0]=x, awuser[15:8]=y
    # If False (default), destination is from address encoding: addr[39:32]=node_id
    use_user_signal_routing: bool = False


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

        # AW/AR Spill Reg + W Payload Store (spec.md)
        self._pending_writes: Dict[int, PendingTransaction] = {}

        # Ready-to-send queue: writes with all W beats, waiting for buffer space
        self._ready_to_send: Deque[PendingTransaction] = deque()

        # Output buffer for request flits
        self.output_buffer = FlitBuffer(
            self.config.req_buffer_depth,
            f"SlaveNI({coord})_req_out"
        )

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

    def _allocate_rob_idx(self) -> int:
        """Allocate a ROB index for tracking."""
        idx = self._next_rob_idx
        self._next_rob_idx = (self._next_rob_idx + 1) % self.config.reorder_buffer_size
        return idx

    def process_aw(self, aw: AXI_AW, timestamp: int = 0) -> bool:
        """
        Process AXI Write Address channel.

        Args:
            aw: Write address request.
            timestamp: Current simulation time.

        Returns:
            True if accepted.
        """
        if len(self._active_transactions) >= self.config.max_outstanding:
            return False

        # Also check if output buffer has space for at least one packet
        # This provides backpressure when the NI is congested
        if self.output_buffer.free_space < 1:
            return False

        self.stats.aw_received += 1

        # Extract destination based on routing mode
        if self.config.use_user_signal_routing:
            # NoC-to-NoC mode: destination from AXI user signal
            # Format: awuser[7:0] = dest_x, awuser[15:8] = dest_y
            awuser = getattr(aw, 'awuser', 0) or 0
            dest_x = awuser & 0xFF
            dest_y = (awuser >> 8) & 0xFF
            dest_coord = (dest_x, dest_y)
            local_addr = aw.awaddr & 0xFFFFFFFF  # 32-bit local address
        else:
            # Host-to-NoC mode: destination from address encoding
            dest_coord, local_addr = self.addr_translator.translate(aw.awaddr)

        # Create pending transaction (AW Spill Reg)
        txn = PendingTransaction(
            axi_id=aw.awid,
            rob_idx=0,  # Will be set when packet is created
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
        return True

    def process_w(self, w: AXI_W, axi_id: int, timestamp: int = 0) -> bool:
        """
        Process AXI Write Data channel.

        Args:
            w: Write data beat.
            axi_id: Associated AXI ID.
            timestamp: Current simulation time.

        Returns:
            True if accepted.
        """
        if axi_id not in self._pending_writes:
            return False

        txn = self._pending_writes[axi_id]
        txn.w_beats.append(w)  # W Payload Store
        txn.w_beats_received += 1
        self.stats.w_received += 1

        # Check if all W beats received
        if w.wlast or txn.w_beats_received >= txn.w_beats_expected:
            # Move from pending_writes to ready_to_send
            del self._pending_writes[axi_id]
            self._ready_to_send.append(txn)
            # Try to create and queue packet (Pack AW + Pack W)
            self._try_send_ready_packets(timestamp)

        return True

    def _try_send_ready_packets(self, timestamp: int) -> None:
        """
        Try to packetize and send ready-to-send transactions.
        
        Limits to 1 flit per cycle for cycle-accurate timing.
        """
        if not self._ready_to_send:
            return
            
        txn = self._ready_to_send[0]  # Peek at front
        all_sent = self._create_write_packet(txn, timestamp)
        if all_sent:
            self._ready_to_send.popleft()  # All flits sent

    def _create_write_packet(self, txn: PendingTransaction, timestamp: int) -> bool:
        """
        Create write request packet from completed W beats.

        For Wormhole routing, flits are sent as buffer space becomes available.
        Returns True when ALL flits have been queued.

        Returns:
            True if all flits queued, False if still have flits to send.
        """
        # Check if packet was already created (retry case)
        if hasattr(txn, '_cached_flits') and txn._cached_flits:
            flits = txn._cached_flits
        else:
            # Concatenate all write data
            data = b"".join(w.wdata for w in txn.w_beats)

            # Create packet
            packet = PacketFactory.create_write_request(
                src=txn.src_coord,
                dest=txn.dest_coord,
                local_addr=txn.local_addr,
                data=data,
                axi_id=txn.axi_id,
            )

            txn.rob_idx = packet.rob_idx

            # Assemble into flits
            flits = self.packet_assembler.assemble(packet)

            # Cache flits for streaming send
            txn._cached_flits = list(flits)  # Make a mutable copy

            # Track transaction as active when first flit is ready
            self._active_transactions[txn.rob_idx] = txn
            self._rob_to_axi[txn.rob_idx] = txn.axi_id
            self.stats.write_requests += 1

        # Wormhole: send only 1 flit per cycle for cycle-accurate timing
        if txn._cached_flits and self.output_buffer.free_space > 0:
            flit = txn._cached_flits.pop(0)
            self.output_buffer.push(flit)

        # Check if all flits sent
        if not txn._cached_flits:
            txn.state = TransactionState.PENDING_SEND
            txn._cached_flits = None
            return True

        # Still have flits to send
        return False

    def process_ar(self, ar: AXI_AR, timestamp: int = 0) -> bool:
        """
        Process AXI Read Address channel.

        Args:
            ar: Read address request.
            timestamp: Current simulation time.

        Returns:
            True if accepted.
        """
        if len(self._active_transactions) >= self.config.max_outstanding:
            return False

        # Also check if output buffer has space for at least one packet
        # This provides backpressure when the NI is congested
        if self.output_buffer.free_space < 1:
            return False

        self.stats.ar_received += 1

        # Extract destination based on routing mode
        if self.config.use_user_signal_routing:
            # NoC-to-NoC mode: destination from AXI user signal
            # Format: aruser[7:0] = dest_x, aruser[15:8] = dest_y
            aruser = getattr(ar, 'aruser', 0) or 0
            dest_x = aruser & 0xFF
            dest_y = (aruser >> 8) & 0xFF
            dest_coord = (dest_x, dest_y)
            local_addr = ar.araddr & 0xFFFFFFFF  # 32-bit local address
        else:
            # Host-to-NoC mode: destination from address encoding
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
            timestamp_start=timestamp,
            src_coord=self.coord,
            dest_coord=dest_coord,
            local_addr=local_addr,
            ar=ar,
            r_beats_expected=ar.burst_length,
        )

        # Assemble into flits (Pack AR)
        flits = self.packet_assembler.assemble(packet)

        # Queue flits
        for flit in flits:
            self.output_buffer.push(flit)

        self._active_transactions[packet.rob_idx] = txn
        self._rob_to_axi[packet.rob_idx] = ar.arid

        self.stats.read_requests += 1
        return True

    def get_output_flit(self) -> Optional[Flit]:
        """Get next flit to send to NoC."""
        flit = self.output_buffer.pop()
        if flit is not None:
            self.stats.req_flits_sent += 1

            # Update transaction state
            if flit.hdr.rob_idx in self._active_transactions:
                txn = self._active_transactions[flit.hdr.rob_idx]
                txn.state = TransactionState.IN_FLIGHT

        return flit

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

    def has_pending_output(self) -> bool:
        """Check if there are flits waiting to be sent."""
        return not self.output_buffer.is_empty()

    def process_cycle(self, timestamp: int = 0) -> None:
        """Process one cycle - try to send any ready packets."""
        self._try_send_ready_packets(timestamp)

    @property
    def outstanding_count(self) -> int:
        """Number of outstanding transactions."""
        return (len(self._active_transactions) +
                len(self._pending_writes) +
                len(self._ready_to_send))


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
        self._on_transaction_complete: Optional[Callable[[int, int], None]] = None

    def set_completion_callback(
        self,
        callback: Callable[[int, int], None]
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
    def get_req_flit(self) -> Optional[Flit]:
        """Get request flit to send to NoC (Request Router)."""
        return self.req_path.get_output_flit()

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
        # Check outstanding limit and output buffer space
        return (
            len(self.req_path._active_transactions) < self.config.max_outstanding
            and self.req_path.output_buffer.free_space >= 1
        )

    @property
    def w_ready(self) -> bool:
        """Check if W channel can accept data beat."""
        # W is ready if there's a pending write transaction
        return len(self.req_path._pending_writes) > 0

    @property
    def ar_ready(self) -> bool:
        """Check if AR channel can accept new request."""
        # Same checks as AW
        return (
            len(self.req_path._active_transactions) < self.config.max_outstanding
            and self.req_path.output_buffer.free_space >= 1
        )

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

        # Output buffer for response flits (to Response Router)
        self.resp_output = FlitBuffer(
            self.config.resp_buffer_depth,
            f"MasterNI({coord})_rsp_out"
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

        # R beat accumulation buffer for multi-beat reads
        # Keyed by AXI ID -> accumulated bytes
        self._pending_r_data: Dict[int, bytes] = {}
        
        # Queue for flits waiting to be sent (Wormhole: one flit per cycle)
        self._pending_flits: Deque[Flit] = deque()

        # Packet arrival callback for metrics collection
        # Called with (packet_id, creation_time, arrival_time) when a packet arrives
        self._packet_arrival_callback: Optional[Callable[[int, int, int], None]] = None

        # === Valid/Ready Interface Signals ===
        # Request input (Router LOCAL → NI)
        self.req_in_valid: bool = False
        self.req_in_flit: Optional[Flit] = None
        self.req_in_ready: bool = True  # Can accept if buffer not full

        # Response output (NI → Router LOCAL)
        self.resp_out_valid: bool = False
        self.resp_out_flit: Optional[Flit] = None
        self.resp_out_ready: bool = False  # Set by downstream (router)

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

    def set_packet_arrival_callback(self, callback: Callable[[int, int, int], None]) -> None:
        """
        Set callback for packet arrival notification.

        Args:
            callback: Function that receives (packet_id, creation_time, arrival_time).
        """
        self._packet_arrival_callback = callback

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
        return self.resp_output.pop()

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

            # Try to reconstruct packet
            packet = self.packet_disassembler.receive_flit(flit)
            if packet is not None:
                self._process_request_packet(packet, current_time)

    def _process_request_packet(self, packet: Packet, current_time: int) -> None:
        """
        Process a complete request packet.

        Converts packet to AXI request, stores info in Per-ID FIFO,
        and forwards to Memory.
        """
        # Notify packet arrival for metrics collection
        if self._packet_arrival_callback:
            self._packet_arrival_callback(packet.rob_idx, 0, current_time)

        # Store request info in Per-ID FIFO for response routing
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

        # Convert to AXI request and forward to Memory
        if packet.packet_type == PacketType.WRITE_REQ:
            self._forward_write_request(packet, current_time)
        elif packet.packet_type == PacketType.READ_REQ:
            self._forward_read_request(packet, current_time)

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

        IMPORTANT: Check resp_output buffer space before collecting to
        prevent packet drops due to backpressure.
        """
        # Collect B responses (write complete)
        while self.axi_slave.has_pending_b():
            # Check if we have space for at least 1 response flit
            if self.resp_output.free_space < 1:
                break  # Backpressure - wait for next cycle

            b_resp = self.axi_slave.get_b_response()
            if b_resp is None:
                break

            # Match with Per-ID FIFO (Routing Logic)
            axi_id = b_resp.bid
            if axi_id in self._per_id_fifo and self._per_id_fifo[axi_id]:
                req_info = self._per_id_fifo[axi_id].popleft()

                # Create response packet with routing info
                resp_packet = PacketFactory.create_write_response_from_info(
                    src=self.coord,
                    dest=req_info.src_coord,  # Route back to source
                    axi_id=axi_id,
                )

                # Pack into flits
                flits = self.packet_assembler.assemble(resp_packet)
                for flit in flits:
                    if not self.resp_output.push(flit):
                        # Should not happen if we checked space, but log it
                        self.stats.resp_flits_dropped = getattr(
                            self.stats, 'resp_flits_dropped', 0
                        ) + 1

                self.stats.b_responses_sent += 1

        # Collect R responses (read data)
        # Wormhole: First try to send pending flits (one per cycle)
        self._try_send_pending_flits()
        
        # Only collect new R beats if no packet is currently being transmitted
        # This ensures one packet at a time, preventing flit interleaving
        if self._pending_flits:
            return  # Wait until current packet is fully sent
        
        # Collect R beats from AXI slave
        while self.axi_slave.has_pending_r():
            r_resp = self.axi_slave.get_r_response()
            if r_resp is None:
                break

            # Accumulate R beat data by AXI ID
            axi_id = r_resp.rid
            if axi_id not in self._pending_r_data:
                self._pending_r_data[axi_id] = b""
            self._pending_r_data[axi_id] = self._pending_r_data[axi_id] + r_resp.rdata

            # Only create response packet on rlast (complete read)
            if not r_resp.rlast:
                continue

            # Get accumulated data for this transaction
            accumulated_data = self._pending_r_data.pop(axi_id, r_resp.rdata)

            # Match with Per-ID FIFO (Routing Logic)
            if axi_id in self._per_id_fifo and self._per_id_fifo[axi_id]:
                req_info = self._per_id_fifo[axi_id].popleft()
                
                # Create response packet and queue flits (Wormhole flow)
                resp_packet = PacketFactory.create_read_response_from_info(
                    src=self.coord,
                    dest=req_info.src_coord,
                    axi_id=axi_id,
                    data=accumulated_data,
                )

                # Assemble into flits and queue for sending
                flits = self.packet_assembler.assemble(resp_packet)
                for flit in flits:
                    self._pending_flits.append(flit)
                
                self.stats.r_responses_sent += 1
                
                # Only queue one response packet per cycle (wormhole serialization)
                break

    def _try_send_pending_flits(self) -> None:
        """
        Wormhole flow control: send ONE flit per cycle.
        
        This is the correct wormhole behavior. The serialization of packets
        should be handled by not collecting new AXI responses while a packet
        is still being transmitted (has flits in _pending_flits).
        """
        # Send one flit per cycle (wormhole behavior)
        if self._pending_flits and self.resp_output.free_space >= 1:
            flit = self._pending_flits.popleft()
            if not self.resp_output.push(flit):
                # Put back if failed
                self._pending_flits.appendleft(flit)
                self.stats.resp_flits_dropped = getattr(
                    self.stats, 'resp_flits_dropped', 0
                ) + 1

    def _process_deferred_responses(self, current_time: int) -> None:
        """
        Process responses that were deferred (put-back) from previous cycles.
        
        This handles the case where accumulated R data exists in _pending_r_data
        with a matching entry in _per_id_fifo, but was deferred because
        _pending_flits was not empty at the time.
        """
        # Find a deferred response to process
        for axi_id in list(self._pending_r_data.keys()):
            if axi_id in self._per_id_fifo and self._per_id_fifo[axi_id]:
                # Found deferred response - process it
                accumulated_data = self._pending_r_data.pop(axi_id)
                req_info = self._per_id_fifo[axi_id].popleft()
                
                # Create response packet and queue flits
                resp_packet = PacketFactory.create_read_response_from_info(
                    src=self.coord,
                    dest=req_info.src_coord,
                    axi_id=axi_id,
                    data=accumulated_data,
                )

                flits = self.packet_assembler.assemble(resp_packet)
                for flit in flits:
                    self._pending_flits.append(flit)
                
                self.stats.r_responses_sent += 1
                break  # Only process one per cycle

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
        return not self.resp_output.is_empty()

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
