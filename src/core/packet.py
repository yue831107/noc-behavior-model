"""
Packet encapsulation and flit assembly/disassembly - FlooNoC Style.

A Packet represents a complete NoC transaction, consisting of
multiple FlooNoC-style Flits (AW+W* for writes, AR for reads, B/R* for responses).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto

from .flit import (
    Flit, FlitHeader, FlitFactory, AxiChannel,
    AxiAwPayload, AxiWPayload, AxiArPayload, AxiBPayload, AxiRPayload,
    encode_node_id, decode_node_id,
)


class PacketType(Enum):
    """Packet type based on AXI transaction."""
    WRITE_REQ = auto()   # AXI Write Request (AW + W)
    WRITE_RESP = auto()  # AXI Write Response (B)
    READ_REQ = auto()    # AXI Read Request (AR)
    READ_RESP = auto()   # AXI Read Response (R)


@dataclass
class Packet:
    """
    Packet data structure.

    A packet is a complete unit of NoC transaction, composed of
    one or more FlooNoC-style flits.

    Attributes:
        packet_id: Unique packet identifier (rob_idx for tracking).
        packet_type: Type of packet (WRITE_REQ, READ_REQ, etc.).
        src: Source coordinate (x, y).
        dest: Destination coordinate (x, y).
        axi_id: AXI transaction ID.
        local_addr: 32-bit local address at destination.
        payload: Complete payload data.
        read_size: For READ_REQ: number of bytes to read.
        rob_idx: RoB index for response matching.
    """
    packet_id: int
    packet_type: PacketType
    src: Tuple[int, int]
    dest: Tuple[int, int]
    axi_id: int = 0
    local_addr: int = 0
    payload: bytes = field(default_factory=bytes)
    read_size: int = 0
    rob_idx: int = 0
    timestamp: int = 0

    # Tracking
    flits: List[Flit] = field(default_factory=list)
    _assembled: bool = field(default=False, repr=False)

    # For backward compatibility
    src_ni_id: int = 0

    @property
    def is_request(self) -> bool:
        """Check if this is a request packet."""
        return self.packet_type in (PacketType.WRITE_REQ, PacketType.READ_REQ)

    @property
    def is_response(self) -> bool:
        """Check if this is a response packet."""
        return self.packet_type in (PacketType.WRITE_RESP, PacketType.READ_RESP)

    @property
    def flit_count(self) -> int:
        """Number of flits in this packet."""
        return len(self.flits)

    @property
    def payload_size(self) -> int:
        """Total payload size in bytes."""
        return len(self.payload)

    def __repr__(self) -> str:
        return (
            f"Packet(id={self.packet_id}, {self.packet_type.name}, "
            f"{self.src}->{self.dest}, flits={self.flit_count})"
        )


class PacketAssembler:
    """
    Assembles a Packet into FlooNoC-style Flits for transmission.

    Write request: AW + W* (one or more W flits)
    Read request: AR (single flit)
    Write response: B (single flit)
    Read response: R* (one or more R flits)
    """

    # FlooNoC flit payload size is 32 bytes (256 bits)
    FLIT_PAYLOAD_SIZE = 32

    def __init__(self, flit_payload_size: int = 32):
        """
        Initialize assembler.

        Args:
            flit_payload_size: Maximum payload bytes per flit (default 32).
        """
        self.flit_payload_size = flit_payload_size

    def assemble(self, packet: Packet, timestamp: int = 0) -> List[Flit]:
        """
        Assemble a packet into FlooNoC-style flits.

        Args:
            packet: Packet to assemble.
            timestamp: Current simulation timestamp.

        Returns:
            List of flits representing the packet.
        """
        if packet._assembled:
            return packet.flits

        if packet.packet_type == PacketType.WRITE_REQ:
            flits = self._assemble_write_request(packet)
        elif packet.packet_type == PacketType.READ_REQ:
            flits = self._assemble_read_request(packet)
        elif packet.packet_type == PacketType.WRITE_RESP:
            flits = self._assemble_write_response(packet)
        elif packet.packet_type == PacketType.READ_RESP:
            flits = self._assemble_read_response(packet)
        else:
            raise ValueError(f"Unknown packet type: {packet.packet_type}")

        packet.flits = flits
        packet._assembled = True
        return flits

    def _assemble_write_request(self, packet: Packet) -> List[Flit]:
        """Assemble write request into AW + W* flits."""
        flits = []
        data = packet.payload
        data_len = len(data)

        # Calculate number of W flits needed
        if data_len == 0:
            num_w_flits = 1
        else:
            num_w_flits = (data_len + self.flit_payload_size - 1) // self.flit_payload_size

        # AW flit (FlooNoC: AW is single-flit packet, last=True)
        aw_flit = FlitFactory.create_aw(
            src=packet.src,
            dest=packet.dest,
            addr=packet.local_addr,
            axi_id=packet.axi_id,
            length=num_w_flits - 1,  # awlen = num_beats - 1
            size=5,  # 32 bytes
            burst=1,  # INCR
            rob_idx=packet.rob_idx,
            rob_req=True,
            last=True,  # FlooNoC: AW is single-flit packet
        )
        flits.append(aw_flit)

        # W flits
        offset = 0
        for i in range(num_w_flits):
            is_last_w = (i == num_w_flits - 1)

            # Get data for this flit
            if data_len == 0:
                chunk = bytes(self.flit_payload_size)
                strb = 0  # No valid bytes
            else:
                chunk = data[offset:offset + self.flit_payload_size]
                valid_bytes = len(chunk)

                # Calculate strb mask based on actual valid bytes
                # strb bit i = 1 means byte i is valid
                if valid_bytes >= self.flit_payload_size:
                    strb = (1 << self.flit_payload_size) - 1  # All bytes valid
                else:
                    # Only first valid_bytes are valid
                    strb = (1 << valid_bytes) - 1

                # Pad chunk to flit_payload_size
                if len(chunk) < self.flit_payload_size:
                    chunk = chunk + bytes(self.flit_payload_size - len(chunk))

                offset += self.flit_payload_size

            w_flit = FlitFactory.create_w(
                src=packet.src,
                dest=packet.dest,
                data=chunk,
                strb=strb,
                last=is_last_w,
                rob_idx=packet.rob_idx,
                seq_num=i,
            )
            flits.append(w_flit)

        return flits

    def _assemble_read_request(self, packet: Packet) -> List[Flit]:
        """Assemble read request into AR flit."""
        # Calculate number of R flits expected in response
        if packet.read_size == 0:
            length = 0
        else:
            length = (packet.read_size + self.flit_payload_size - 1) // self.flit_payload_size - 1

        ar_flit = FlitFactory.create_ar(
            src=packet.src,
            dest=packet.dest,
            addr=packet.local_addr,
            axi_id=packet.axi_id,
            length=length,
            size=5,  # 32 bytes
            burst=1,  # INCR
            rob_idx=packet.rob_idx,
            rob_req=True,
        )
        return [ar_flit]

    def _assemble_write_response(self, packet: Packet) -> List[Flit]:
        """Assemble write response into B flit."""
        b_flit = FlitFactory.create_b(
            src=packet.src,
            dest=packet.dest,
            axi_id=packet.axi_id,
            resp=0,  # OKAY
            rob_idx=packet.rob_idx,
        )
        return [b_flit]

    def _assemble_read_response(self, packet: Packet) -> List[Flit]:
        """Assemble read response into R* flits."""
        flits = []
        data = packet.payload
        data_len = len(data)

        if data_len == 0:
            # Single R flit with no data
            r_flit = FlitFactory.create_r(
                src=packet.src,
                dest=packet.dest,
                data=bytes(self.flit_payload_size),
                axi_id=packet.axi_id,
                resp=0,
                last=True,
                rob_idx=packet.rob_idx,
                seq_num=0,
            )
            return [r_flit]

        # Multiple R flits
        num_r_flits = (data_len + self.flit_payload_size - 1) // self.flit_payload_size
        offset = 0

        for i in range(num_r_flits):
            is_last_r = (i == num_r_flits - 1)
            chunk = data[offset:offset + self.flit_payload_size]

            # Pad chunk to flit_payload_size
            if len(chunk) < self.flit_payload_size:
                chunk = chunk + bytes(self.flit_payload_size - len(chunk))

            offset += self.flit_payload_size

            r_flit = FlitFactory.create_r(
                src=packet.src,
                dest=packet.dest,
                data=chunk,
                axi_id=packet.axi_id,
                resp=0,
                last=is_last_r,
                rob_idx=packet.rob_idx,
                seq_num=i,
            )
            flits.append(r_flit)

        return flits

    def calculate_flit_count(self, payload_size: int) -> int:
        """
        Calculate number of flits needed for given payload size.

        For write: 1 AW + ceil(payload_size / flit_payload_size) W flits
        """
        if payload_size == 0:
            return 2  # AW + 1 W
        w_flits = (payload_size + self.flit_payload_size - 1) // self.flit_payload_size
        return 1 + w_flits  # AW + W*


class PacketDisassembler:
    """
    Disassembles FlooNoC-style Flits back into a Packet.

    Collects flits and reconstructs the original packet.
    Uses strb mask to extract only valid bytes from W flits.
    """

    # FlooNoC flit payload size is 32 bytes (256 bits)
    FLIT_PAYLOAD_SIZE = 32

    def __init__(self):
        """Initialize disassembler."""
        # Pending packets: key = (src_id, rob_idx), value = list of flits
        self._pending: dict[tuple[int, int], List[Flit]] = {}

    def receive_flit(self, flit: Flit) -> Optional[Packet]:
        """
        Receive a flit and attempt to reconstruct packet.

        Args:
            flit: Received flit.

        Returns:
            Reconstructed Packet if complete, None otherwise.
        """
        # Use src_id and rob_idx as key for matching
        key = (flit.hdr.src_id, flit.hdr.rob_idx)

        # Single flit packet
        if flit.is_single_flit():
            return self._create_packet_from_flits([flit])

        # Multi-flit packet
        if key not in self._pending:
            if not flit.is_head():
                # Received non-HEAD as first flit - error state
                return None
            self._pending[key] = []

        self._pending[key].append(flit)

        # Check if packet is complete
        if flit.is_tail():
            flits = self._pending.pop(key)
            return self._create_packet_from_flits(flits)

        return None

    def _count_valid_bytes(self, strb: int) -> int:
        """
        Count number of valid bytes from strb mask.

        Assumes contiguous valid bytes starting from byte 0.

        Args:
            strb: 32-bit strobe mask.

        Returns:
            Number of valid bytes.
        """
        if strb == 0xFFFFFFFF:
            return 32
        if strb == 0:
            return 0

        # Count contiguous bits from LSB
        count = 0
        mask = strb
        while mask & 1:
            count += 1
            mask >>= 1
        return count

    def _create_packet_from_flits(self, flits: List[Flit]) -> Packet:
        """
        Create a Packet from a list of FlooNoC-style flits.

        Uses strb to extract only valid bytes from W flits.

        Args:
            flits: List of flits.

        Returns:
            Reconstructed Packet.
        """
        if not flits:
            raise ValueError("Cannot create packet from empty flit list")

        head = flits[0]
        hdr = head.hdr

        # Determine packet type from first flit's AXI channel
        if hdr.axi_ch == AxiChannel.AW:
            packet_type = PacketType.WRITE_REQ
        elif hdr.axi_ch == AxiChannel.AR:
            packet_type = PacketType.READ_REQ
        elif hdr.axi_ch == AxiChannel.B:
            packet_type = PacketType.WRITE_RESP
        elif hdr.axi_ch == AxiChannel.R:
            packet_type = PacketType.READ_RESP
        elif hdr.axi_ch == AxiChannel.W:
            # W should not be first flit, but handle it
            packet_type = PacketType.WRITE_REQ
        else:
            packet_type = PacketType.WRITE_REQ

        # Extract metadata from header flit
        src = decode_node_id(hdr.src_id)
        dest = decode_node_id(hdr.dst_id)
        rob_idx = hdr.rob_idx
        axi_id = 0
        local_addr = 0
        read_size = 0

        if isinstance(head.payload, AxiAwPayload):
            axi_id = head.payload.axi_id
            local_addr = head.payload.addr
        elif isinstance(head.payload, AxiArPayload):
            axi_id = head.payload.axi_id
            local_addr = head.payload.addr
            # Calculate read_size from length
            read_size = (head.payload.length + 1) * self.FLIT_PAYLOAD_SIZE
        elif isinstance(head.payload, AxiBPayload):
            axi_id = head.payload.axi_id
        elif isinstance(head.payload, AxiRPayload):
            axi_id = head.payload.axi_id

        # Reconstruct payload data
        payload = b""

        for flit in flits:
            pl = flit.payload

            if isinstance(pl, AxiWPayload):
                # Use strb to extract only valid bytes - THIS IS THE FIX
                valid_bytes = self._count_valid_bytes(pl.strb)
                if valid_bytes > 0:
                    payload += pl.data[:valid_bytes]

            elif isinstance(pl, AxiRPayload):
                # R flits always have full data (no strb)
                # For last R flit, we might need to trim based on expected size
                # But since read_size tracking is complex, include all data
                payload += pl.data

        packet = Packet(
            packet_id=rob_idx,
            packet_type=packet_type,
            src=src,
            dest=dest,
            axi_id=axi_id,
            local_addr=local_addr,
            payload=payload,
            read_size=read_size,
            rob_idx=rob_idx,
        )
        packet.flits = flits
        packet._assembled = True

        return packet

    def clear(self) -> None:
        """Clear all pending flits."""
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        """Number of packets currently being assembled."""
        return len(self._pending)


class PacketFactory:
    """Factory for creating packets."""

    _rob_idx_counter: int = 0

    @classmethod
    def _next_rob_idx(cls) -> int:
        """Generate next RoB index (wraps at 32)."""
        idx = cls._rob_idx_counter
        cls._rob_idx_counter = (cls._rob_idx_counter + 1) % 32
        return idx

    @classmethod
    def reset_packet_id(cls) -> None:
        """Reset packet ID counter (for testing)."""
        cls._rob_idx_counter = 0

    @classmethod
    def create_write_request(
        cls,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        local_addr: int,
        data: bytes,
        axi_id: int = 0,
        src_ni_id: int = 0,
        timestamp: int = 0,
    ) -> Packet:
        """Create a write request packet."""
        rob_idx = cls._next_rob_idx()
        return Packet(
            packet_id=rob_idx,
            packet_type=PacketType.WRITE_REQ,
            src=src,
            dest=dest,
            axi_id=axi_id,
            local_addr=local_addr,
            payload=data,
            rob_idx=rob_idx,
            timestamp=timestamp,
            src_ni_id=src_ni_id,
        )

    @classmethod
    def create_read_request(
        cls,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        local_addr: int,
        read_size: int = 0,
        axi_id: int = 0,
        src_ni_id: int = 0,
        timestamp: int = 0,
    ) -> Packet:
        """Create a read request packet."""
        rob_idx = cls._next_rob_idx()
        return Packet(
            packet_id=rob_idx,
            packet_type=PacketType.READ_REQ,
            src=src,
            dest=dest,
            axi_id=axi_id,
            local_addr=local_addr,
            payload=b"",
            read_size=read_size,
            rob_idx=rob_idx,
            timestamp=timestamp,
            src_ni_id=src_ni_id,
        )

    @classmethod
    def create_write_response(
        cls,
        request: Packet,
        timestamp: int = 0,
    ) -> Packet:
        """Create a write response packet from a request."""
        return Packet(
            packet_id=request.rob_idx,
            packet_type=PacketType.WRITE_RESP,
            src=request.dest,
            dest=request.src,
            axi_id=request.axi_id,
            local_addr=request.local_addr,
            payload=b"",
            rob_idx=request.rob_idx,
            timestamp=timestamp,
        )

    @classmethod
    def create_read_response(
        cls,
        request: Packet,
        data: bytes,
        timestamp: int = 0,
    ) -> Packet:
        """Create a read response packet from a request."""
        return Packet(
            packet_id=request.rob_idx,
            packet_type=PacketType.READ_RESP,
            src=request.dest,
            dest=request.src,
            axi_id=request.axi_id,
            local_addr=request.local_addr,
            payload=data,
            rob_idx=request.rob_idx,
            timestamp=timestamp,
        )

    @classmethod
    def create_write_response_from_info(
        cls,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        axi_id: int,
        timestamp: int = 0,
    ) -> Packet:
        """
        Create a write response packet from routing info.

        Used by MasterNI when it doesn't have the original request packet.
        """
        rob_idx = cls._next_rob_idx()
        return Packet(
            packet_id=rob_idx,
            packet_type=PacketType.WRITE_RESP,
            src=src,
            dest=dest,
            axi_id=axi_id,
            local_addr=0,
            payload=b"",
            rob_idx=rob_idx,
            timestamp=timestamp,
        )

    @classmethod
    def create_read_response_from_info(
        cls,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        axi_id: int,
        data: bytes,
        timestamp: int = 0,
        rob_idx: Optional[int] = None,
    ) -> Packet:
        """
        Create a read response packet from routing info.

        Used by MasterNI when it doesn't have the original request packet.

        Args:
            src: Source coordinate.
            dest: Destination coordinate.
            axi_id: AXI transaction ID.
            data: Read data payload.
            timestamp: Packet timestamp.
            rob_idx: RoB index (uses original request's rob_idx for matching).
                     If None, generates a new one (not recommended for multi-flit).
        """
        if rob_idx is None:
            rob_idx = cls._next_rob_idx()
        return Packet(
            packet_id=rob_idx,
            packet_type=PacketType.READ_RESP,
            src=src,
            dest=dest,
            axi_id=axi_id,
            local_addr=0,
            payload=data,
            rob_idx=rob_idx,
            timestamp=timestamp,
        )
