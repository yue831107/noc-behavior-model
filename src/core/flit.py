"""
Flit (Flow Control Unit) data structure - FlooNoC Style.

Flit is the basic unit of data transfer in NoC. Uses FlooNoC-style format
with 20-bit header and AXI channel-specific payloads.

Header (20 bits):
  [0]     rob_req   - RoB request flag
  [5:1]   rob_idx   - RoB index (32 entries)
  [10:6]  dst_id    - Destination node {x[2:0], y[1:0]}
  [15:11] src_id    - Source node {x[2:0], y[1:0]}
  [16]    last      - Last flit of packet
  [19:17] axi_ch    - AXI channel type

Physical Links:
  - Request: 310 bits (valid + ready + 308-bit flit)
  - Response: 288 bits (valid + ready + 286-bit flit)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Union


# =============================================================================
# Constants
# =============================================================================

# Node ID encoding: 5 bits = x[2:0] (3 bits) + y[1:0] (2 bits)
X_BITS = 3
Y_BITS = 2
NODE_ID_BITS = X_BITS + Y_BITS  # 5 bits

# RoB configuration
ROB_IDX_BITS = 5  # 32 entries max

# AXI configuration
AXI_ID_WIDTH = 8      # 8-bit AXI ID
AXI_ADDR_WIDTH = 32   # 32-bit address
AXI_DATA_WIDTH = 256  # 256-bit data (32 bytes)
AXI_STRB_WIDTH = AXI_DATA_WIDTH // 8  # 32-bit strobe


# =============================================================================
# Enums
# =============================================================================

class AxiChannel(IntEnum):
    """AXI Channel types (3 bits)."""
    AW = 0  # Write Address (Request)
    W = 1   # Write Data (Request)
    AR = 2  # Read Address (Request)
    B = 3   # Write Response (Response)
    R = 4   # Read Response (Response)

    def is_request(self) -> bool:
        """Check if this is a request channel."""
        return self in (AxiChannel.AW, AxiChannel.W, AxiChannel.AR)

    def is_response(self) -> bool:
        """Check if this is a response channel."""
        return self in (AxiChannel.B, AxiChannel.R)


# =============================================================================
# Helper Functions
# =============================================================================

def encode_node_id(coord: tuple[int, int]) -> int:
    """
    Encode (x, y) coordinate to 5-bit node_id.

    Format: {x[2:0], y[1:0]}

    Args:
        coord: (x, y) coordinate tuple

    Returns:
        5-bit node ID

    Raises:
        ValueError: If coordinates are out of valid range.
    """
    x, y = coord
    max_x = (1 << X_BITS) - 1  # 7
    max_y = (1 << Y_BITS) - 1  # 3

    if not (0 <= x <= max_x):
        raise ValueError(f"x coordinate {x} out of range [0, {max_x}]")
    if not (0 <= y <= max_y):
        raise ValueError(f"y coordinate {y} out of range [0, {max_y}]")

    return (x << Y_BITS) | y


def decode_node_id(node_id: int) -> tuple[int, int]:
    """
    Decode 5-bit node_id to (x, y) coordinate.

    Format: {x[2:0], y[1:0]}

    Args:
        node_id: 5-bit node ID

    Returns:
        (x, y) coordinate tuple

    Raises:
        ValueError: If node_id is out of valid range.
    """
    max_node_id = (1 << NODE_ID_BITS) - 1  # 31

    if not (0 <= node_id <= max_node_id):
        raise ValueError(f"node_id {node_id} out of range [0, {max_node_id}]")

    x = (node_id >> Y_BITS) & 0x7
    y = node_id & 0x3
    return (x, y)


# =============================================================================
# AXI Payload Structures
# =============================================================================

@dataclass
class AxiAwPayload:
    """
    AXI Write Address (AW) channel payload - 53 bits.

    Used for write address phase of AXI transaction.
    """
    addr: int = 0       # 32 bits - write address
    axi_id: int = 0     # 8 bits - transaction ID
    length: int = 0     # 8 bits - burst length (awlen)
    size: int = 0       # 3 bits - burst size (awsize)
    burst: int = 1      # 2 bits - burst type (awburst), default INCR

    def __repr__(self) -> str:
        return f"AW(addr=0x{self.addr:08x}, id={self.axi_id}, len={self.length})"


@dataclass
class AxiWPayload:
    """
    AXI Write Data (W) channel payload - 288 bits.

    Used for write data phase of AXI transaction.
    """
    data: bytes = field(default_factory=lambda: bytes(32))  # 256 bits (32 bytes)
    strb: int = 0xFFFFFFFF  # 32 bits - write strobe

    def __repr__(self) -> str:
        return f"W(data={len(self.data)}B, strb=0x{self.strb:08x})"


@dataclass
class AxiArPayload:
    """
    AXI Read Address (AR) channel payload - 53 bits.

    Used for read address phase of AXI transaction.
    """
    addr: int = 0       # 32 bits - read address
    axi_id: int = 0     # 8 bits - transaction ID
    length: int = 0     # 8 bits - burst length (arlen)
    size: int = 0       # 3 bits - burst size (arsize)
    burst: int = 1      # 2 bits - burst type (arburst), default INCR

    def __repr__(self) -> str:
        return f"AR(addr=0x{self.addr:08x}, id={self.axi_id}, len={self.length})"


@dataclass
class AxiBPayload:
    """
    AXI Write Response (B) channel payload - 10 bits.

    Used for write response phase of AXI transaction.
    """
    axi_id: int = 0     # 8 bits - transaction ID
    resp: int = 0       # 2 bits - response status (OKAY=0, EXOKAY=1, SLVERR=2, DECERR=3)

    def __repr__(self) -> str:
        resp_names = ["OKAY", "EXOKAY", "SLVERR", "DECERR"]
        return f"B(id={self.axi_id}, resp={resp_names[self.resp]})"


@dataclass
class AxiRPayload:
    """
    AXI Read Response (R) channel payload - 266 bits.

    Used for read response phase of AXI transaction.
    """
    data: bytes = field(default_factory=lambda: bytes(32))  # 256 bits (32 bytes)
    axi_id: int = 0     # 8 bits - transaction ID
    resp: int = 0       # 2 bits - response status

    def __repr__(self) -> str:
        return f"R(data={len(self.data)}B, id={self.axi_id})"


# Union type for all payloads
FlitPayload = Union[AxiAwPayload, AxiWPayload, AxiArPayload, AxiBPayload, AxiRPayload, None]


# =============================================================================
# Flit Header
# =============================================================================

@dataclass
class FlitHeader:
    """
    FlooNoC-style Flit Header - 20 bits.

    | Bit     | Field   | Width | Description                    |
    |---------|---------|-------|--------------------------------|
    | [0]     | rob_req | 1     | RoB request flag               |
    | [5:1]   | rob_idx | 5     | RoB index (0-31)               |
    | [10:6]  | dst_id  | 5     | Destination {x[2:0], y[1:0]}   |
    | [15:11] | src_id  | 5     | Source {x[2:0], y[1:0]}        |
    | [16]    | last    | 1     | Last flit of packet            |
    | [19:17] | axi_ch  | 3     | AXI channel type               |
    """
    rob_req: bool = False       # 1 bit - RoB request flag
    rob_idx: int = 0            # 5 bits - RoB index (0-31)
    dst_id: int = 0             # 5 bits - destination node ID
    src_id: int = 0             # 5 bits - source node ID
    last: bool = True           # 1 bit - last flit marker
    axi_ch: AxiChannel = AxiChannel.AW  # 3 bits - AXI channel type

    @property
    def src(self) -> tuple[int, int]:
        """Get source coordinate as (x, y) tuple."""
        return decode_node_id(self.src_id)

    @property
    def dest(self) -> tuple[int, int]:
        """Get destination coordinate as (x, y) tuple."""
        return decode_node_id(self.dst_id)

    def is_request(self) -> bool:
        """Check if this is a request flit."""
        return self.axi_ch.is_request()

    def is_response(self) -> bool:
        """Check if this is a response flit."""
        return self.axi_ch.is_response()

    def to_int(self) -> int:
        """Pack header to 20-bit integer."""
        value = 0
        value |= (1 if self.rob_req else 0)         # bit 0
        value |= (self.rob_idx & 0x1F) << 1         # bits 5:1
        value |= (self.dst_id & 0x1F) << 6          # bits 10:6
        value |= (self.src_id & 0x1F) << 11         # bits 15:11
        value |= (1 if self.last else 0) << 16      # bit 16
        value |= (int(self.axi_ch) & 0x7) << 17     # bits 19:17
        return value

    @classmethod
    def from_int(cls, value: int) -> FlitHeader:
        """
        Unpack header from 20-bit integer.

        Raises:
            ValueError: If axi_ch value is invalid.
        """
        axi_ch_val = (value >> 17) & 0x7
        if axi_ch_val > 4:
            raise ValueError(f"Invalid axi_ch value: {axi_ch_val} (valid range: 0-4)")

        return cls(
            rob_req=bool(value & 0x1),
            rob_idx=(value >> 1) & 0x1F,
            dst_id=(value >> 6) & 0x1F,
            src_id=(value >> 11) & 0x1F,
            last=bool((value >> 16) & 0x1),
            axi_ch=AxiChannel(axi_ch_val),
        )

    def __repr__(self) -> str:
        src = decode_node_id(self.src_id)
        dst = decode_node_id(self.dst_id)
        return (
            f"Hdr({src}→{dst}, {self.axi_ch.name}, "
            f"last={self.last}, rob_idx={self.rob_idx})"
        )


# =============================================================================
# Flit
# =============================================================================

@dataclass
class Flit:
    """
    FlooNoC-style Flit structure.

    A flit consists of a header (20 bits) and a channel-specific payload.
    The payload size depends on the AXI channel type:

    | Channel | Header | Payload  | Total     |
    |---------|--------|----------|-----------|
    | AW      | 20     | 53 bits  | 73 bits   |
    | W       | 20     | 288 bits | 308 bits  |
    | AR      | 20     | 53 bits  | 73 bits   |
    | B       | 20     | 10 bits  | 30 bits   |
    | R       | 20     | 266 bits | 286 bits  |

    Request flits are union-aligned to 308 bits.
    Response flits are union-aligned to 286 bits.
    """
    hdr: FlitHeader = field(default_factory=FlitHeader)
    payload: FlitPayload = None

    # Internal tracking (not part of wire format)
    _seq_num: int = field(default=0, repr=False)
    injection_cycle: Optional[int] = field(default=None, repr=False)

    def is_head(self) -> bool:
        """Check if this flit starts a packet.

        FlooNoC packet structure:
        - AW: Single flit packet (last=1)
        - AR: Single flit packet (last=1)
        - B: Single flit packet (last=1)
        - W: Independent data packet (first W is HEAD, last W has last=1)
        - R: Independent data packet (first R is HEAD, last R has last=1)
        """
        # AW, AR, B are always single-flit packets (HEAD)
        if self.hdr.axi_ch in (AxiChannel.AW, AxiChannel.AR, AxiChannel.B):
            return True
        # W and R: first flit (seq_num == 0) is HEAD
        return self._seq_num == 0

    def is_tail(self) -> bool:
        """Check if this is the last flit of a packet."""
        return self.hdr.last

    def is_single_flit(self) -> bool:
        """Check if this is a single-flit packet."""
        return self.is_head() and self.is_tail()

    @property
    def src(self) -> tuple[int, int]:
        """Get source coordinate."""
        return self.hdr.src

    @property
    def dest(self) -> tuple[int, int]:
        """Get destination coordinate."""
        return self.hdr.dest

    @property
    def is_request(self) -> bool:
        """Check if this is a request flit."""
        return self.hdr.is_request()

    def __repr__(self) -> str:
        payload_str = ""
        if self.payload:
            payload_str = f", {self.payload}"
        return f"Flit({self.hdr}{payload_str})"


# =============================================================================
# Flit Factory
# =============================================================================

class FlitFactory:
    """Factory for creating FlooNoC-style flits."""

    _rob_idx_counter: int = 0

    @classmethod
    def _next_rob_idx(cls) -> int:
        """Generate next RoB index (wraps at 32)."""
        idx = cls._rob_idx_counter
        cls._rob_idx_counter = (cls._rob_idx_counter + 1) % 32
        return idx

    @classmethod
    def reset(cls) -> None:
        """Reset factory state (for testing)."""
        cls._rob_idx_counter = 0

    @classmethod
    def create_aw(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        addr: int,
        axi_id: int = 0,
        length: int = 0,
        size: int = 5,  # 32 bytes
        burst: int = 1,  # INCR
        rob_idx: Optional[int] = None,
        rob_req: bool = False,
        last: bool = False,  # AW is followed by W
    ) -> Flit:
        """Create an AW (Write Address) flit."""
        if rob_idx is None:
            rob_idx = cls._next_rob_idx() if rob_req else 0
        return Flit(
            hdr=FlitHeader(
                rob_req=rob_req,
                rob_idx=rob_idx,
                dst_id=encode_node_id(dest),
                src_id=encode_node_id(src),
                last=last,
                axi_ch=AxiChannel.AW,
            ),
            payload=AxiAwPayload(
                addr=addr,
                axi_id=axi_id,
                length=length,
                size=size,
                burst=burst,
            ),
        )

    @classmethod
    def create_w(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        data: bytes,
        strb: int = 0xFFFFFFFF,
        last: bool = True,
        rob_idx: int = 0,
        seq_num: int = 0,
    ) -> Flit:
        """Create a W (Write Data) flit."""
        # Pad or truncate data to 32 bytes
        if len(data) < 32:
            data = data + bytes(32 - len(data))
        elif len(data) > 32:
            data = data[:32]

        return Flit(
            hdr=FlitHeader(
                rob_req=False,  # W doesn't need rob_req
                rob_idx=rob_idx,
                dst_id=encode_node_id(dest),
                src_id=encode_node_id(src),
                last=last,
                axi_ch=AxiChannel.W,
            ),
            payload=AxiWPayload(data=data, strb=strb),
            _seq_num=seq_num,
        )

    @classmethod
    def create_ar(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        addr: int,
        axi_id: int = 0,
        length: int = 0,
        size: int = 5,  # 32 bytes
        burst: int = 1,  # INCR
        rob_idx: Optional[int] = None,
        rob_req: bool = True,  # AR needs RoB for response matching
    ) -> Flit:
        """Create an AR (Read Address) flit."""
        if rob_idx is None:
            rob_idx = cls._next_rob_idx() if rob_req else 0
        return Flit(
            hdr=FlitHeader(
                rob_req=rob_req,
                rob_idx=rob_idx,
                dst_id=encode_node_id(dest),
                src_id=encode_node_id(src),
                last=True,  # AR is always single flit
                axi_ch=AxiChannel.AR,
            ),
            payload=AxiArPayload(
                addr=addr,
                axi_id=axi_id,
                length=length,
                size=size,
                burst=burst,
            ),
        )

    @classmethod
    def create_b(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        axi_id: int = 0,
        resp: int = 0,  # OKAY
        rob_idx: int = 0,
        rob_req: bool = False,
    ) -> Flit:
        """Create a B (Write Response) flit."""
        return Flit(
            hdr=FlitHeader(
                rob_req=rob_req,
                rob_idx=rob_idx,
                dst_id=encode_node_id(dest),
                src_id=encode_node_id(src),
                last=True,  # B is always single flit
                axi_ch=AxiChannel.B,
            ),
            payload=AxiBPayload(axi_id=axi_id, resp=resp),
        )

    @classmethod
    def create_r(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        data: bytes,
        axi_id: int = 0,
        resp: int = 0,  # OKAY
        last: bool = True,
        rob_idx: int = 0,
        rob_req: bool = False,
        seq_num: int = 0,
    ) -> Flit:
        """Create an R (Read Response) flit."""
        # Pad or truncate data to 32 bytes
        if len(data) < 32:
            data = data + bytes(32 - len(data))
        elif len(data) > 32:
            data = data[:32]

        return Flit(
            hdr=FlitHeader(
                rob_req=rob_req,
                rob_idx=rob_idx,
                dst_id=encode_node_id(dest),
                src_id=encode_node_id(src),
                last=last,
                axi_ch=AxiChannel.R,
            ),
            payload=AxiRPayload(data=data, axi_id=axi_id, resp=resp),
            _seq_num=seq_num,
        )

def create_response_flit(request_flit: Flit, payload: FlitPayload = None) -> Flit:
    """
    Create a response flit from a request flit.

    Swaps src and dest, converts channel type to response.

    Args:
        request_flit: Original request flit
        payload: Response payload (B or R)

    Returns:
        Response flit with swapped addresses
    """
    req_hdr = request_flit.hdr

    # Determine response channel
    if req_hdr.axi_ch in (AxiChannel.AW, AxiChannel.W):
        rsp_ch = AxiChannel.B
    else:  # AR
        rsp_ch = AxiChannel.R

    return Flit(
        hdr=FlitHeader(
            rob_req=req_hdr.rob_req,
            rob_idx=req_hdr.rob_idx,
            dst_id=req_hdr.src_id,  # Response goes to request source
            src_id=req_hdr.dst_id,  # Response comes from request dest
            last=True,  # Single response for now
            axi_ch=rsp_ch,
        ),
        payload=payload,
    )
