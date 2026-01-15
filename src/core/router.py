"""
Router implementation with Req/Resp physical separation.

Implements XY routing with Virtual Cut-Through switching
and credit-based flow control.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum, auto

from .flit import Flit, AxiChannel, decode_node_id
from .buffer import FlitBuffer, CreditFlowControl, PortBuffer, Link


class Direction(Enum):
    """Router port directions."""
    NORTH = auto()
    EAST = auto()
    SOUTH = auto()
    WEST = auto()
    LOCAL = auto()

    def opposite(self) -> Direction:
        """Get the opposite direction."""
        opposites = {
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST,
            Direction.WEST: Direction.EAST,
            Direction.LOCAL: Direction.LOCAL,
        }
        return opposites[self]

    def to_delta(self) -> Tuple[int, int]:
        """Get coordinate delta for this direction."""
        deltas = {
            Direction.NORTH: (0, 1),
            Direction.SOUTH: (0, -1),
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
            Direction.LOCAL: (0, 0),
        }
        return deltas[self]


class ChannelMode(Enum):
    """
    NoC Physical Channel Mode.

    GENERAL: 2 Sub-Routers (ReqRouter + RespRouter)
             - AW/W/AR share Request channel
             - B/R share Response channel
             - Has Head-of-Line blocking

    AXI: 5 Sub-Routers (AW + W + AR + B + R)
         - Each AXI channel has independent Sub-Router
         - No Head-of-Line blocking
         - Requires 10 wires per direction (5 channels × in/out)
    """
    GENERAL = auto()  # 2 Sub-Routers (default, backward compatible)
    AXI = auto()      # 5 Sub-Routers (independent channels)


@dataclass
class PipelineConfig:
    """
    Router Pipeline Stage Configuration.
    
    Configures the latency of each router pipeline stage:
    - RC (Route Computation): Determine output port
    - VA (Virtual Channel Allocation): Allocate VC (simplified in this model)
    - SA (Switch Allocation): Arbitrate for crossbar
    - ST (Switch Traversal): Transfer flit through crossbar
    
    Set latency to 0 to disable/bypass a stage.
    """
    # Stage latencies (in cycles)
    rc_latency: int = 1    # Route Computation
    va_latency: int = 0    # Virtual Channel Allocation (0 = disabled)
    sa_latency: int = 0    # Switch Allocation (0 = disabled)
    st_latency: int = 0    # Switch Traversal (0 = disabled)
    
    @property
    def total_latency(self) -> int:
        """Total pipeline depth in cycles."""
        return self.rc_latency + self.va_latency + self.sa_latency + self.st_latency
    
    @property
    def enabled_stages(self) -> int:
        """Number of enabled stages."""
        count = 0
        if self.rc_latency > 0:
            count += 1
        if self.va_latency > 0:
            count += 1
        if self.sa_latency > 0:
            count += 1
        if self.st_latency > 0:
            count += 1
        return count
    
    @classmethod
    def fast(cls) -> "PipelineConfig":
        """Fast mode: 1-cycle single-stage (default, backward compatible)."""
        return cls(rc_latency=1, va_latency=0, sa_latency=0, st_latency=0)
    
    @classmethod
    def standard(cls) -> "PipelineConfig":
        """Standard mode: 2-stage pipeline (RC + SA)."""
        return cls(rc_latency=1, va_latency=0, sa_latency=1, st_latency=0)
    
    @classmethod
    def hardware(cls) -> "PipelineConfig":
        """Hardware mode: 4-stage pipeline matching typical NoC routers."""
        return cls(rc_latency=1, va_latency=1, sa_latency=1, st_latency=1)


@dataclass
class RouterConfig:
    """Router configuration parameters."""
    buffer_depth: int = 4           # Input buffer depth per port (flits)
    output_buffer_depth: int = 0    # Output buffer depth (0 = no output buffer)
    flit_width: int = 64            # Flit width in bits
    routing_algorithm: str = "XY"   # Routing algorithm: "XY" only (Y→X disabled)
    arbitration: str = "wormhole"   # Arbitration: "wormhole" (packet locking)
    switching: str = "wormhole"     # Switching: "wormhole" (default)
    pipeline: PipelineConfig = None # Pipeline configuration
    channel_mode: ChannelMode = ChannelMode.GENERAL  # Physical channel mode

    def __post_init__(self):
        if self.pipeline is None:
            self.pipeline = PipelineConfig.fast()


@dataclass
class RouterStats:
    """Router statistics for performance analysis."""
    flits_received: int = 0
    flits_forwarded: int = 0
    flits_dropped: int = 0
    total_latency: int = 0  # Sum of latencies
    arbitration_cycles: int = 0
    buffer_full_events: int = 0

    # Per-port statistics
    port_utilization: Dict[Direction, int] = field(default_factory=dict)

    def __post_init__(self):
        for d in Direction:
            self.port_utilization[d] = 0

    @property
    def avg_latency(self) -> float:
        """Average flit latency through router."""
        if self.flits_forwarded == 0:
            return 0.0
        return self.total_latency / self.flits_forwarded


class WormholeArbiter:
    """
    Wormhole Arbiter - locks input→output path during packet transfer.

    Mechanism:
    1. When HEAD flit wins arbitration, lock the input→output path
    2. Subsequent BODY/TAIL flits bypass arbitration and follow locked path
    3. When TAIL flit is transmitted, release the lock

    This ensures packet integrity (no flit interleaving) and improves
    performance by avoiding re-arbitration for each flit.
    """

    def __init__(self):
        """Initialize Wormhole Arbiter."""
        # Track which output is locked by which input
        # output_lock[output_dir] = input_dir that holds the lock (or None)
        self._output_lock: Dict[Direction, Optional[Direction]] = {
            d: None for d in Direction
        }

        # Track which output each input is locked to
        # input_lock[input_dir] = output_dir that input is sending to (or None)
        self._input_lock: Dict[Direction, Optional[Direction]] = {
            d: None for d in Direction
        }

        # Round-robin state for new requests
        self._rr_priority: List[Direction] = list(Direction)
        self._rr_index = 0

    def is_output_locked(self, output_dir: Direction) -> bool:
        """Check if output port is currently locked."""
        return self._output_lock[output_dir] is not None

    def is_input_locked(self, input_dir: Direction) -> bool:
        """Check if input port has an active lock."""
        return self._input_lock[input_dir] is not None

    def get_locked_output(self, input_dir: Direction) -> Optional[Direction]:
        """Get the output port that input is locked to."""
        return self._input_lock[input_dir]

    def get_lock_holder(self, output_dir: Direction) -> Optional[Direction]:
        """Get the input port that holds lock on output."""
        return self._output_lock[output_dir]

    def lock(self, input_dir: Direction, output_dir: Direction) -> bool:
        """
        Lock input→output path.

        Args:
            input_dir: Input port direction.
            output_dir: Output port direction.

        Returns:
            True if lock acquired, False if output already locked by another.
        """
        # Check if output is already locked by another input
        if self._output_lock[output_dir] is not None:
            if self._output_lock[output_dir] != input_dir:
                return False  # Another input holds the lock

        # Check if input already has a different lock
        if self._input_lock[input_dir] is not None:
            if self._input_lock[input_dir] != output_dir:
                # Input trying to lock different output - should not happen
                return False

        # Acquire lock
        self._output_lock[output_dir] = input_dir
        self._input_lock[input_dir] = output_dir
        return True

    def release(self, input_dir: Direction) -> None:
        """
        Release lock held by input port.

        Called when TAIL flit is transmitted.

        Args:
            input_dir: Input port direction.
        """
        output_dir = self._input_lock[input_dir]
        if output_dir is not None:
            self._output_lock[output_dir] = None
            self._input_lock[input_dir] = None

    def arbitrate(
        self,
        requests: Dict[Direction, Tuple[Flit, Direction]]
    ) -> List[Tuple[Direction, Direction, Flit]]:
        """
        Arbitrate among input requests, respecting existing locks.

        Processing order:
        1. First, honor all locked paths (no arbitration needed)
        2. Then, use round-robin for new requests on free outputs

        Args:
            requests: Dict of input_dir → (flit, desired_output_dir)

        Returns:
            List of (input_dir, output_dir, flit) grants.
        """
        grants: List[Tuple[Direction, Direction, Flit]] = []
        outputs_granted: Dict[Direction, bool] = {d: False for d in Direction}

        # Phase 1: Honor locked paths first
        for input_dir, (flit, desired_output) in requests.items():
            if self.is_input_locked(input_dir):
                locked_output = self.get_locked_output(input_dir)
                if locked_output == desired_output:
                    # Locked path matches desired - grant immediately
                    grants.append((input_dir, desired_output, flit))
                    outputs_granted[desired_output] = True
                # If locked to different output, something is wrong in routing

        # Phase 2: Round-robin for new requests
        rr_order = self._get_rr_order()

        for input_dir in rr_order:
            if input_dir not in requests:
                continue
            if self.is_input_locked(input_dir):
                continue  # Already handled in phase 1

            flit, desired_output = requests[input_dir]

            # Check if output is available
            if outputs_granted[desired_output]:
                continue  # Output already used this cycle
            if self.is_output_locked(desired_output):
                continue  # Output locked by another input

            # Grant this request
            grants.append((input_dir, desired_output, flit))
            outputs_granted[desired_output] = True

        # Advance round-robin if any new grants were made
        new_grants = len(grants) - sum(
            1 for inp, _, _ in grants if self.is_input_locked(inp)
        )
        if new_grants > 0:
            self._advance_rr()

        return grants

    def _get_rr_order(self) -> List[Direction]:
        """Get directions in round-robin order."""
        result = []
        for i in range(len(self._rr_priority)):
            idx = (self._rr_index + i) % len(self._rr_priority)
            result.append(self._rr_priority[idx])
        return result

    def _advance_rr(self) -> None:
        """Advance round-robin pointer."""
        self._rr_index = (self._rr_index + 1) % len(self._rr_priority)

    def reset(self) -> None:
        """Reset all locks (for testing/debug)."""
        for d in Direction:
            self._output_lock[d] = None
            self._input_lock[d] = None

    def get_lock_status(self) -> Dict[str, Dict[Direction, Optional[Direction]]]:
        """Get current lock status (for debugging)."""
        return {
            "output_lock": dict(self._output_lock),
            "input_lock": dict(self._input_lock)
        }


class RouterPort:
    """
    Router port with valid/ready handshake interface.

    Each port has:
    - Internal buffer: stores incoming flits
    - Ingress interface: in_valid/in_flit (from upstream), out_ready (to upstream)
    - Egress interface: out_valid/out_flit (to downstream), in_ready (from downstream)

    The internal credit mechanism is preserved for backpressure tracking,
    but external communication uses valid/ready handshake.
    """

    def __init__(
        self,
        direction: Direction,
        buffer_depth: int,
        output_buffer_depth: int = 0,
        name: str = ""
    ):
        """
        Initialize router port.

        Args:
            direction: Port direction (N/E/S/W/L).
            buffer_depth: Input buffer depth.
            output_buffer_depth: Output buffer depth (0 = no output buffer).
            name: Port name for identification.
        """
        self.direction = direction
        self.name = name or f"Port_{direction.name}"
        self._buffer = FlitBuffer(buffer_depth, f"{self.name}_in")
        self._buffer_depth = buffer_depth

        # Output buffer (optional)
        self._output_buffer_depth = output_buffer_depth
        if output_buffer_depth > 0:
            self._output_buffer: Optional[FlitBuffer] = FlitBuffer(
                output_buffer_depth, f"{self.name}_out"
            )
        else:
            self._output_buffer = None

        # Internal credit tracking (for backpressure awareness)
        # Credits are based on downstream buffer depth
        self._output_credit = CreditFlowControl(initial_credits=buffer_depth)

        # --- Ingress Interface (receiving from upstream) ---
        # These signals are SET BY upstream (or PortWire propagation)
        self.in_valid: bool = False
        self.in_flit: Optional[Flit] = None
        # This signal is SET BY this port (reflects buffer availability)
        self.out_ready: bool = True

        # --- Egress Interface (sending to downstream) ---
        # These signals are SET BY this port
        self.out_valid: bool = False
        self.out_flit: Optional[Flit] = None
        # This signal is SET BY downstream (or PortWire propagation)
        self.in_ready: bool = False

        # Track if we consumed a flit this cycle (for credit release)
        self._consumed_this_cycle: bool = False

        # Legacy neighbor pointer (for backward compatibility with Selector)
        # Will be removed when Selector is updated to use valid/ready interface
        self.neighbor: Optional[RouterPort] = None

    # =========================================================================
    # Signal Update Methods (called during cycle processing)
    # =========================================================================

    def update_ready(self) -> None:
        """
        Update out_ready signal based on buffer availability.

        Called at the beginning of each cycle.
        """
        self.out_ready = not self._buffer.is_full()

    def sample_input(self) -> bool:
        """
        Sample input and perform handshake if valid && ready.

        Called after update_ready and signal propagation.

        Returns:
            True if a flit was successfully received.
        """
        if self.in_valid and self.out_ready and self.in_flit is not None:
            # Handshake successful - receive the flit
            success = self._buffer.push(self.in_flit)
            if success:
                return True
        return False

    def clear_input_signals(self) -> None:
        """Clear input signals after sampling."""
        self.in_valid = False
        self.in_flit = None

    def clear_output_if_accepted(self) -> bool:
        """
        Clear output signals if downstream accepted (handshake occurred).

        Called at end of cycle after downstream has sampled.
        If output buffer is enabled, pops the accepted flit from buffer.

        Returns:
            True if output was accepted and cleared.
        """
        if self.out_valid and self.in_ready:
            # Handshake successful - downstream accepted
            self._output_credit.consume(1)

            if self._output_buffer is not None:
                # Pop accepted flit from output buffer
                self._output_buffer.pop()

            self.out_valid = False
            self.out_flit = None
            return True
        return False

    def release_credit(self) -> None:
        """
        Release one credit (called when downstream consumed a flit).

        This is called by the upstream port after it detects that
        this port has consumed a flit from its buffer.
        """
        self._output_credit.release(1)

    # =========================================================================
    # Buffer Access Methods
    # =========================================================================

    def peek(self) -> Optional[Flit]:
        """Peek at the front flit without removing."""
        return self._buffer.peek()

    def pop_for_routing(self) -> Optional[Flit]:
        """
        Pop flit from buffer for internal routing.

        Note: This does NOT directly trigger credit release.
        Credit release is handled via signal propagation.

        Returns:
            The popped flit, or None if buffer empty.
        """
        flit = self._buffer.pop()
        if flit is not None:
            self._consumed_this_cycle = True
        return flit

    def set_output(self, flit: Flit) -> bool:
        """
        Set output valid/flit for sending to downstream.

        If output buffer is enabled, pushes to output buffer.
        Otherwise, directly sets out_valid/out_flit.

        Args:
            flit: Flit to send.

        Returns:
            True if output was set/buffered, False if blocked.
        """
        if self._output_buffer is not None:
            # Output buffer mode: push to buffer
            if self._output_buffer.is_full():
                return False
            if not self._output_credit.can_send(1):
                return False
            return self._output_buffer.push(flit)
        else:
            # Direct mode: set out_valid/out_flit
            if self.out_valid:
                # Already have pending output
                return False
            if not self._output_credit.can_send(1):
                # No credits available
                return False
            self.out_valid = True
            self.out_flit = flit
            return True

    def can_send(self) -> bool:
        """Check if we can accept a flit for output (has credits and space)."""
        if not self._output_credit.can_send(1):
            return False
        if self._output_buffer is not None:
            return not self._output_buffer.is_full()
        else:
            return not self.out_valid

    def has_pending_output(self) -> bool:
        """Check if there's a pending output waiting for acceptance."""
        if self._output_buffer is not None:
            return not self._output_buffer.is_empty()
        return self.out_valid

    def update_output_from_buffer(self) -> None:
        """
        Update out_valid/out_flit from output buffer.

        Called at the beginning of each cycle when output buffer is enabled.
        If there's a flit in the output buffer and no current out_valid,
        peek the flit and set out_valid/out_flit.
        """
        if self._output_buffer is None:
            return

        if not self.out_valid and not self._output_buffer.is_empty():
            flit = self._output_buffer.peek()
            if flit is not None:
                self.out_valid = True
                self.out_flit = flit

    def check_and_clear_consumed(self) -> bool:
        """
        Check if flit was consumed this cycle and clear flag.

        Returns:
            True if a flit was consumed this cycle.
        """
        consumed = self._consumed_this_cycle
        self._consumed_this_cycle = False
        return consumed

    # =========================================================================
    # Legacy and Test Helper Methods
    # =========================================================================

    def can_accept(self) -> bool:
        """Check if port can accept incoming flit (for test injection)."""
        return not self._buffer.is_full()

    def receive(self, flit: Flit) -> bool:
        """
        Receive a flit directly into buffer (for test injection).

        Prefer using valid/ready interface for simulation.
        """
        if not self.can_accept():
            return False
        return self._buffer.push(flit)



    def pop(self) -> Optional[Flit]:
        """
        Legacy pop with automatic credit release (for Selector compatibility).

        Prefer using pop_for_routing() for new code.
        """
        flit = self._buffer.pop()
        if flit is not None:
            self._consumed_this_cycle = True
            if self.neighbor is not None:
                # Release credit to upstream (legacy pattern)
                self.neighbor.output_credit.release(1)
        return flit


    @property
    def input_buffer(self) -> FlitBuffer:
        """Access to internal buffer (for compatibility)."""
        return self._buffer

    @property
    def output_credit(self) -> CreditFlowControl:
        """Access to output credit (for compatibility)."""
        return self._output_credit

    @property
    def occupancy(self) -> int:
        """Current buffer occupancy."""
        return self._buffer.occupancy

    @property
    def credits_available(self) -> int:
        """Available output credits."""
        return self._output_credit.credits

    def __repr__(self) -> str:
        return (
            f"RouterPort({self.name}, "
            f"buf={self.occupancy}/{self._buffer_depth}, "
            f"credit={self.credits_available}, "
            f"in={self.in_valid}, out={self.out_valid})"
        )


class PortWire:
    """
    Bidirectional wire connecting two RouterPorts.

    Propagates valid/ready/flit signals between connected ports:
    - A.out_valid/out_flit → B.in_valid/in_flit
    - B.out_ready → A.in_ready
    - B.out_valid/out_flit → A.in_valid/in_flit
    - A.out_ready → B.in_ready
    """

    def __init__(self, port_a: RouterPort, port_b: RouterPort):
        """
        Create a bidirectional wire between two ports.

        Args:
            port_a: First port.
            port_b: Second port.
        """
        self.port_a = port_a
        self.port_b = port_b

    def propagate_signals(self) -> None:
        """
        Propagate signals between connected ports.

        This should be called after ports update their own signals
        and before they sample inputs.
        """
        # A → B direction
        self.port_b.in_valid = self.port_a.out_valid
        self.port_b.in_flit = self.port_a.out_flit
        self.port_a.in_ready = self.port_b.out_ready

        # B → A direction
        self.port_a.in_valid = self.port_b.out_valid
        self.port_a.in_flit = self.port_b.out_flit
        self.port_b.in_ready = self.port_a.out_ready

    def propagate_credit_release(self) -> None:
        """
        Handle credit release based on consumption flags.

        Called after ports have processed flits.
        """
        # If A consumed a flit from its buffer, B should release credit
        if self.port_a.check_and_clear_consumed():
            self.port_b.release_credit()

        # If B consumed a flit from its buffer, A should release credit
        if self.port_b.check_and_clear_consumed():
            self.port_a.release_credit()

    def __repr__(self) -> str:
        return f"PortWire({self.port_a.name} <-> {self.port_b.name})"


class XYRouter:
    """
    XY Routing Router implementation.

    Implements deterministic XY routing:
    1. First route along X axis (EAST/WEST)
    2. Then route along Y axis (NORTH/SOUTH)

    Uses Virtual Cut-Through (VCT) switching - entire packet
    must be buffered before forwarding.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None,
        name: str = ""
    ):
        """
        Initialize router.

        Args:
            coord: Router coordinate (x, y).
            config: Router configuration.
            name: Router name for identification.
        """
        self.coord = coord
        self.config = config or RouterConfig()
        self.name = name or f"Router({coord[0]},{coord[1]})"

        # Create ports
        self.ports: Dict[Direction, RouterPort] = {}
        for d in Direction:
            self.ports[d] = RouterPort(
                direction=d,
                buffer_depth=self.config.buffer_depth,
                output_buffer_depth=self.config.output_buffer_depth,
                name=f"{self.name}_{d.name}"
            )

        # Statistics
        self.stats = RouterStats()

        # Wormhole Arbiter for packet-level locking
        self._arbiter = WormholeArbiter()

        # Packet tracking for debugging
        # Maps (src_id, dst_id, rob_idx) -> (input_port, flit_count_received, last_seq)
        self._packet_state: Dict[Tuple[int, int, int], Tuple[Direction, int, int]] = {}
        
        # === Pipeline Registers ===
        # Each pipeline stage holds flits in transit with remaining cycles
        # Entry format: (flit, input_dir, output_dir, remaining_cycles)
        self._pipeline_stages: List[List[Tuple[Flit, Direction, Direction, int]]] = []
        
        # Calculate total pipeline depth
        pipeline_cfg = self.config.pipeline
        self._stage_config = [
            ("RC", pipeline_cfg.rc_latency),
            ("VA", pipeline_cfg.va_latency),
            ("SA", pipeline_cfg.sa_latency),
            ("ST", pipeline_cfg.st_latency),
        ]
        
        # Only include enabled stages
        self._active_stages = [(name, lat) for name, lat in self._stage_config if lat > 0]
        
        # Initialize pipeline stage buffers (one list per stage)
        for _ in self._active_stages:
            self._pipeline_stages.append([])

    def get_port(self, direction: Direction) -> RouterPort:
        """Get port by direction."""
        return self.ports[direction]


    def compute_output_port(
        self,
        flit: Flit,
        input_dir: Optional[Direction] = None
    ) -> Optional[Direction]:
        """
        Compute output port using XY routing with Y→X turn prevention.

        XY Routing Rules:
        1. First route along X axis (EAST/WEST)
        2. Then route along Y axis (NORTH/SOUTH)
        3. Once on Y axis, CANNOT turn back to X axis (prevents deadlock)

        Prohibited turns (Y→X):
        - NORTH → EAST/WEST
        - SOUTH → EAST/WEST

        Args:
            flit: Flit to route.
            input_dir: Input direction (for Y→X prevention check).

        Returns:
            Output direction, or None if at destination or illegal turn.
        """
        dest = flit.dest
        curr = self.coord

        # At destination?
        if dest == curr:
            return Direction.LOCAL

        # Determine desired output based on XY routing
        desired_output: Optional[Direction] = None

        # XY Routing: X first, then Y
        if dest[0] > curr[0]:
            desired_output = Direction.EAST
        elif dest[0] < curr[0]:
            desired_output = Direction.WEST
        elif dest[1] > curr[1]:
            desired_output = Direction.NORTH
        elif dest[1] < curr[1]:
            desired_output = Direction.SOUTH

        if desired_output is None:
            return None

        # Y→X Turn Prevention:
        # If flit came from Y direction (NORTH/SOUTH), cannot go to X direction (EAST/WEST)
        if input_dir in (Direction.NORTH, Direction.SOUTH):
            if desired_output in (Direction.EAST, Direction.WEST):
                # Illegal Y→X turn detected!
                # This should not happen in correctly routed packets.
                # Log warning and return None (will be dropped)
                return None

        return desired_output

    # =========================================================================
    # Phased Cycle Methods (for wire-based processing)
    # =========================================================================

    def update_all_ready(self) -> None:
        """
        Phase 1: Update ready signals for all ports.

        Called at the beginning of each cycle before signal propagation.
        Also updates output valid signals from output buffers.
        """
        for port in self.ports.values():
            port.update_ready()
            port.update_output_from_buffer()

    def sample_all_inputs(self) -> int:
        """
        Phase 2: Sample inputs and perform handshakes.

        Called after wire signal propagation.

        Returns:
            Number of flits received.
        """
        received = 0
        for port in self.ports.values():
            if port.sample_input():
                received += 1
                self.stats.flits_received += 1
        return received

    def route_and_forward(self, current_time: int = 0) -> List[Tuple[Flit, Direction]]:
        """
        Phase 3: Route flits from input buffers to output ports.

        Supports configurable pipeline latency:
        - Fast mode (1 cycle): Direct input→output in same cycle
        - Multi-stage mode: Flits traverse pipeline stages with configurable latency

        Uses Wormhole Arbiter for packet-level locking:
        1. Process pipeline stages (advance flits, emit completed ones)
        2. Collect routing requests from input ports
        3. Arbiter grants requests
        4. Enter granted flits into first pipeline stage

        Args:
            current_time: Current simulation time.

        Returns:
            List of (flit, output_direction) pairs for forwarded flits.
        """
        forwarded: List[Tuple[Flit, Direction]] = []
        
        # === Step 1: Process pipeline stages (advance and emit) ===
        if len(self._active_stages) > 1:
            # Multi-stage pipeline: advance flits through stages
            forwarded = self._advance_pipeline(current_time)
        
        # === Step 2: Collect routing requests from input ports ===
        requests: Dict[Direction, Tuple[Flit, Direction]] = {}

        for in_dir in Direction:
            in_port = self.ports[in_dir]

            # Check if port has flit to forward
            flit = in_port.peek()
            if flit is None:
                continue

            # Compute output port
            out_dir = self.compute_output_port(flit, in_dir)

            if out_dir is None:
                # Routing failed (illegal Y→X turn or at destination error)
                in_port.pop_for_routing()
                self.stats.flits_dropped += 1
                continue

            # Check output availability (for fast mode, check can_send)
            # For pipeline mode, check if pipeline can accept
            if len(self._active_stages) <= 1:
                # Fast mode: check output port directly
                out_port = self.ports[out_dir]
                if not out_port.can_send():
                    self.stats.buffer_full_events += 1
                    continue
            else:
                # Pipeline mode: check if first stage has capacity
                # (simplified: allow up to buffer_depth entries per stage)
                if len(self._pipeline_stages[0]) >= self.config.buffer_depth:
                    self.stats.buffer_full_events += 1
                    continue

            requests[in_dir] = (flit, out_dir)

        # === Step 3: Arbiter grants requests ===
        grants = self._arbiter.arbitrate(requests)

        # === Step 4: Execute grants ===
        for in_dir, out_dir, flit in grants:
            in_port = self.ports[in_dir]

            # Pop from input buffer
            popped_flit = in_port.pop_for_routing()
            if popped_flit is None:
                continue

            # Handle Wormhole locking
            # In AXI mode, each channel is independent - bypass wormhole locking
            # because there's no mux/demux and each flit routes independently
            if self.config.channel_mode != ChannelMode.AXI:
                if popped_flit.is_head() and not popped_flit.is_single_flit():
                    self._arbiter.lock(in_dir, out_dir)

                if popped_flit.is_tail() or popped_flit.is_single_flit():
                    self._arbiter.release(in_dir)

            # Enter into pipeline or emit directly
            if len(self._active_stages) <= 1:
                # Fast mode: directly set output
                out_port = self.ports[out_dir]
                if out_port.set_output(popped_flit):
                    forwarded.append((popped_flit, out_dir))
                    self.stats.flits_forwarded += 1
                    self.stats.port_utilization[out_dir] += 1
                    self._update_packet_state_on_forward(popped_flit, in_dir)
                else:
                    self.stats.flits_dropped += 1
            else:
                # Pipeline mode: enter first stage with full latency
                first_stage_latency = self._active_stages[0][1]
                self._pipeline_stages[0].append(
                    (popped_flit, in_dir, out_dir, first_stage_latency)
                )
                self._update_packet_state_on_forward(popped_flit, in_dir)

        if grants:
            self.stats.arbitration_cycles += 1

        return forwarded

    def _advance_pipeline(self, current_time: int = 0) -> List[Tuple[Flit, Direction]]:
        """
        Advance flits through pipeline stages.
        
        Each cycle:
        1. Decrement remaining cycles for flits in each stage
        2. Move completed flits to next stage (or output if last stage)
        
        Returns:
            List of (flit, output_direction) for flits completing pipeline.
        """
        forwarded: List[Tuple[Flit, Direction]] = []
        num_stages = len(self._active_stages)
        
        if num_stages == 0:
            return forwarded
        
        # Process stages in reverse order (from last to first)
        # This allows flits to move forward without overwriting
        for stage_idx in range(num_stages - 1, -1, -1):
            stage_name, stage_latency = self._active_stages[stage_idx]
            new_stage_contents = []
            
            for flit, in_dir, out_dir, remaining in self._pipeline_stages[stage_idx]:
                remaining -= 1
                
                if remaining <= 0:
                    # Flit completed this stage
                    if stage_idx == num_stages - 1:
                        # Last stage: emit to output port
                        out_port = self.ports[out_dir]
                        if out_port.can_send() and out_port.set_output(flit):
                            forwarded.append((flit, out_dir))
                            self.stats.flits_forwarded += 1
                            self.stats.port_utilization[out_dir] += 1
                        else:
                            # Output blocked, stay in stage (stall at 0)
                            new_stage_contents.append((flit, in_dir, out_dir, 0))
                    else:
                        # Move to next stage
                        next_stage_latency = self._active_stages[stage_idx + 1][1]
                        self._pipeline_stages[stage_idx + 1].append(
                            (flit, in_dir, out_dir, next_stage_latency)
                        )
                else:
                    # Still processing in this stage
                    new_stage_contents.append((flit, in_dir, out_dir, remaining))
            
            self._pipeline_stages[stage_idx] = new_stage_contents
        
        return forwarded
    
    @property
    def pipeline_depth(self) -> int:
        """Get total pipeline depth (in cycles)."""
        return self.config.pipeline.total_latency
    
    @property
    def flits_in_pipeline(self) -> int:
        """Get number of flits currently in pipeline stages."""
        return sum(len(stage) for stage in self._pipeline_stages)

    def clear_accepted_outputs(self) -> int:
        """
        Phase 4: Clear outputs that were accepted by downstream.

        Called after wire signal propagation at end of cycle.

        Returns:
            Number of outputs cleared.
        """
        cleared = 0
        for port in self.ports.values():
            if port.clear_output_if_accepted():
                cleared += 1
        return cleared

    def clear_all_input_signals(self) -> None:
        """Clear input signals on all ports after sampling."""
        for port in self.ports.values():
            port.clear_input_signals()

    # =========================================================================
    # Legacy process_cycle (for backward compatibility)
    # =========================================================================

    def process_cycle(self, current_time: int = 0) -> List[Tuple[Flit, Direction]]:
        """
        Process one simulation cycle (legacy method).

        This method provides backward compatibility. For wire-based processing,
        use the phased methods: update_all_ready, sample_all_inputs,
        route_and_forward, clear_accepted_outputs.

        Args:
            current_time: Current simulation time.

        Returns:
            List of (flit, output_direction) pairs for forwarded flits.
        """
        # Simply delegate to route_and_forward
        # Note: In wire-based mode, the mesh orchestrates the full cycle
        return self.route_and_forward(current_time)

    def _update_packet_state_on_forward(self, flit: Flit, in_dir: Direction) -> None:
        """
        Update packet tracking state after forwarding a flit.

        Uses (src_id, dst_id, rob_idx) as packet key for FlooNoC format.
        """
        # Create packet key from header fields
        hdr = flit.hdr
        packet_key = (hdr.src_id, hdr.dst_id, hdr.rob_idx)

        if flit.is_head():
            # New packet
            self._packet_state[packet_key] = (in_dir, 1, flit._seq_num)
        elif packet_key in self._packet_state:
            # Update count
            in_d, count, _ = self._packet_state[packet_key]
            self._packet_state[packet_key] = (in_d, count + 1, flit._seq_num)

            if flit.is_tail():
                # Packet complete, clean up
                del self._packet_state[packet_key]

    def receive_flit(self, direction: Direction, flit: Flit) -> bool:
        """
        Receive a flit from an external source (for testing/injection).

        Args:
            direction: Input port direction.
            flit: Flit to receive.

        Returns:
            True if accepted.
        """
        success = self.ports[direction].receive(flit)
        if success:
            self.stats.flits_received += 1
        return success

    def sample_stats(self) -> None:
        """Sample current statistics."""
        for port in self.ports.values():
            port.input_buffer.sample_stats()

    def __repr__(self) -> str:
        occupancies = {d.name[0]: p.occupancy for d, p in self.ports.items()}
        return f"XYRouter{self.coord} {occupancies}"


class ReqRouter(XYRouter):
    """
    Request Router - handles request traffic.

    Inherits XY routing behavior.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None
    ):
        super().__init__(coord, config, f"ReqRouter({coord[0]},{coord[1]})")


class RespRouter(XYRouter):
    """
    Response Router - handles response traffic.

    Inherits XY routing behavior.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None
    ):
        super().__init__(coord, config, f"RespRouter({coord[0]},{coord[1]})")


class Router:
    """
    Combined Router with Req/Resp physical separation.

    Contains a ReqRouter and a RespRouter operating independently.
    This follows the FlooNoC-style architecture where request and
    response networks are physically separated.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None
    ):
        """
        Initialize combined router.

        Args:
            coord: Router coordinate (x, y).
            config: Router configuration.
        """
        self.coord = coord
        self.config = config or RouterConfig()

        # Create separate request and response routers
        self.req_router = ReqRouter(coord, self.config)
        self.resp_router = RespRouter(coord, self.config)

    def get_req_port(self, direction: Direction) -> RouterPort:
        """Get request router port."""
        return self.req_router.get_port(direction)

    def get_resp_port(self, direction: Direction) -> RouterPort:
        """Get response router port."""
        return self.resp_router.get_port(direction)

    def connect_req(self, direction: Direction, neighbor_port: RouterPort) -> None:
        """Connect request router port to neighbor (legacy method)."""
        self.req_router.connect(direction, neighbor_port)

    def connect_resp(self, direction: Direction, neighbor_port: RouterPort) -> None:
        """Connect response router port to neighbor (legacy method)."""
        self.resp_router.connect(direction, neighbor_port)

    # =========================================================================
    # Phased Cycle Methods (for wire-based processing)
    # =========================================================================

    def update_all_ready(self) -> None:
        """Phase 1: Update ready signals for both routers."""
        self.req_router.update_all_ready()
        self.resp_router.update_all_ready()

    def sample_all_inputs(self) -> Tuple[int, int]:
        """
        Phase 2: Sample inputs for both routers.

        Returns:
            Tuple of (req_received, resp_received) counts.
        """
        req_received = self.req_router.sample_all_inputs()
        resp_received = self.resp_router.sample_all_inputs()
        return req_received, resp_received

    def route_and_forward(self, current_time: int = 0) -> Tuple[
        List[Tuple[Flit, Direction]],
        List[Tuple[Flit, Direction]]
    ]:
        """
        Phase 3: Route and forward flits for both routers.

        Returns:
            Tuple of (req_forwarded, resp_forwarded) lists.
        """
        req_fwd = self.req_router.route_and_forward(current_time)
        resp_fwd = self.resp_router.route_and_forward(current_time)
        return req_fwd, resp_fwd

    def clear_accepted_outputs(self) -> Tuple[int, int]:
        """
        Phase 4: Clear accepted outputs for both routers.

        Returns:
            Tuple of (req_cleared, resp_cleared) counts.
        """
        req_cleared = self.req_router.clear_accepted_outputs()
        resp_cleared = self.resp_router.clear_accepted_outputs()
        return req_cleared, resp_cleared

    def clear_all_input_signals(self) -> None:
        """Clear input signals on both routers."""
        self.req_router.clear_all_input_signals()
        self.resp_router.clear_all_input_signals()

    # =========================================================================
    # Legacy Methods
    # =========================================================================

    def process_cycle(self, current_time: int = 0) -> Tuple[
        List[Tuple[Flit, Direction]],
        List[Tuple[Flit, Direction]]
    ]:
        """
        Process one simulation cycle for both routers.

        Args:
            current_time: Current simulation time.

        Returns:
            Tuple of (req_forwarded, resp_forwarded) lists.
        """
        req_fwd = self.req_router.process_cycle(current_time)
        resp_fwd = self.resp_router.process_cycle(current_time)
        return req_fwd, resp_fwd

    def receive_request(self, direction: Direction, flit: Flit) -> bool:
        """Receive request flit."""
        return self.req_router.receive_flit(direction, flit)

    def receive_response(self, direction: Direction, flit: Flit) -> bool:
        """Receive response flit."""
        return self.resp_router.receive_flit(direction, flit)

    def sample_stats(self) -> None:
        """Sample statistics for both routers."""
        self.req_router.sample_stats()
        self.resp_router.sample_stats()

    @property
    def total_req_occupancy(self) -> int:
        """Total request buffer occupancy."""
        return sum(p.occupancy for p in self.req_router.ports.values())

    @property
    def total_resp_occupancy(self) -> int:
        """Total response buffer occupancy."""
        return sum(p.occupancy for p in self.resp_router.ports.values())

    def __repr__(self) -> str:
        return (
            f"Router{self.coord}("
            f"req_occ={self.total_req_occupancy}, "
            f"resp_occ={self.total_resp_occupancy})"
        )


# =============================================================================
# Edge Router (Column 0 - no NI, only router)
# =============================================================================

class EdgeRouter(Router):
    """
    Edge Router for Column 0.

    Edge routers:
    - Have no NI (Local port used for Selector connection)
    - Must have N/S interconnection for response routing
    - Act as entry/exit points between Selector and mesh
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None
    ):
        """
        Initialize edge router.

        Args:
            coord: Router coordinate (should be (0, y)).
            config: Router configuration.
        """
        if coord[0] != 0:
            raise ValueError(f"Edge router must be in column 0, got {coord}")

        super().__init__(coord, config)

        # Edge routers have Local port connected to Selector
        self.selector_connected = False

    def connect_to_selector(
        self,
        req_port: RouterPort,
        resp_port: RouterPort
    ) -> None:
        """
        Connect Local ports to Selector.

        Args:
            req_port: Selector's request port.
            resp_port: Selector's response port.
        """
        self.req_router.connect(Direction.LOCAL, req_port)
        self.resp_router.connect(Direction.LOCAL, resp_port)
        self.selector_connected = True

    def clear_accepted_outputs(self) -> Tuple[int, int]:
        """
        Phase 4: Clear accepted outputs for both routers.

        EdgeRouter overrides this to SKIP clearing the LOCAL port on the
        response router. The LOCAL port is connected to Selector, and its
        handshake timing is managed by Selector's wire propagation cycle,
        not the mesh's internal cycle.

        Without this override, the following timing issue occurs:
        1. Selector propagates ready=True to EdgeRouter.resp.LOCAL.in_ready
        2. EdgeRouter routes and sets LOCAL.out_valid=True
        3. Mesh's clear_accepted_outputs() sees in_ready=True, clears output
        4. Selector never gets to sample the output!

        Returns:
            Tuple of (req_cleared, resp_cleared) counts.
        """
        req_cleared = self.req_router.clear_accepted_outputs()

        # Clear response outputs EXCEPT LOCAL port
        resp_cleared = 0
        for direction, port in self.resp_router.ports.items():
            if direction == Direction.LOCAL:
                # Skip LOCAL - managed by Selector's wire cycle
                continue
            if port.clear_output_if_accepted():
                resp_cleared += 1

        return req_cleared, resp_cleared

    def __repr__(self) -> str:
        sel = "↔Sel" if self.selector_connected else ""
        return f"EdgeRouter{self.coord}{sel}"


# =============================================================================
# Helper functions
# =============================================================================

def create_router(
    coord: Tuple[int, int],
    is_edge: bool = False,
    config: Optional[RouterConfig] = None
) -> "Router":
    """
    Factory function to create appropriate router type.

    Args:
        coord: Router coordinate.
        is_edge: True for edge router (column 0).
        config: Router configuration.

    Returns:
        Router, EdgeRouter, AXIModeRouter, or AXIModeEdgeRouter instance.
    """
    config = config or RouterConfig()

    if config.channel_mode == ChannelMode.AXI:
        if is_edge:
            return AXIModeEdgeRouter(coord, config)
        return AXIModeRouter(coord, config)
    else:
        if is_edge:
            return EdgeRouter(coord, config)
        return Router(coord, config)


# =============================================================================
# AXI Mode Channel-Specific Sub-Routers
# =============================================================================

class AWRouter(XYRouter):
    """
    AW (Write Address) Channel Router for AXI Mode.

    Handles only AW flits. Inherits all XY routing logic from XYRouter.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None,
        name: str = ""
    ):
        name = name or f"AWRouter({coord[0]},{coord[1]})"
        super().__init__(coord, config, name)
        self.axi_channel = AxiChannel.AW


class WRouter(XYRouter):
    """
    W (Write Data) Channel Router for AXI Mode.

    Handles only W flits. Inherits all XY routing logic from XYRouter.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None,
        name: str = ""
    ):
        name = name or f"WRouter({coord[0]},{coord[1]})"
        super().__init__(coord, config, name)
        self.axi_channel = AxiChannel.W


class ARRouter(XYRouter):
    """
    AR (Read Address) Channel Router for AXI Mode.

    Handles only AR flits. Inherits all XY routing logic from XYRouter.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None,
        name: str = ""
    ):
        name = name or f"ARRouter({coord[0]},{coord[1]})"
        super().__init__(coord, config, name)
        self.axi_channel = AxiChannel.AR


class BRouter(XYRouter):
    """
    B (Write Response) Channel Router for AXI Mode.

    Handles only B flits. Inherits all XY routing logic from XYRouter.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None,
        name: str = ""
    ):
        name = name or f"BRouter({coord[0]},{coord[1]})"
        super().__init__(coord, config, name)
        self.axi_channel = AxiChannel.B


class RRouter(XYRouter):
    """
    R (Read Data) Channel Router for AXI Mode.

    Handles only R flits. Inherits all XY routing logic from XYRouter.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None,
        name: str = ""
    ):
        name = name or f"RRouter({coord[0]},{coord[1]})"
        super().__init__(coord, config, name)
        self.axi_channel = AxiChannel.R


# =============================================================================
# AXI Mode Combined Router
# =============================================================================

class AXIModeRouter:
    """
    Combined Router with 5 independent AXI channel Sub-Routers.

    Contains: AWRouter, WRouter, ARRouter, BRouter, RRouter
    Each Sub-Router operates independently with its own PortWire connections.

    This eliminates Head-of-Line blocking by giving each AXI channel
    its own physical path through the network.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None
    ):
        """
        Initialize AXI Mode combined router.

        Args:
            coord: Router coordinate (x, y).
            config: Router configuration.
        """
        self.coord = coord
        self.config = config or RouterConfig(channel_mode=ChannelMode.AXI)

        # Create 5 channel-specific sub-routers
        self.aw_router = AWRouter(coord, self.config)
        self.w_router = WRouter(coord, self.config)
        self.ar_router = ARRouter(coord, self.config)
        self.b_router = BRouter(coord, self.config)
        self.r_router = RRouter(coord, self.config)

        # Channel to router mapping
        self._channel_routers: Dict[AxiChannel, XYRouter] = {
            AxiChannel.AW: self.aw_router,
            AxiChannel.W: self.w_router,
            AxiChannel.AR: self.ar_router,
            AxiChannel.B: self.b_router,
            AxiChannel.R: self.r_router,
        }

    def get_channel_router(self, channel: AxiChannel) -> XYRouter:
        """Get Sub-Router for specific AXI channel."""
        return self._channel_routers[channel]

    def get_channel_port(
        self,
        channel: AxiChannel,
        direction: Direction
    ) -> "RouterPort":
        """Get port for specific channel and direction."""
        return self._channel_routers[channel].get_port(direction)

    # Request channel accessors (for convenience)
    def get_aw_port(self, direction: Direction) -> "RouterPort":
        """Get AW channel port."""
        return self.aw_router.get_port(direction)

    def get_w_port(self, direction: Direction) -> "RouterPort":
        """Get W channel port."""
        return self.w_router.get_port(direction)

    def get_ar_port(self, direction: Direction) -> "RouterPort":
        """Get AR channel port."""
        return self.ar_router.get_port(direction)

    # Response channel accessors
    def get_b_port(self, direction: Direction) -> "RouterPort":
        """Get B channel port."""
        return self.b_router.get_port(direction)

    def get_r_port(self, direction: Direction) -> "RouterPort":
        """Get R channel port."""
        return self.r_router.get_port(direction)

    # =========================================================================
    # Phased Cycle Methods (for wire-based processing)
    # =========================================================================

    def update_all_ready(self) -> None:
        """Phase 1: Update ready signals for all 5 channel routers."""
        self.aw_router.update_all_ready()
        self.w_router.update_all_ready()
        self.ar_router.update_all_ready()
        self.b_router.update_all_ready()
        self.r_router.update_all_ready()

    def sample_all_inputs(self) -> Dict[AxiChannel, int]:
        """
        Phase 2: Sample inputs for all 5 channel routers.

        Returns:
            Dict mapping channel to received flit count.
        """
        return {
            AxiChannel.AW: self.aw_router.sample_all_inputs(),
            AxiChannel.W: self.w_router.sample_all_inputs(),
            AxiChannel.AR: self.ar_router.sample_all_inputs(),
            AxiChannel.B: self.b_router.sample_all_inputs(),
            AxiChannel.R: self.r_router.sample_all_inputs(),
        }

    def route_and_forward(
        self,
        current_time: int = 0
    ) -> Dict[AxiChannel, List[Tuple[Flit, Direction]]]:
        """
        Phase 3: Route and forward flits for all 5 channel routers.

        Returns:
            Dict mapping channel to list of (flit, direction) tuples.
        """
        return {
            AxiChannel.AW: self.aw_router.route_and_forward(current_time),
            AxiChannel.W: self.w_router.route_and_forward(current_time),
            AxiChannel.AR: self.ar_router.route_and_forward(current_time),
            AxiChannel.B: self.b_router.route_and_forward(current_time),
            AxiChannel.R: self.r_router.route_and_forward(current_time),
        }

    def clear_accepted_outputs(self) -> Dict[AxiChannel, int]:
        """
        Phase 4: Clear accepted outputs for all 5 channel routers.

        Returns:
            Dict mapping channel to cleared count.
        """
        return {
            AxiChannel.AW: self.aw_router.clear_accepted_outputs(),
            AxiChannel.W: self.w_router.clear_accepted_outputs(),
            AxiChannel.AR: self.ar_router.clear_accepted_outputs(),
            AxiChannel.B: self.b_router.clear_accepted_outputs(),
            AxiChannel.R: self.r_router.clear_accepted_outputs(),
        }

    def clear_all_input_signals(self) -> None:
        """Clear input signals on all 5 channel routers."""
        self.aw_router.clear_all_input_signals()
        self.w_router.clear_all_input_signals()
        self.ar_router.clear_all_input_signals()
        self.b_router.clear_all_input_signals()
        self.r_router.clear_all_input_signals()

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_total_flits_forwarded(self) -> int:
        """Get total flits forwarded across all channels."""
        return sum(
            router.stats.flits_forwarded
            for router in self._channel_routers.values()
        )

    def get_channel_stats(self) -> Dict[AxiChannel, RouterStats]:
        """Get statistics per channel."""
        return {
            channel: router.stats
            for channel, router in self._channel_routers.items()
        }

    def __repr__(self) -> str:
        return f"AXIModeRouter{self.coord}"


class AXIModeEdgeRouter(AXIModeRouter):
    """
    Edge Router for AXI Mode at Column 0.

    Similar to EdgeRouter but with 5 Sub-Routers instead of 2.
    Connects to RoutingSelector via LOCAL ports.
    """

    def __init__(
        self,
        coord: Tuple[int, int],
        config: Optional[RouterConfig] = None
    ):
        super().__init__(coord, config)
        self.selector_connected = False

    def connect_to_selector(
        self,
        aw_port: "RouterPort",
        w_port: "RouterPort",
        ar_port: "RouterPort",
        b_port: "RouterPort",
        r_port: "RouterPort"
    ) -> None:
        """
        Connect LOCAL ports to Selector for all 5 channels.

        Args:
            aw_port: Selector's AW port for this edge router.
            w_port: Selector's W port for this edge router.
            ar_port: Selector's AR port for this edge router.
            b_port: Selector's B port for this edge router.
            r_port: Selector's R port for this edge router.
        """
        self.aw_router.connect(Direction.LOCAL, aw_port)
        self.w_router.connect(Direction.LOCAL, w_port)
        self.ar_router.connect(Direction.LOCAL, ar_port)
        self.b_router.connect(Direction.LOCAL, b_port)
        self.r_router.connect(Direction.LOCAL, r_port)
        self.selector_connected = True

    def clear_accepted_outputs(self) -> Dict[AxiChannel, int]:
        """
        Phase 4: Clear accepted outputs, skipping LOCAL ports.

        EdgeRouter-specific: LOCAL port outputs are cleared by Selector's
        wire propagation, not during this phase.
        """
        result = {}

        for channel, router in self._channel_routers.items():
            cleared = 0
            for direction, port in router.ports.items():
                if direction == Direction.LOCAL:
                    # Skip LOCAL - managed by Selector's wire cycle
                    continue
                if port.clear_output_if_accepted():
                    cleared += 1
            result[channel] = cleared

        return result

    def __repr__(self) -> str:
        sel = "↔Sel" if self.selector_connected else ""
        return f"AXIModeEdgeRouter{self.coord}{sel}"
