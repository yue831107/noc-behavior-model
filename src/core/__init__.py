"""Core NoC components: Flit, Buffer, Router, NI, Mesh."""

from .flit import (
    Flit,
    FlitHeader,
    FlitFactory,
    AxiChannel,
    AxiAwPayload,
    AxiWPayload,
    AxiArPayload,
    AxiBPayload,
    AxiRPayload,
    FlitPayload,
    create_response_flit,
    encode_node_id,
    decode_node_id,
)
from .buffer import (
    Buffer,
    FlitBuffer,
    BufferStats,
    CreditFlowControl,
    PortBuffer,
)
from .packet import (
    Packet,
    PacketType,
    PacketAssembler,
    PacketDisassembler,
    PacketFactory,
)
from .router import (
    Direction,
    ChannelMode,
    PipelineConfig,
    RouterConfig,
    RouterStats,
    RouterPort,
    XYRouter,
    ReqRouter,
    RespRouter,
    Router,
    EdgeRouter,
    PortWire,
    create_router,
    create_channel_router,
)
from .channel_mode_strategy import (
    ChannelModeStrategy,
    GeneralModeStrategy,
    AXIModeStrategy,
    get_channel_mode_strategy,
)
from .ni import (
    NIConfig,
    NIStats,
    SlaveNI,
    MasterNI,
    AXISlave,
    LocalMemoryUnit,
    # Backward compatibility aliases
    NetworkInterface,  # Alias for SlaveNI
    ReqNI,             # Alias for _SlaveNI_ReqPath (deprecated)
    RespNI,            # Alias for _SlaveNI_RspPath (deprecated)
)
from .mesh import (
    MeshConfig,
    MeshStats,
    Mesh,
    create_mesh,
)
from .routing_selector import (
    RoutingSelectorConfig,
    SelectorStats,
    EdgeRouterPort,
    AXIModeEdgeRouterPort,
    RoutingSelector,
    V1System,
    NoCSystem,
)

# Re-export from testbench for backward compatibility
from src.testbench import (
    LocalAXIMasterState,
    LocalTransferConfig,
    LocalAXIMaster,
    NodeController,
    NodeControllerStats,
    MemoryConfig,
    Memory,
    MemoryStats,
    HostMemory,
    LocalMemory,
    MemoryCopyDescriptor,
    AXIIdConfig,
    AXIIdGenerator,
    AXIMasterState,
    PendingTransaction,
    AXIMasterStats,
    AXIMasterController,
    HostAXIMasterState,
    AXIChannelPort,
    AXIResponsePort,
    HostAXIMasterStats,
    HostAXIMaster,
)

# Re-export from verification for backward compatibility
from src.verification import (
    GoldenKey,
    GoldenSource,
    GoldenManager,
    GoldenEntry,
    VerificationResult,
    VerificationReport,
)

__all__ = [
    # Flit
    "Flit",
    "FlitHeader",
    "FlitFactory",
    "AxiChannel",
    "AxiAwPayload",
    "AxiWPayload",
    "AxiArPayload",
    "AxiBPayload",
    "AxiRPayload",
    "FlitPayload",
    "create_response_flit",
    "encode_node_id",
    "decode_node_id",
    # Buffer
    "Buffer",
    "FlitBuffer",
    "BufferStats",
    "CreditFlowControl",
    "PortBuffer",
    # Packet
    "Packet",
    "PacketType",
    "PacketAssembler",
    "PacketDisassembler",
    "PacketFactory",
    # Router
    "Direction",
    "ChannelMode",
    "PipelineConfig",
    "RouterConfig",
    "RouterStats",
    "RouterPort",
    "XYRouter",
    "ReqRouter",
    "RespRouter",
    "Router",
    "EdgeRouter",
    "PortWire",
    "create_router",
    "create_channel_router",
    # Channel Mode Strategy
    "ChannelModeStrategy",
    "GeneralModeStrategy",
    "AXIModeStrategy",
    "get_channel_mode_strategy",
    # NI
    "NIConfig",
    "NIStats",
    "SlaveNI",
    "MasterNI",
    "AXISlave",
    "NetworkInterface",  # Backward compatibility alias for SlaveNI
    "ReqNI",             # Deprecated alias
    "RespNI",            # Deprecated alias
    # Mesh
    "MeshConfig",
    "MeshStats",
    "Mesh",
    "create_mesh",
    # Routing Selector
    "RoutingSelectorConfig",
    "SelectorStats",
    "EdgeRouterPort",
    "AXIModeEdgeRouterPort",
    "RoutingSelector",
    "V1System",
    "NoCSystem",
    # Backward compatibility re-exports from testbench
    "LocalAXIMasterState",
    "LocalTransferConfig",
    "LocalAXIMaster",
    "NodeController",
    "NodeControllerStats",
    "MemoryConfig",
    "Memory",
    "MemoryStats",
    "HostMemory",
    "LocalMemory",
    "MemoryCopyDescriptor",
    "AXIIdConfig",
    "AXIIdGenerator",
    "AXIMasterState",
    "PendingTransaction",
    "AXIMasterStats",
    "AXIMasterController",
    "HostAXIMasterState",
    "AXIChannelPort",
    "AXIResponsePort",
    "HostAXIMasterStats",
    "HostAXIMaster",
    # Backward compatibility re-exports from verification
    "GoldenKey",
    "GoldenSource",
    "GoldenManager",
    "GoldenEntry",
    "VerificationResult",
    "VerificationReport",
]
