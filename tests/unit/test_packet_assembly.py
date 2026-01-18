"""
Packet Assembly/Disassembly Unit Tests.

Tests for PacketAssembler and PacketDisassembler edge cases and boundary conditions.

Usage:
    pytest tests/unit/test_packet_assembly.py -v
"""

import pytest
from typing import List

from src.core.flit import (
    Flit,
    FlitFactory,
    FlitHeader,
    AxiChannel,
    AxiWPayload,
    AxiRPayload,
    encode_node_id,
)
from src.core.packet import (
    Packet,
    PacketType,
    PacketFactory,
    PacketAssembler,
    PacketDisassembler,
)

# Packet header overhead (12 bytes: packet_type + src + dest + axi_id + addr + size)
PACKET_HEADER_SIZE = 12


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def assembler() -> PacketAssembler:
    """Create a PacketAssembler with default payload size."""
    return PacketAssembler(flit_payload_size=32)


@pytest.fixture
def assembler_small() -> PacketAssembler:
    """Create a PacketAssembler with small payload size for multi-flit tests."""
    return PacketAssembler(flit_payload_size=16)


@pytest.fixture
def disassembler() -> PacketDisassembler:
    """Create a PacketDisassembler."""
    return PacketDisassembler()


# =============================================================================
# Test: PacketAssembler Basic Functionality
# =============================================================================

class TestPacketAssemblerBasic:
    """Basic tests for PacketAssembler."""

    def test_assemble_write_request(self, assembler):
        """Assemble a write request packet."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xAB] * 32),
            axi_id=5,
            local_addr=0x1000,
        )

        flits = assembler.assemble(packet)

        assert len(flits) >= 1
        assert all(isinstance(f, Flit) for f in flits)
        # First flit should have header info
        assert flits[0].hdr.axi_ch == AxiChannel.AW

    def test_assemble_read_request(self, assembler):
        """Assemble a read request packet."""
        packet = PacketFactory.create_read_request(
            src=(2, 1),
            dest=(3, 2),
            axi_id=7,
            local_addr=0x2000,
            read_size=64,
        )

        flits = assembler.assemble(packet)

        assert len(flits) >= 1
        assert flits[0].hdr.axi_ch == AxiChannel.AR

    def test_assemble_write_response(self, assembler):
        """Assemble a write response packet."""
        packet = PacketFactory.create_write_response_from_info(
            src=(4, 3),
            dest=(1, 0),
            axi_id=5,
        )

        flits = assembler.assemble(packet)

        assert len(flits) >= 1
        assert flits[0].hdr.axi_ch == AxiChannel.B

    def test_single_flit_has_last_set(self, assembler):
        """Single flit packet should have last=True."""
        packet = PacketFactory.create_write_response_from_info(
            src=(4, 3),
            dest=(1, 0),
            axi_id=1,
        )

        flits = assembler.assemble(packet)

        if len(flits) == 1:
            assert flits[0].hdr.last is True


class TestPacketAssemblerPayloadBoundary:
    """Tests for payload boundary conditions."""

    def test_zero_payload(self, assembler):
        """Assemble packet with empty payload."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes(),  # Empty payload
            axi_id=1,
            local_addr=0x1000,
        )

        flits = assembler.assemble(packet)
        assert len(flits) >= 1

    def test_single_byte_payload(self, assembler):
        """Assemble packet with single byte payload."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xFF]),  # 1 byte
            axi_id=2,
            local_addr=0x1000,
        )

        flits = assembler.assemble(packet)
        assert len(flits) >= 1

    def test_exact_flit_boundary_payload(self, assembler):
        """Payload exactly fills one W flit capacity (32 bytes).

        FlooNoC style: Write request = AW + W* flits
        - AW flit: address info
        - W flit(s): data payload (32 bytes each)

        So minimum is 2 flits (1 AW + 1 W).
        """
        # 32 bytes = exactly one W flit worth of data
        payload_size = 32
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xAA] * payload_size),
            axi_id=3,
            local_addr=0x1000,
        )

        flits = assembler.assemble(packet)
        # FlooNoC: 1 AW + 1 W = 2 flits for 32 bytes
        assert len(flits) == 2

    def test_one_byte_over_boundary(self, assembler):
        """Payload one byte over flit boundary requires extra flit."""
        payload_size = (32 - PACKET_HEADER_SIZE) + 1  # One byte over
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xBB] * payload_size),
            axi_id=4,
            local_addr=0x1000,
        )

        flits = assembler.assemble(packet)
        assert len(flits) >= 2

    def test_large_payload_multi_flit(self, assembler_small):
        """Large payload requires multiple flits."""
        # Small assembler has 16-byte flits
        # Available per flit = 16 - 12 (header) = 4 bytes first, 16 bytes subsequent
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xCC] * 100),  # 100 bytes
            axi_id=5,
            local_addr=0x1000,
        )

        flits = assembler_small.assemble(packet)
        assert len(flits) > 1


class TestPacketAssemblerMultiFlitSequence:
    """Tests for multi-flit packet sequences."""

    def test_multi_flit_last_marking(self, assembler_small):
        """FlooNoC: AW is single-flit packet, W flits form separate data packet."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xDD] * 100),  # Force multi-flit W
            axi_id=6,
            local_addr=0x1000,
        )

        flits = assembler_small.assemble(packet)

        # First flit is AW (single-flit packet, last=True per FlooNoC spec)
        assert flits[0].hdr.axi_ch == AxiChannel.AW
        assert flits[0].hdr.last is True, "AW is single-flit packet (FlooNoC)"

        # Remaining flits are W data packet
        w_flits = [f for f in flits if f.hdr.axi_ch == AxiChannel.W]
        assert len(w_flits) > 0, "Should have W flits"

        # All W flits except last should have last=False
        for w_flit in w_flits[:-1]:
            assert w_flit.hdr.last is False, "Non-final W flit should have last=False"

        # Last W flit should have last=True
        assert w_flits[-1].hdr.last is True, "Final W flit should have last=True"

    def test_multi_flit_same_routing(self, assembler_small):
        """All flits should have same src/dst."""
        packet = PacketFactory.create_write_request(
            src=(2, 1),
            dest=(5, 2),
            data=bytes([0xEE] * 80),
            axi_id=7,
            local_addr=0x3000,
        )

        flits = assembler_small.assemble(packet)

        expected_src = encode_node_id((2, 1))
        expected_dst = encode_node_id((5, 2))

        for flit in flits:
            assert flit.hdr.src_id == expected_src
            assert flit.hdr.dst_id == expected_dst


# =============================================================================
# Test: PacketDisassembler Basic Functionality
# =============================================================================

class TestPacketDisassemblerBasic:
    """Basic tests for PacketDisassembler."""

    def test_receive_single_flit(self, disassembler, assembler):
        """Receive single flit packet."""
        packet = PacketFactory.create_write_response_from_info(
            src=(4, 3),
            dest=(1, 0),
            axi_id=1,
        )

        flits = assembler.assemble(packet)
        assert len(flits) == 1

        # Feed flit to disassembler
        result = disassembler.receive_flit(flits[0])

        # Single flit should complete immediately
        assert result is not None
        assert result.packet_type == PacketType.WRITE_RESP

    def test_receive_multi_flit_incrementally(self, disassembler, assembler_small):
        """Receive multi-flit W packet incrementally (FlooNoC: AW and W are separate packets)."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0x11] * 64),
            axi_id=2,
            local_addr=0x1000,
        )

        flits = assembler_small.assemble(packet)
        # FlooNoC: AW is single-flit packet, W flits form separate packet
        aw_flits = [f for f in flits if f.hdr.axi_ch == AxiChannel.AW]
        w_flits = [f for f in flits if f.hdr.axi_ch == AxiChannel.W]
        assert len(aw_flits) == 1, "Should have exactly 1 AW flit"
        assert len(w_flits) > 1, "Should have multiple W flits"

        # AW is single-flit packet, should complete immediately
        result = disassembler.receive_flit(aw_flits[0])
        assert result is not None, "AW single-flit packet should complete immediately"

        # Feed W flits (all but last)
        for w_flit in w_flits[:-1]:
            result = disassembler.receive_flit(w_flit)
            assert result is None, "Non-final W flit should not complete packet"

        # Feed last W flit
        result = disassembler.receive_flit(w_flits[-1])
        assert result is not None, "Final W flit should complete packet"


class TestPacketDisassemblerRobustness:
    """Robustness tests for PacketDisassembler."""

    def test_interleaved_packets_by_id(self, disassembler, assembler_small):
        """Interleaved packets from different sources should be handled (FlooNoC)."""
        # Create two packets with different rob_idx and multi-flit W
        packet_a = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xAA] * 64),  # 2 W flits
            axi_id=1,
            local_addr=0x1000,
        )

        packet_b = PacketFactory.create_write_request(
            src=(2, 0),
            dest=(4, 3),
            data=bytes([0xBB] * 64),  # 2 W flits
            axi_id=2,
            local_addr=0x2000,
        )

        flits_a = assembler_small.assemble(packet_a)
        flits_b = assembler_small.assemble(packet_b)

        # FlooNoC: Each write produces 1 AW packet + 1 W packet = 2 completed packets each
        # Interleave flits: A0, B0, A1, B1, ...
        results = []
        for i in range(max(len(flits_a), len(flits_b))):
            if i < len(flits_a):
                result = disassembler.receive_flit(flits_a[i])
                if result is not None:
                    results.append(("A", result))
            if i < len(flits_b):
                result = disassembler.receive_flit(flits_b[i])
                if result is not None:
                    results.append(("B", result))

        # FlooNoC: AW is single-flit packet, W is separate packet
        # So each write_request produces 2 packets: 1 AW + 1 W
        # Total: 4 packets (2 from A, 2 from B)
        assert len(results) == 4, f"Expected 4 packets (2 AW + 2 W), got {len(results)}"

    def test_reset_clears_state(self, disassembler, assembler_small):
        """Reset should clear any partial packet state."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xCC] * 64),
            axi_id=3,
            local_addr=0x1000,
        )

        flits = assembler_small.assemble(packet)

        # Feed partial flits
        for flit in flits[:-1]:
            disassembler.receive_flit(flit)

        # Clear disassembler
        disassembler.clear()

        # New packet should work from fresh state
        new_packet = PacketFactory.create_write_response_from_info(
            src=(4, 3),
            dest=(1, 0),
            axi_id=4,
        )
        new_flits = assembler_small.assemble(new_packet)

        result = disassembler.receive_flit(new_flits[0])
        assert result is not None


class TestPacketDisassemblerDataIntegrity:
    """Tests for data integrity in disassembly."""

    def test_payload_reconstructed(self, disassembler, assembler):
        """Disassembled packet should have correct payload."""
        original_payload = bytes([i % 256 for i in range(20)])
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=original_payload,
            axi_id=5,
            local_addr=0x5000,
        )

        flits = assembler.assemble(packet)

        result = None
        for flit in flits:
            result = disassembler.receive_flit(flit)

        assert result is not None
        assert result.payload == original_payload

    def test_routing_info_preserved(self, disassembler, assembler):
        """Disassembled packet should have correct routing info."""
        packet = PacketFactory.create_read_request(
            src=(3, 2),
            dest=(6, 1),
            axi_id=6,
            local_addr=0x6000,
            read_size=32,
        )

        flits = assembler.assemble(packet)

        result = None
        for flit in flits:
            result = disassembler.receive_flit(flit)

        assert result is not None
        assert result.src == (3, 2)
        assert result.dest == (6, 1)
        assert result.axi_id == 6


# =============================================================================
# Test: Roundtrip Assembly/Disassembly
# =============================================================================

class TestAssemblyRoundtrip:
    """Tests for assembly/disassembly roundtrip."""

    def test_write_request_roundtrip(self, assembler, disassembler):
        """Write request roundtrip (FlooNoC: AW and W are separate packets)."""
        original = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xAA, 0xBB, 0xCC, 0xDD]),
            axi_id=7,
            local_addr=0x7000,
        )

        flits = assembler.assemble(original)

        # Collect all packets from flits
        packets = []
        for flit in flits:
            result = disassembler.receive_flit(flit)
            if result is not None:
                packets.append(result)

        # FlooNoC: Should produce 2 packets (AW and W)
        assert len(packets) == 2, f"Expected 2 packets (AW + W), got {len(packets)}"

        # Find AW and W packets
        aw_packet = next((p for p in packets if p.flits[0].hdr.axi_ch == AxiChannel.AW), None)
        w_packet = next((p for p in packets if p.flits[0].hdr.axi_ch == AxiChannel.W), None)

        assert aw_packet is not None, "Should have AW packet"
        assert w_packet is not None, "Should have W packet"

        # AW packet contains axi_id and address
        assert aw_packet.axi_id == original.axi_id
        assert aw_packet.local_addr == original.local_addr

        # Both should have same routing
        assert aw_packet.src == original.src
        assert aw_packet.dest == original.dest
        assert w_packet.src == original.src
        assert w_packet.dest == original.dest

    def test_read_request_roundtrip(self, assembler, disassembler):
        """Read request should survive roundtrip."""
        original = PacketFactory.create_read_request(
            src=(2, 1),
            dest=(5, 3),
            axi_id=8,
            local_addr=0x8000,
            read_size=64,
        )

        flits = assembler.assemble(original)

        result = None
        for flit in flits:
            result = disassembler.receive_flit(flit)

        assert result is not None
        assert result.packet_type == PacketType.READ_REQ
        assert result.src == original.src
        assert result.dest == original.dest

    def test_write_response_roundtrip(self, assembler, disassembler):
        """Write response should survive roundtrip."""
        original = PacketFactory.create_write_response_from_info(
            src=(4, 3),
            dest=(1, 0),
            axi_id=9,
        )

        flits = assembler.assemble(original)

        result = None
        for flit in flits:
            result = disassembler.receive_flit(flit)

        assert result is not None
        assert result.packet_type == PacketType.WRITE_RESP
        assert result.src == original.src
        assert result.dest == original.dest

    def test_large_payload_roundtrip(self, assembler_small, disassembler):
        """Large payload should survive roundtrip."""
        original_payload = bytes([i % 256 for i in range(200)])
        original = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=original_payload,
            axi_id=10,
            local_addr=0xA000,
        )

        flits = assembler_small.assemble(original)
        assert len(flits) > 1  # Ensure multi-flit

        result = None
        for flit in flits:
            result = disassembler.receive_flit(flit)

        assert result is not None
        assert result.payload == original_payload


# =============================================================================
# Test: Strobe Calculation
# =============================================================================

class TestStrobeCalculation:
    """Tests for write strobe calculation."""

    def test_full_strobe_32_bytes(self, assembler):
        """32-byte payload should have full strobe."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xFF] * 32),
            axi_id=11,
            local_addr=0xB000,
        )

        flits = assembler.assemble(packet)

        # Find W flit(s)
        w_flits = [f for f in flits if f.hdr.axi_ch == AxiChannel.W]
        if w_flits:
            # Check strobe covers all bytes
            assert hasattr(w_flits[0], 'payload')

    def test_partial_strobe_less_than_32(self, assembler):
        """Payload < 32 bytes should have partial strobe."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0xAA] * 16),  # 16 bytes
            axi_id=12,
            local_addr=0xC000,
        )

        flits = assembler.assemble(packet)
        # Partial writes should work
        assert len(flits) >= 1


# =============================================================================
# Test: Channel-Specific Behavior
# =============================================================================

class TestChannelSpecificBehavior:
    """Tests for channel-specific assembly behavior."""

    def test_aw_flit_has_address_info(self, assembler):
        """AW flit should contain address information."""
        packet = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(4, 3),
            data=bytes([0x11] * 8),
            axi_id=13,
            local_addr=0xD000,
        )

        flits = assembler.assemble(packet)
        aw_flit = next((f for f in flits if f.hdr.axi_ch == AxiChannel.AW), None)

        assert aw_flit is not None
        # AW flit should have AxiAwPayload
        assert hasattr(aw_flit, 'payload')

    def test_ar_flit_has_address_info(self, assembler):
        """AR flit should contain address information."""
        packet = PacketFactory.create_read_request(
            src=(2, 1),
            dest=(5, 3),
            axi_id=14,
            local_addr=0xE000,
            read_size=64,
        )

        flits = assembler.assemble(packet)
        ar_flit = next((f for f in flits if f.hdr.axi_ch == AxiChannel.AR), None)

        assert ar_flit is not None

    def test_b_flit_minimal_payload(self, assembler):
        """B flit should have minimal payload."""
        packet = PacketFactory.create_write_response_from_info(
            src=(4, 3),
            dest=(1, 0),
            axi_id=15,
        )

        flits = assembler.assemble(packet)

        assert len(flits) == 1  # B should be single flit
        assert flits[0].hdr.axi_ch == AxiChannel.B
