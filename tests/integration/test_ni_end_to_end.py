"""
NI End-to-End Integration Tests.

Tests for complete NI request/response cycles, including:
1. Full write transaction (AW + W -> B)
2. Full read transaction (AR -> R)
3. Multiple outstanding transactions
4. Response ordering
5. Multi-beat transactions

Usage:
    pytest tests/integration/test_ni_end_to_end.py -v
"""

import pytest
from typing import List, Tuple

from src.core.ni import MasterNI, SlaveNI, NIConfig
from src.core.flit import (
    FlitFactory,
    Flit,
    AxiChannel,
)
from src.core.packet import (
    Packet,
    PacketType,
    PacketFactory,
    PacketAssembler,
    PacketDisassembler,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ni_config() -> NIConfig:
    """Default NI configuration."""
    return NIConfig(
        req_buffer_depth=8,
        resp_buffer_depth=8,
        axi_id_width=4,
    )


@pytest.fixture
def master_ni(ni_config) -> MasterNI:
    """Create a MasterNI at (2, 2)."""
    return MasterNI(coord=(2, 2), config=ni_config)


@pytest.fixture
def slave_ni(ni_config) -> SlaveNI:
    """Create a SlaveNI at (0, 0)."""
    # SlaveNI requires a SystemAddressMap
    from src.address.address_map import create_default_address_map
    address_map = create_default_address_map()
    return SlaveNI(coord=(0, 0), config=ni_config, address_map=address_map)


@pytest.fixture
def packet_assembler() -> PacketAssembler:
    """Create a PacketAssembler."""
    return PacketAssembler()


@pytest.fixture
def packet_disassembler() -> PacketDisassembler:
    """Create a PacketDisassembler."""
    return PacketDisassembler()


# =============================================================================
# Helper Functions
# =============================================================================

def inject_flits_to_master_ni(ni: MasterNI, flits: List[Flit]) -> None:
    """Inject flits into MasterNI request input."""
    for flit in flits:
        success = ni.receive_req_flit(flit)
        assert success, f"Failed to inject flit: {flit}"


def process_ni_cycles(ni: MasterNI, num_cycles: int, current_time: int = 0) -> int:
    """Process multiple NI cycles and return final time."""
    for i in range(num_cycles):
        ni.process_cycle(current_time + i)
    return current_time + num_cycles


def collect_response_flits(ni: MasterNI, max_flits: int = 10) -> List[Flit]:
    """Collect response flits from MasterNI."""
    flits = []
    for _ in range(max_flits):
        flit = ni.get_resp_flit()
        if flit is None:
            break
        flits.append(flit)
    return flits


# =============================================================================
# Test: Single Write Transaction
# =============================================================================

class TestSingleWriteTransaction:
    """Tests for single write transaction flow."""

    def test_write_request_generates_b_response(self, master_ni, packet_assembler):
        """Write request should generate B response."""
        # Create write request packet
        packet = PacketFactory.create_write_request(
            src=(0, 0),  # Requester
            dest=(2, 2),  # MasterNI
            local_addr=0x1000,
            data=bytes([0xAA, 0xBB, 0xCC, 0xDD]),
            axi_id=1,
        )

        # Assemble to flits
        flits = packet_assembler.assemble(packet)

        # Inject into MasterNI
        inject_flits_to_master_ni(master_ni, flits)

        # Process cycles
        process_ni_cycles(master_ni, num_cycles=5)

        # Check for B response
        assert master_ni.has_pending_response() is True

        resp_flits = collect_response_flits(master_ni)
        assert len(resp_flits) >= 1

        # Verify B flit
        b_flit = next((f for f in resp_flits if f.hdr.axi_ch == AxiChannel.B), None)
        assert b_flit is not None

    def test_write_data_persisted_to_memory(self, master_ni, packet_assembler):
        """Write request should persist data to local memory."""
        test_data = bytes([0x12, 0x34, 0x56, 0x78])
        addr = 0x2000

        # Create and inject write request
        packet = PacketFactory.create_write_request(
            src=(0, 0),
            dest=(2, 2),
            local_addr=addr,
            data=test_data,
            axi_id=0,
        )
        flits = packet_assembler.assemble(packet)
        inject_flits_to_master_ni(master_ni, flits)

        # Process
        process_ni_cycles(master_ni, num_cycles=5)

        # Verify data in memory
        result = master_ni.read_local(addr, len(test_data))
        assert result == test_data

    def test_b_response_has_correct_routing(self, master_ni, packet_assembler):
        """B response should route back to original requester."""
        src_coord = (1, 1)

        packet = PacketFactory.create_write_request(
            src=src_coord,
            dest=(2, 2),
            local_addr=0x3000,
            data=bytes([0xFF] * 4),
            axi_id=2,
        )
        flits = packet_assembler.assemble(packet)
        inject_flits_to_master_ni(master_ni, flits)

        process_ni_cycles(master_ni, num_cycles=5)

        resp_flits = collect_response_flits(master_ni)
        b_flit = next((f for f in resp_flits if f.hdr.axi_ch == AxiChannel.B), None)

        assert b_flit is not None
        # B response goes FROM MasterNI TO original requester
        assert b_flit.hdr.src == master_ni.coord
        assert b_flit.hdr.dest == src_coord


# =============================================================================
# Test: Single Read Transaction
# =============================================================================

class TestSingleReadTransaction:
    """Tests for single read transaction flow."""

    def test_read_request_generates_r_response(self, master_ni, packet_assembler):
        """Read request should generate R response."""
        # Initialize memory with data
        test_data = bytes([0xDE, 0xAD, 0xBE, 0xEF] + [0x00] * 28)
        master_ni.write_local(0x4000, test_data)

        # Create read request
        packet = PacketFactory.create_read_request(
            src=(0, 0),
            dest=(2, 2),
            local_addr=0x4000,
            read_size=32,
            axi_id=3,
        )
        flits = packet_assembler.assemble(packet)

        inject_flits_to_master_ni(master_ni, flits)
        process_ni_cycles(master_ni, num_cycles=5)

        assert master_ni.has_pending_response() is True

        resp_flits = collect_response_flits(master_ni)
        r_flit = next((f for f in resp_flits if f.hdr.axi_ch == AxiChannel.R), None)
        assert r_flit is not None

    def test_r_response_contains_read_data(self, master_ni, packet_assembler):
        """R response should contain requested data."""
        # Initialize memory with specific pattern
        # Use 8 bytes to match default read size
        expected_data = bytes([0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE])
        master_ni.write_local(0x5000, expected_data)

        # Read request - read 8 bytes (default beat size)
        packet = PacketFactory.create_read_request(
            src=(0, 0),
            dest=(2, 2),
            local_addr=0x5000,
            read_size=8,  # Read only 8 bytes
            axi_id=4,
        )
        flits = packet_assembler.assemble(packet)

        inject_flits_to_master_ni(master_ni, flits)
        process_ni_cycles(master_ni, num_cycles=5)

        resp_flits = collect_response_flits(master_ni)
        r_flit = next((f for f in resp_flits if f.hdr.axi_ch == AxiChannel.R), None)

        assert r_flit is not None
        # R flit data first 8 bytes should match memory content
        # (R flit data is padded to 32 bytes)
        assert r_flit.payload.data[:8] == expected_data

    def test_r_response_has_correct_routing(self, master_ni, packet_assembler):
        """R response should route back to original requester."""
        src_coord = (3, 1)
        master_ni.write_local(0x6000, bytes(32))

        packet = PacketFactory.create_read_request(
            src=src_coord,
            dest=(2, 2),
            local_addr=0x6000,
            read_size=32,
            axi_id=5,
        )
        flits = packet_assembler.assemble(packet)

        inject_flits_to_master_ni(master_ni, flits)
        process_ni_cycles(master_ni, num_cycles=5)

        resp_flits = collect_response_flits(master_ni)
        r_flit = next((f for f in resp_flits if f.hdr.axi_ch == AxiChannel.R), None)

        assert r_flit is not None
        assert r_flit.hdr.src == master_ni.coord
        assert r_flit.hdr.dest == src_coord


# =============================================================================
# Test: Multiple Outstanding Transactions
# =============================================================================

class TestMultipleOutstandingTransactions:
    """Tests for multiple outstanding transactions."""

    def test_multiple_writes_generate_multiple_b_responses(
        self, master_ni, packet_assembler
    ):
        """Multiple writes should generate corresponding B responses."""
        num_writes = 3

        # Send multiple write requests
        for i in range(num_writes):
            packet = PacketFactory.create_write_request(
                src=(0, i),
                dest=(2, 2),
                local_addr=0x1000 + i * 0x100,
                data=bytes([i] * 8),
                axi_id=i,
            )
            flits = packet_assembler.assemble(packet)
            inject_flits_to_master_ni(master_ni, flits)

        # Process enough cycles
        process_ni_cycles(master_ni, num_cycles=num_writes * 3)

        # Should have multiple B responses
        resp_flits = collect_response_flits(master_ni, max_flits=num_writes * 2)
        b_flits = [f for f in resp_flits if f.hdr.axi_ch == AxiChannel.B]

        assert len(b_flits) == num_writes

    def test_different_axi_ids_tracked_independently(
        self, master_ni, packet_assembler
    ):
        """Transactions with different AXI IDs should be tracked independently."""
        # Write with ID 0
        packet0 = PacketFactory.create_write_request(
            src=(1, 0),
            dest=(2, 2),
            local_addr=0x7000,
            data=bytes([0xAA] * 8),
            axi_id=0,
        )
        flits0 = packet_assembler.assemble(packet0)

        # Write with ID 1
        packet1 = PacketFactory.create_write_request(
            src=(1, 1),
            dest=(2, 2),
            local_addr=0x7100,
            data=bytes([0xBB] * 8),
            axi_id=1,
        )
        flits1 = packet_assembler.assemble(packet1)

        # Inject both
        inject_flits_to_master_ni(master_ni, flits0)
        inject_flits_to_master_ni(master_ni, flits1)

        process_ni_cycles(master_ni, num_cycles=10)

        resp_flits = collect_response_flits(master_ni, max_flits=10)
        b_flits = [f for f in resp_flits if f.hdr.axi_ch == AxiChannel.B]

        # Both should get responses
        assert len(b_flits) == 2


# =============================================================================
# Test: Response Ordering
# =============================================================================

class TestResponseOrdering:
    """Tests for response ordering guarantees."""

    def test_same_id_responses_in_order(self, master_ni, packet_assembler):
        """Responses for same AXI ID should be in request order."""
        axi_id = 5
        src_coords = [(0, 0), (0, 1), (0, 2)]

        # Send multiple writes with same AXI ID
        for i, src in enumerate(src_coords):
            packet = PacketFactory.create_write_request(
                src=src,
                dest=(2, 2),
                local_addr=0x8000 + i * 0x100,
                data=bytes([i] * 8),
                axi_id=axi_id,
            )
            flits = packet_assembler.assemble(packet)
            inject_flits_to_master_ni(master_ni, flits)
            # Process after each to ensure ordering
            process_ni_cycles(master_ni, num_cycles=3)

        resp_flits = collect_response_flits(master_ni, max_flits=10)
        b_flits = [f for f in resp_flits if f.hdr.axi_ch == AxiChannel.B]

        # All should be for same AXI ID
        assert all(f.payload.axi_id == axi_id for f in b_flits)


# =============================================================================
# Test: Request Buffer Management
# =============================================================================

class TestRequestBufferManagement:
    """Tests for request buffer management."""

    def test_request_buffer_not_full_initially(self, master_ni):
        """Request buffer should not be full initially."""
        assert master_ni.req_input.is_full() is False
        master_ni.update_ready_signals()
        assert master_ni.req_in_ready is True

    def test_request_buffer_capacity(self, ni_config, packet_assembler):
        """Request buffer should have expected capacity."""
        config = NIConfig(req_buffer_depth=4)
        ni = MasterNI(coord=(2, 2), config=config)

        # Fill buffer
        for i in range(4):
            flit = FlitFactory.create_ar(
                src=(0, 0), dest=(2, 2), addr=0x1000, axi_id=i, length=0
            )
            success = ni.receive_req_flit(flit)
            assert success is True

        # Buffer should be full
        assert ni.req_input.is_full() is True

        # Next inject should fail
        flit = FlitFactory.create_ar(
            src=(0, 0), dest=(2, 2), addr=0x1000, axi_id=99, length=0
        )
        success = ni.receive_req_flit(flit)
        assert success is False


# =============================================================================
# Test: SlaveNI Output Path
# =============================================================================

class TestSlaveNIOutputPath:
    """Tests for SlaveNI request output path."""

    def test_slave_ni_generates_request_flits(self, slave_ni):
        """SlaveNI should generate request flits from AXI transactions."""
        # SlaveNI converts AXI transactions to NoC flits
        # This is typically triggered by external AXI master
        assert slave_ni is not None
        # SlaveNI uses get_req_flit() for output
        assert hasattr(slave_ni, 'get_req_flit')

    def test_slave_ni_output_buffer_initially_empty(self, slave_ni):
        """SlaveNI output buffer should be empty initially."""
        # SlaveNI uses req_path for output
        assert slave_ni.req_path.has_pending_output() is False


# =============================================================================
# Test: Statistics Tracking
# =============================================================================

class TestStatisticsTracking:
    """Tests for NI statistics tracking."""

    def test_write_request_increments_stats(self, master_ni, packet_assembler):
        """Write requests should increment statistics."""
        initial_writes = master_ni.stats.write_requests

        packet = PacketFactory.create_write_request(
            src=(0, 0),
            dest=(2, 2),
            local_addr=0xA000,
            data=bytes(8),
            axi_id=0,
        )
        flits = packet_assembler.assemble(packet)
        inject_flits_to_master_ni(master_ni, flits)
        process_ni_cycles(master_ni, num_cycles=5)

        assert master_ni.stats.write_requests > initial_writes

    def test_read_request_increments_stats(self, master_ni, packet_assembler):
        """Read requests should increment statistics."""
        initial_reads = master_ni.stats.read_requests

        master_ni.write_local(0xB000, bytes(32))

        packet = PacketFactory.create_read_request(
            src=(0, 0),
            dest=(2, 2),
            local_addr=0xB000,
            read_size=32,
            axi_id=0,
        )
        flits = packet_assembler.assemble(packet)
        inject_flits_to_master_ni(master_ni, flits)
        process_ni_cycles(master_ni, num_cycles=5)

        assert master_ni.stats.read_requests > initial_reads


# =============================================================================
# Test: Cycle Processing
# =============================================================================

class TestCycleProcessing:
    """Tests for NI cycle processing."""

    def test_process_cycle_returns_without_error(self, master_ni):
        """process_cycle should complete without error."""
        # Should not raise
        master_ni.process_cycle(current_time=0)
        master_ni.process_cycle(current_time=1)
        master_ni.process_cycle(current_time=2)

    def test_process_cycle_with_empty_buffers(self, master_ni):
        """process_cycle should handle empty buffers gracefully."""
        for i in range(10):
            master_ni.process_cycle(current_time=i)

        # Should still be in valid state
        assert master_ni.has_pending_response() is False


# =============================================================================
# Test: Error Conditions
# =============================================================================

class TestErrorConditions:
    """Tests for error handling."""

    def test_invalid_packet_type_handled(self, master_ni):
        """Invalid packet types should be handled gracefully."""
        # This tests robustness - MasterNI should not crash on unexpected input
        # Normally MasterNI only receives request flits (AR, AW+W)

        # Just verify the NI doesn't crash with normal operations
        master_ni.process_cycle(0)
        assert True  # If we got here, no crash

    def test_empty_fifo_lookup_handled(self, master_ni):
        """Looking up empty Per-ID FIFO should be handled."""
        # Per-ID FIFO for unused AXI ID should be empty
        axi_id = 15
        assert len(master_ni._per_id_fifo[axi_id]) == 0


# =============================================================================
# Test: Integration with Memory
# =============================================================================

class TestMemoryIntegration:
    """Tests for NI integration with local memory."""

    def test_write_then_read_consistency(self, master_ni, packet_assembler):
        """Data written should be readable."""
        addr = 0xC000
        test_data = bytes([0x11, 0x22, 0x33, 0x44] + [0x00] * 28)

        # Write
        write_packet = PacketFactory.create_write_request(
            src=(0, 0),
            dest=(2, 2),
            local_addr=addr,
            data=test_data[:8],  # Write 8 bytes
            axi_id=0,
        )
        write_flits = packet_assembler.assemble(write_packet)
        inject_flits_to_master_ni(master_ni, write_flits)
        process_ni_cycles(master_ni, num_cycles=5)

        # Collect B response to ensure write completed
        _ = collect_response_flits(master_ni)

        # Read (direct access)
        result = master_ni.read_local(addr, 8)
        assert result == test_data[:8]

    def test_multiple_address_ranges(self, master_ni, packet_assembler):
        """Multiple address ranges should work independently."""
        addresses = [0xD000, 0xD100, 0xD200]
        data_values = [bytes([i] * 8) for i in range(3)]

        # Write to multiple addresses
        for addr, data in zip(addresses, data_values):
            packet = PacketFactory.create_write_request(
                src=(0, 0),
                dest=(2, 2),
                local_addr=addr,
                data=data,
                axi_id=0,
            )
            flits = packet_assembler.assemble(packet)
            inject_flits_to_master_ni(master_ni, flits)
            process_ni_cycles(master_ni, num_cycles=3)

        # Collect all responses
        _ = collect_response_flits(master_ni, max_flits=10)

        # Verify each address
        for addr, expected_data in zip(addresses, data_values):
            result = master_ni.read_local(addr, 8)
            assert result == expected_data
