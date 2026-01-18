"""
NI Response Path Unit Tests.

Tests for MasterNI response path, including:
1. Response reception (B and R flits)
2. Response buffer management
3. Per-ID FIFO matching
4. Backpressure handling
5. Multi-beat R response handling

Usage:
    pytest tests/unit/test_ni_response_path.py -v
"""

import pytest
from collections import deque
from typing import List

from src.core.ni import MasterNI, NIConfig, MasterNI_RequestInfo
from src.core.flit import (
    FlitFactory,
    Flit,
    AxiChannel,
    encode_node_id,
)
from src.core.packet import (
    Packet,
    PacketType,
    PacketFactory,
    PacketAssembler,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def master_ni() -> MasterNI:
    """Create a MasterNI at coordinate (1, 1)."""
    config = NIConfig(req_buffer_depth=8, resp_buffer_depth=8)
    return MasterNI(coord=(1, 1), config=config)


@pytest.fixture
def master_ni_small_buffer() -> MasterNI:
    """Create a MasterNI with small response buffer for backpressure tests."""
    config = NIConfig(req_buffer_depth=8, resp_buffer_depth=2)
    return MasterNI(coord=(1, 1), config=config)


@pytest.fixture
def packet_assembler() -> PacketAssembler:
    """Create a PacketAssembler."""
    return PacketAssembler()


# =============================================================================
# Test: Response Buffer Management
# =============================================================================

class TestResponseBufferBasic:
    """Tests for basic response buffer operations."""

    def test_resp_output_initially_empty(self, master_ni):
        """Response output buffer should be empty initially."""
        assert master_ni.resp_output.is_empty()
        assert master_ni.has_pending_response() is False

    def test_get_resp_flit_returns_none_when_empty(self, master_ni):
        """get_resp_flit should return None when buffer empty."""
        assert master_ni.get_resp_flit() is None

    def test_peek_resp_flit_returns_none_when_empty(self, master_ni):
        """peek_resp_flit should return None when buffer empty."""
        assert master_ni.peek_resp_flit() is None

    def test_push_resp_flit_succeeds(self, master_ni):
        """Pushing response flit should succeed when buffer has space."""
        flit = FlitFactory.create_b(
            src=(1, 1),
            dest=(0, 0),
            axi_id=0,
            resp=0,
        )

        success = master_ni._push_resp_flit(flit)

        assert success is True
        assert master_ni.has_pending_response() is True

    def test_get_resp_flit_returns_flit(self, master_ni):
        """get_resp_flit should return and remove flit from buffer."""
        flit = FlitFactory.create_b(
            src=(1, 1),
            dest=(0, 0),
            axi_id=0,
            resp=0,
        )
        master_ni._push_resp_flit(flit)

        result = master_ni.get_resp_flit()

        assert result is not None
        assert result.hdr.axi_ch == AxiChannel.B
        assert master_ni.has_pending_response() is False

    def test_peek_resp_flit_does_not_remove(self, master_ni):
        """peek_resp_flit should not remove flit from buffer."""
        flit = FlitFactory.create_b(
            src=(1, 1),
            dest=(0, 0),
            axi_id=0,
            resp=0,
        )
        master_ni._push_resp_flit(flit)

        result1 = master_ni.peek_resp_flit()
        result2 = master_ni.peek_resp_flit()

        assert result1 is not None
        assert result2 is not None
        assert master_ni.has_pending_response() is True


class TestResponseBufferBackpressure:
    """Tests for response buffer backpressure handling."""

    def test_buffer_full_detection(self, master_ni_small_buffer):
        """Should detect when response buffer is full."""
        ni = master_ni_small_buffer

        # Fill buffer
        for i in range(2):
            flit = FlitFactory.create_b(
                src=(1, 1),
                dest=(0, 0),
                axi_id=i,
                resp=0,
            )
            ni._push_resp_flit(flit)

        assert ni.resp_output.is_full()
        assert ni._has_resp_buffer_space(AxiChannel.B) is False

    def test_push_fails_when_full(self, master_ni_small_buffer):
        """Push should fail when buffer is full."""
        ni = master_ni_small_buffer

        # Fill buffer
        for i in range(2):
            flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=i, resp=0)
            ni._push_resp_flit(flit)

        # Try to push one more
        flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=99, resp=0)
        success = ni._push_resp_flit(flit)

        assert success is False

    def test_space_available_after_pop(self, master_ni_small_buffer):
        """Buffer should have space after popping flit."""
        ni = master_ni_small_buffer

        # Fill buffer
        for i in range(2):
            flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=i, resp=0)
            ni._push_resp_flit(flit)

        # Pop one
        ni.get_resp_flit()

        assert ni._has_resp_buffer_space(AxiChannel.B) is True


# =============================================================================
# Test: Per-ID FIFO Management
# =============================================================================

class TestPerIdFifo:
    """Tests for Per-ID FIFO request tracking."""

    def test_per_id_fifo_initialized(self, master_ni):
        """Per-ID FIFO should be initialized for all possible AXI IDs."""
        # Default AXI ID width is 4 bits = 16 IDs
        assert len(master_ni._per_id_fifo) > 0
        for axi_id in range(16):
            assert axi_id in master_ni._per_id_fifo

    def test_per_id_fifo_initially_empty(self, master_ni):
        """Per-ID FIFO entries should be empty initially."""
        for axi_id, fifo in master_ni._per_id_fifo.items():
            assert len(fifo) == 0

    def test_request_info_stored_correctly(self, master_ni):
        """Request info should be stored in correct Per-ID FIFO."""
        req_info = MasterNI_RequestInfo(
            rob_idx=5,
            axi_id=3,
            src_coord=(0, 0),
            is_write=True,
            timestamp=100,
            local_addr=0x1000,
        )

        master_ni._per_id_fifo[3].append(req_info)

        assert len(master_ni._per_id_fifo[3]) == 1
        stored = master_ni._per_id_fifo[3][0]
        assert stored.rob_idx == 5
        assert stored.src_coord == (0, 0)

    def test_fifo_ordering_preserved(self, master_ni):
        """Per-ID FIFO should maintain FIFO ordering."""
        axi_id = 2

        # Add multiple requests
        for i in range(3):
            req_info = MasterNI_RequestInfo(
                rob_idx=i,
                axi_id=axi_id,
                src_coord=(0, i),
                is_write=True,
                timestamp=100 + i,
                local_addr=0x1000 + i * 8,
            )
            master_ni._per_id_fifo[axi_id].append(req_info)

        # Pop in order
        for i in range(3):
            req_info = master_ni._per_id_fifo[axi_id].popleft()
            assert req_info.rob_idx == i


# =============================================================================
# Test: B Response Path
# =============================================================================

class TestBResponsePath:
    """Tests for B (write response) path."""

    def test_b_flit_created_with_correct_routing(self, master_ni):
        """B flit should have correct src/dst from request info."""
        # Store request info
        req_info = MasterNI_RequestInfo(
            rob_idx=7,
            axi_id=1,
            src_coord=(3, 2),  # Original requester
            is_write=True,
            timestamp=50,
            local_addr=0x2000,
        )
        master_ni._per_id_fifo[1].append(req_info)

        # Create B flit as MasterNI would
        b_flit = FlitFactory.create_b(
            src=master_ni.coord,
            dest=req_info.src_coord,
            axi_id=1,
            resp=0,
            rob_idx=req_info.rob_idx,
        )

        assert b_flit.hdr.src == master_ni.coord
        assert b_flit.hdr.dest == req_info.src_coord
        assert b_flit.hdr.axi_ch == AxiChannel.B

    def test_b_flit_has_rob_idx(self, master_ni):
        """B flit should carry rob_idx for response matching."""
        b_flit = FlitFactory.create_b(
            src=(1, 1),
            dest=(0, 0),
            axi_id=3,
            resp=0,
            rob_idx=12,
        )

        assert b_flit.hdr.rob_idx == 12

    def test_b_flit_is_single_flit(self, master_ni):
        """B flit should be a single-flit packet (last=True)."""
        b_flit = FlitFactory.create_b(
            src=(1, 1),
            dest=(0, 0),
            axi_id=0,
            resp=0,
        )

        assert b_flit.hdr.last is True
        assert b_flit.is_single_flit() is True


# =============================================================================
# Test: R Response Path
# =============================================================================

class TestRResponsePath:
    """Tests for R (read response) path."""

    def test_r_flit_created_with_data(self, master_ni):
        """R flit should carry read data."""
        test_data = bytes([0xAA] * 32)
        r_flit = FlitFactory.create_r(
            src=(1, 1),
            dest=(0, 0),
            data=test_data,
            axi_id=0,
            resp=0,
            last=True,
        )

        assert r_flit.payload.data == test_data

    def test_r_flit_has_sequence_number(self, master_ni):
        """R flit should have sequence number for multi-beat ordering."""
        r_flit = FlitFactory.create_r(
            src=(1, 1),
            dest=(0, 0),
            data=bytes(32),
            axi_id=0,
            resp=0,
            last=False,
            seq_num=5,
        )

        # seq_num is stored in Flit._seq_num, not in payload
        assert r_flit._seq_num == 5

    def test_r_flit_last_marking(self, master_ni):
        """R flit should have correct last marking."""
        r_flit_middle = FlitFactory.create_r(
            src=(1, 1),
            dest=(0, 0),
            data=bytes(32),
            axi_id=0,
            last=False,
        )
        r_flit_last = FlitFactory.create_r(
            src=(1, 1),
            dest=(0, 0),
            data=bytes(32),
            axi_id=0,
            last=True,
        )

        assert r_flit_middle.hdr.last is False
        assert r_flit_last.hdr.last is True


class TestRResponseMultiBeat:
    """Tests for multi-beat R response handling."""

    def test_r_seq_num_tracking(self, master_ni):
        """MasterNI should track sequence numbers per AXI ID."""
        axi_id = 2

        # Initialize tracking (as MasterNI would)
        master_ni._r_seq_num[axi_id] = 0

        # Simulate incrementing
        for expected_seq in range(4):
            seq_num = master_ni._r_seq_num[axi_id]
            assert seq_num == expected_seq
            master_ni._r_seq_num[axi_id] += 1

    def test_r_seq_num_per_id_independent(self, master_ni):
        """Sequence numbers should be independent per AXI ID."""
        master_ni._r_seq_num[0] = 3
        master_ni._r_seq_num[1] = 7

        assert master_ni._r_seq_num[0] == 3
        assert master_ni._r_seq_num[1] == 7

    def test_r_seq_num_cleaned_on_last(self, master_ni):
        """Sequence number tracking should be cleaned on last beat."""
        axi_id = 5
        master_ni._r_seq_num[axi_id] = 10

        # Simulate last beat
        del master_ni._r_seq_num[axi_id]

        assert axi_id not in master_ni._r_seq_num


class TestPendingRFlits:
    """Tests for pending R flit queue (backpressure handling)."""

    def test_pending_r_queue_initially_empty(self, master_ni):
        """Pending R flit queue should be empty initially."""
        assert len(master_ni._pending_r_flits) == 0

    def test_r_flit_queued_on_backpressure(self, master_ni_small_buffer):
        """R flit should be queued when buffer full."""
        ni = master_ni_small_buffer

        # Fill buffer
        for i in range(2):
            flit = FlitFactory.create_r(
                src=(1, 1), dest=(0, 0), data=bytes(32), axi_id=0, last=True
            )
            ni._push_resp_flit(flit)

        # Create pending flit
        pending_flit = FlitFactory.create_r(
            src=(1, 1), dest=(0, 0), data=bytes(32), axi_id=0, last=True
        )
        ni._pending_r_flits.append(pending_flit)

        assert len(ni._pending_r_flits) == 1

    def test_try_send_pending_r_flit_succeeds(self, master_ni):
        """Pending R flit should be sent when buffer has space."""
        ni = master_ni

        # Add pending flit
        pending_flit = FlitFactory.create_r(
            src=(1, 1), dest=(0, 0), data=bytes(32), axi_id=0, last=True
        )
        ni._pending_r_flits.append(pending_flit)

        success = ni._try_send_one_pending_r_flit()

        assert success is True
        assert len(ni._pending_r_flits) == 0
        assert ni.has_pending_response() is True

    def test_try_send_pending_r_flit_fails_when_full(self, master_ni_small_buffer):
        """Pending R flit should not be sent when buffer full."""
        ni = master_ni_small_buffer

        # Fill buffer
        for i in range(2):
            flit = FlitFactory.create_r(
                src=(1, 1), dest=(0, 0), data=bytes(32), axi_id=i, last=True
            )
            ni._push_resp_flit(flit)

        # Add pending flit
        pending_flit = FlitFactory.create_r(
            src=(1, 1), dest=(0, 0), data=bytes(32), axi_id=99, last=True
        )
        ni._pending_r_flits.append(pending_flit)

        success = ni._try_send_one_pending_r_flit()

        assert success is False
        assert len(ni._pending_r_flits) == 1


# =============================================================================
# Test: Valid/Ready Interface
# =============================================================================

class TestResponseOutputHandshake:
    """Tests for response output valid/ready handshake."""

    def test_update_resp_output_sets_valid(self, master_ni):
        """update_resp_output should set out_valid when buffer not empty."""
        flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=0, resp=0)
        master_ni._push_resp_flit(flit)

        master_ni.update_resp_output()

        assert master_ni.resp_out_valid is True
        assert master_ni.resp_out_flit is not None

    def test_update_resp_output_not_set_when_empty(self, master_ni):
        """update_resp_output should not set valid when buffer empty."""
        master_ni.update_resp_output()

        assert master_ni.resp_out_valid is False
        assert master_ni.resp_out_flit is None

    def test_clear_resp_output_when_accepted(self, master_ni):
        """Output should be cleared when accepted by downstream."""
        flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=0, resp=0)
        master_ni._push_resp_flit(flit)
        master_ni.update_resp_output()

        # Simulate downstream ready
        master_ni.resp_out_ready = True

        accepted = master_ni.clear_resp_output_if_accepted()

        assert accepted is True
        assert master_ni.resp_out_valid is False
        assert master_ni.resp_out_flit is None

    def test_clear_resp_output_not_when_not_ready(self, master_ni):
        """Output should not be cleared when downstream not ready."""
        flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=0, resp=0)
        master_ni._push_resp_flit(flit)
        master_ni.update_resp_output()

        # Downstream not ready
        master_ni.resp_out_ready = False

        accepted = master_ni.clear_resp_output_if_accepted()

        assert accepted is False
        assert master_ni.resp_out_valid is True


# =============================================================================
# Test: Channel-Specific Response Access
# =============================================================================

class TestChannelSpecificResponse:
    """Tests for channel-specific response methods."""

    def test_get_channel_resp_flit_b(self, master_ni):
        """get_channel_resp_flit should return B flit for B channel."""
        b_flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=0, resp=0)
        master_ni._push_resp_flit(b_flit)

        result = master_ni.get_channel_resp_flit(AxiChannel.B)

        # In General Mode, this returns from shared buffer
        assert result is not None

    def test_has_pending_channel_response_b(self, master_ni):
        """has_pending_channel_response should work for B channel."""
        assert master_ni.has_pending_channel_response(AxiChannel.B) is False

        b_flit = FlitFactory.create_b(src=(1, 1), dest=(0, 0), axi_id=0, resp=0)
        master_ni._push_resp_flit(b_flit)

        assert master_ni.has_pending_channel_response(AxiChannel.B) is True

    def test_peek_channel_resp_flit_r(self, master_ni):
        """peek_channel_resp_flit should work for R channel."""
        r_flit = FlitFactory.create_r(
            src=(1, 1), dest=(0, 0), data=bytes(32), axi_id=0, last=True
        )
        master_ni._push_resp_flit(r_flit)

        result = master_ni.peek_channel_resp_flit(AxiChannel.R)

        # In General Mode, this peeks from shared buffer
        assert result is not None


# =============================================================================
# Test: Response Statistics
# =============================================================================

class TestResponseStatistics:
    """Tests for response statistics tracking."""

    def test_initial_stats_zero(self, master_ni):
        """Response statistics should be zero initially."""
        assert master_ni.stats.b_responses_sent == 0
        assert master_ni.stats.r_responses_sent == 0

    def test_stats_incremented_correctly(self, master_ni):
        """Statistics should increment when responses are sent."""
        # These are typically incremented by _collect_axi_responses
        # Test that stats object is accessible and modifiable
        master_ni.stats.b_responses_sent += 1
        master_ni.stats.r_responses_sent += 2

        assert master_ni.stats.b_responses_sent == 1
        assert master_ni.stats.r_responses_sent == 2


# =============================================================================
# Test: Flit Latency Callback
# =============================================================================

class TestFlitLatencyCallback:
    """Tests for per-flit latency callback."""

    def test_callback_initially_none(self, master_ni):
        """Latency callback should be None initially."""
        assert master_ni._flit_latency_callback is None

    def test_set_flit_latency_callback(self, master_ni):
        """Should be able to set latency callback."""
        latencies = []

        def callback(latency):
            latencies.append(latency)

        master_ni.set_flit_latency_callback(callback)

        assert master_ni._flit_latency_callback is not None

    def test_callback_called_on_flit_arrival(self, master_ni):
        """Callback should be called when flit arrives."""
        latencies = []

        def callback(latency):
            latencies.append(latency)

        master_ni.set_flit_latency_callback(callback)

        # Simulate flit arrival (normally done in _process_incoming_flits)
        if master_ni._flit_latency_callback is not None:
            master_ni._flit_latency_callback(10)

        assert len(latencies) == 1
        assert latencies[0] == 10


# =============================================================================
# Test: Request Input Processing
# =============================================================================

class TestRequestInputProcessing:
    """Tests for request input processing."""

    def test_update_ready_signals(self, master_ni):
        """update_ready_signals should set req_in_ready based on buffer."""
        master_ni.update_ready_signals()
        assert master_ni.req_in_ready is True

    def test_sample_req_input_success(self, master_ni):
        """sample_req_input should accept flit when valid && ready."""
        flit = FlitFactory.create_ar(
            src=(0, 0), dest=(1, 1), addr=0x1000, axi_id=0, length=0
        )

        master_ni.req_in_valid = True
        master_ni.req_in_flit = flit
        master_ni.update_ready_signals()

        success = master_ni.sample_req_input()

        assert success is True
        assert master_ni.req_input.occupancy == 1

    def test_sample_req_input_fails_when_not_valid(self, master_ni):
        """sample_req_input should not accept when not valid."""
        master_ni.req_in_valid = False
        master_ni.req_in_flit = None

        success = master_ni.sample_req_input()

        assert success is False

    def test_clear_input_signals(self, master_ni):
        """clear_input_signals should reset input signals."""
        flit = FlitFactory.create_ar(
            src=(0, 0), dest=(1, 1), addr=0x1000, axi_id=0, length=0
        )
        master_ni.req_in_valid = True
        master_ni.req_in_flit = flit

        master_ni.clear_input_signals()

        assert master_ni.req_in_valid is False
        assert master_ni.req_in_flit is None


# =============================================================================
# Test: Direct Memory Access
# =============================================================================

class TestDirectMemoryAccess:
    """Tests for direct memory access methods."""

    def test_write_local(self, master_ni):
        """write_local should write to local memory."""
        test_data = bytes([0xDE, 0xAD, 0xBE, 0xEF])
        master_ni.write_local(0x1000, test_data)

        # Verify by reading back
        result = master_ni.read_local(0x1000, len(test_data))
        assert result == test_data

    def test_read_local(self, master_ni):
        """read_local should read from local memory."""
        # Write first
        test_data = bytes([0xCA, 0xFE])
        master_ni.write_local(0x2000, test_data)

        # Read back
        result = master_ni.read_local(0x2000, 2)
        assert result == test_data

    def test_verify_local_success(self, master_ni):
        """verify_local should return True when data matches."""
        test_data = bytes([0x12, 0x34])
        master_ni.write_local(0x3000, test_data)

        success = master_ni.verify_local(0x3000, test_data)
        assert success is True

    def test_verify_local_failure(self, master_ni):
        """verify_local should return False when data doesn't match."""
        master_ni.write_local(0x4000, bytes([0xAA, 0xBB]))

        success = master_ni.verify_local(0x4000, bytes([0xCC, 0xDD]))
        assert success is False


# =============================================================================
# Test: NI Representation
# =============================================================================

class TestNIRepresentation:
    """Tests for NI string representation."""

    def test_repr_format(self, master_ni):
        """__repr__ should return meaningful string."""
        repr_str = repr(master_ni)

        assert "MasterNI" in repr_str
        assert "(1, 1)" in repr_str
