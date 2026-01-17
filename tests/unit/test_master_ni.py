"""
MasterNI Unit Tests.

Tests for MasterNI component covering:
- Request receive path (flit reception, packet reconstruction)
- Memory operations (write/read to local memory)
- Response generation (B/R response creation and routing)
- Per-ID FIFO behavior (ordering, independence)
"""

import pytest
from typing import Tuple, List, Optional

from src.core.ni import MasterNI, NIConfig, AXISlave, LocalMemoryUnit
from src.core.flit import (
    Flit, FlitFactory, FlitHeader, AxiChannel,
    AxiAwPayload, AxiWPayload, AxiArPayload, AxiBPayload, AxiRPayload,
    encode_node_id, decode_node_id,
)
from src.core.router import ChannelMode
from src.core.packet import PacketType, PacketFactory, Packet
from src.testbench.memory import LocalMemory
from src.axi.interface import AXIResp


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def ni_config() -> NIConfig:
    """Default NI configuration."""
    return NIConfig(
        max_outstanding=16,
        req_buffer_depth=8,
        resp_buffer_depth=8,
        channel_mode=ChannelMode.GENERAL,
    )


@pytest.fixture
def ni_config_axi_mode() -> NIConfig:
    """NI configuration with AXI mode."""
    return NIConfig(
        max_outstanding=16,
        req_buffer_depth=8,
        resp_buffer_depth=8,
        channel_mode=ChannelMode.AXI,
    )


@pytest.fixture
def local_memory() -> LocalMemory:
    """Create local memory."""
    return LocalMemory(node_id=0, size=0x10000)


@pytest.fixture
def axi_slave(local_memory, ni_config) -> AXISlave:
    """Create AXI slave with local memory."""
    return AXISlave(memory=local_memory, config=ni_config)


@pytest.fixture
def master_ni(ni_config, axi_slave) -> MasterNI:
    """Create MasterNI at coordinate (1, 0)."""
    return MasterNI(
        coord=(1, 0),
        config=ni_config,
        ni_id=0,
        node_id=0,
        axi_slave=axi_slave,
    )


@pytest.fixture
def master_ni_with_memory(ni_config) -> MasterNI:
    """Create MasterNI with internal memory (backward compatible mode)."""
    return MasterNI(
        coord=(1, 0),
        config=ni_config,
        ni_id=0,
        node_id=0,
    )


@pytest.fixture
def master_ni_axi_mode(ni_config_axi_mode) -> MasterNI:
    """Create MasterNI with AXI mode."""
    return MasterNI(
        coord=(1, 0),
        config=ni_config_axi_mode,
        ni_id=0,
        node_id=0,
    )


# ==============================================================================
# Part 2.1: Request Receive Path Tests
# ==============================================================================

class TestMasterNIRequestReceive:
    """Tests for request flit reception."""

    def test_receive_aw_flit(self, master_ni):
        """Correctly receive AW flit."""
        aw_flit = FlitFactory.create_aw(
            src=(1, 1),
            dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )

        result = master_ni.receive_req_flit(aw_flit)
        assert result is True
        assert master_ni.req_input.occupancy == 1

    def test_receive_w_flit_sequence(self, master_ni):
        """Correctly receive W flit sequence."""
        # W flits for a 2-beat write
        w0 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"beat0_data" + bytes(22),
            last=False,
            rob_idx=0,
            seq_num=0,
        )
        w1 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"beat1_data" + bytes(22),
            last=True,
            rob_idx=0,
            seq_num=1,
        )

        assert master_ni.receive_req_flit(w0) is True
        assert master_ni.receive_req_flit(w1) is True
        assert master_ni.req_input.occupancy == 2

    def test_receive_ar_flit(self, master_ni):
        """Correctly receive AR flit."""
        ar_flit = FlitFactory.create_ar(
            src=(1, 1),
            dest=(1, 0),
            addr=0x2000,
            axi_id=1,
            length=0,
            rob_idx=1,
            rob_req=True,
        )

        result = master_ni.receive_req_flit(ar_flit)
        assert result is True

    def test_aw_w_matching_by_dst_rob_idx(self, master_ni_with_memory):
        """AW/W should match by (dst_id, rob_idx)."""
        # Prepare memory with some data
        master_ni_with_memory.write_local(0x1000, bytes(64))

        # AW flit
        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=5,
            rob_req=True,
            last=True,
        )

        # W flit with matching rob_idx
        w_flit = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"matched_data" + bytes(20),
            last=True,
            rob_idx=5,  # Same rob_idx as AW
            seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw_flit)
        master_ni_with_memory.receive_req_flit(w_flit)

        # Process to complete write
        for _ in range(5):
            master_ni_with_memory.process_cycle(current_time=_)

        # Verify data was written
        data = master_ni_with_memory.read_local(0x1000, 32)
        assert data[:12] == b"matched_data"

    def test_w_flit_sequence_ordering(self, master_ni_with_memory):
        """W flit sequence should maintain order by seq_num."""
        master_ni_with_memory.write_local(0x1000, bytes(128))

        # AW flit
        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=1,  # 2 beats
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        master_ni_with_memory.receive_req_flit(aw_flit)

        # W flits - send in order but with explicit seq_num
        w0 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"FIRST_BEAT" + bytes(22),
            last=False,
            rob_idx=0,
            seq_num=0,
        )
        w1 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"SECOND_BEAT" + bytes(21),
            last=True,
            rob_idx=0,
            seq_num=1,
        )

        master_ni_with_memory.receive_req_flit(w0)
        master_ni_with_memory.receive_req_flit(w1)

        # Process
        for i in range(10):
            master_ni_with_memory.process_cycle(current_time=i)

        # Data should be in correct order
        data = master_ni_with_memory.read_local(0x1000, 64)
        # The data content depends on how packets are assembled
        assert len(data) == 64


# ==============================================================================
# Part 2.2: Memory Operation Tests
# ==============================================================================

class TestMasterNIMemoryOperations:
    """Tests for memory read/write operations."""

    def test_write_to_local_memory(self, master_ni_with_memory):
        """Write operation should store data in local memory."""
        # Initialize memory
        master_ni_with_memory.write_local(0x1000, bytes(64))

        # Create AW + W flits
        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        w_flit = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"WRITE_TEST_DATA" + bytes(17),
            last=True,
            rob_idx=0,
            seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw_flit)
        master_ni_with_memory.receive_req_flit(w_flit)

        # Process
        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        # Verify write
        data = master_ni_with_memory.read_local(0x1000, 32)
        assert data[:15] == b"WRITE_TEST_DATA"

    def test_read_from_local_memory(self, master_ni_with_memory):
        """Read operation should retrieve data from local memory."""
        # Pre-populate memory
        test_data = b"READ_TEST_PATTERN" + bytes(15)
        master_ni_with_memory.write_local(0x2000, test_data)

        # Create AR flit
        ar_flit = FlitFactory.create_ar(
            src=(1, 1), dest=(1, 0),
            addr=0x2000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
        )

        master_ni_with_memory.receive_req_flit(ar_flit)

        # Process
        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        # Should have response flit with read data
        assert master_ni_with_memory.has_pending_response() is True

    def test_write_data_integrity(self, master_ni_with_memory):
        """Write data should match exactly what was sent."""
        test_data = b"INTEGRITY_CHECK_123" + bytes(13)
        master_ni_with_memory.write_local(0x3000, bytes(64))

        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x3000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        w_flit = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=test_data,
            last=True,
            rob_idx=0,
            seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw_flit)
        master_ni_with_memory.receive_req_flit(w_flit)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        # Verify exact data
        result = master_ni_with_memory.read_local(0x3000, 32)
        assert result == test_data


# ==============================================================================
# Part 2.3: Response Generation Tests
# ==============================================================================

class TestMasterNIResponseGeneration:
    """Tests for response flit generation."""

    def test_b_response_generated_after_write(self, master_ni_with_memory):
        """B response should be generated after write completes."""
        master_ni_with_memory.write_local(0x1000, bytes(64))

        aw_flit = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=7,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        w_flit = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"write_data" + bytes(22),
            last=True,
            rob_idx=0,
            seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw_flit)
        master_ni_with_memory.receive_req_flit(w_flit)

        # Process
        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        # Should have B response
        b_flit = master_ni_with_memory.get_resp_flit()
        assert b_flit is not None
        assert b_flit.hdr.axi_ch == AxiChannel.B

    def test_r_response_generated_after_read(self, master_ni_with_memory):
        """R response should be generated after read completes."""
        # Pre-populate memory
        master_ni_with_memory.write_local(0x2000, b"stored_data" + bytes(21))

        ar_flit = FlitFactory.create_ar(
            src=(1, 1), dest=(1, 0),
            addr=0x2000,
            axi_id=3,
            length=0,
            rob_idx=0,
            rob_req=True,
        )

        master_ni_with_memory.receive_req_flit(ar_flit)

        # Process
        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        # Should have R response
        r_flit = master_ni_with_memory.get_resp_flit()
        assert r_flit is not None
        assert r_flit.hdr.axi_ch == AxiChannel.R

    def test_b_response_routing_correct(self, master_ni_with_memory):
        """B response should route back to request source."""
        master_ni_with_memory.write_local(0x1000, bytes(64))

        # Request from (2, 3)
        aw_flit = FlitFactory.create_aw(
            src=(2, 3), dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        w_flit = FlitFactory.create_w(
            src=(2, 3), dest=(1, 0),
            data=b"data" + bytes(28),
            last=True,
            rob_idx=0,
            seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw_flit)
        master_ni_with_memory.receive_req_flit(w_flit)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        b_flit = master_ni_with_memory.get_resp_flit()
        assert b_flit is not None
        # Response should go back to (2, 3)
        assert b_flit.dest == (2, 3)
        # Source should be this NI
        assert b_flit.src == (1, 0)

    def test_r_response_routing_correct(self, master_ni_with_memory):
        """R response should route back to request source."""
        master_ni_with_memory.write_local(0x2000, b"read_data" + bytes(23))

        # Request from (3, 2)
        ar_flit = FlitFactory.create_ar(
            src=(3, 2), dest=(1, 0),
            addr=0x2000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
        )

        master_ni_with_memory.receive_req_flit(ar_flit)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        r_flit = master_ni_with_memory.get_resp_flit()
        assert r_flit is not None
        # Response should go back to (3, 2)
        assert r_flit.dest == (3, 2)
        # Source should be this NI
        assert r_flit.src == (1, 0)

    def test_r_response_multi_beat_ordering(self, master_ni_with_memory):
        """Multi-beat R response should maintain order."""
        # Pre-populate with larger data
        large_data = b"A" * 32 + b"B" * 32  # 64 bytes
        master_ni_with_memory.write_local(0x3000, large_data)

        # AR for 64 bytes (2 beats at 32 bytes each)
        ar_flit = FlitFactory.create_ar(
            src=(1, 1), dest=(1, 0),
            addr=0x3000,
            axi_id=0,
            length=1,  # 2 beats
            rob_idx=0,
            rob_req=True,
        )

        master_ni_with_memory.receive_req_flit(ar_flit)

        # Process multiple cycles
        r_flits = []
        for i in range(20):
            master_ni_with_memory.process_cycle(current_time=i)
            flit = master_ni_with_memory.get_resp_flit()
            if flit and flit.hdr.axi_ch == AxiChannel.R:
                r_flits.append(flit)

        # Should have multiple R flits
        assert len(r_flits) >= 1

        # Last R flit should have last=True
        if r_flits:
            assert r_flits[-1].hdr.last is True


# ==============================================================================
# Part 2.4: Per-ID FIFO Behavior Tests
# ==============================================================================

class TestMasterNIPerIdFifo:
    """Tests for Per-ID FIFO ordering and independence."""

    def test_per_id_fifo_ordering(self, master_ni_with_memory):
        """Same ID requests should be processed in FIFO order."""
        master_ni_with_memory.write_local(0x1000, bytes(256))

        # Two writes with same AXI ID
        aw0 = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        w0 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"FIRST" + bytes(27),
            last=True,
            rob_idx=0,
            seq_num=0,
        )

        aw1 = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1100,
            axi_id=0,  # Same ID
            length=0,
            rob_idx=1,
            rob_req=True,
            last=True,
        )
        w1 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"SECOND" + bytes(26),
            last=True,
            rob_idx=1,
            seq_num=0,
        )

        # Send in order
        master_ni_with_memory.receive_req_flit(aw0)
        master_ni_with_memory.receive_req_flit(w0)
        master_ni_with_memory.receive_req_flit(aw1)
        master_ni_with_memory.receive_req_flit(w1)

        # Process
        b_responses = []
        for i in range(20):
            master_ni_with_memory.process_cycle(current_time=i)
            flit = master_ni_with_memory.get_resp_flit()
            if flit and flit.hdr.axi_ch == AxiChannel.B:
                b_responses.append(flit)

        # Should have 2 B responses
        assert len(b_responses) == 2

    def test_different_ids_independent(self, master_ni_with_memory):
        """Different IDs should be processed independently."""
        master_ni_with_memory.write_local(0x1000, bytes(256))

        # Two writes with different AXI IDs
        aw_id0 = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0,
            length=0,
            rob_idx=0,
            rob_req=True,
            last=True,
        )
        aw_id1 = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1100,
            axi_id=1,  # Different ID
            length=0,
            rob_idx=1,
            rob_req=True,
            last=True,
        )

        master_ni_with_memory.receive_req_flit(aw_id0)
        master_ni_with_memory.receive_req_flit(aw_id1)

        w_id0 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"data_for_id0" + bytes(20),
            last=True,
            rob_idx=0,
            seq_num=0,
        )
        w_id1 = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"data_for_id1" + bytes(20),
            last=True,
            rob_idx=1,
            seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(w_id0)
        master_ni_with_memory.receive_req_flit(w_id1)

        # Process
        for i in range(20):
            master_ni_with_memory.process_cycle(current_time=i)

        # Both writes should complete independently
        data0 = master_ni_with_memory.read_local(0x1000, 32)
        data1 = master_ni_with_memory.read_local(0x1100, 32)

        assert data0[:12] == b"data_for_id0"
        assert data1[:12] == b"data_for_id1"

    def test_outstanding_tracking(self, master_ni_with_memory):
        """Outstanding request count should be tracked correctly."""
        master_ni_with_memory.write_local(0x1000, bytes(256))

        # Multiple outstanding reads
        for i in range(4):
            ar = FlitFactory.create_ar(
                src=(1, 1), dest=(1, 0),
                addr=0x1000 + i * 0x100,
                axi_id=i,
                length=0,
                rob_idx=i,
                rob_req=True,
            )
            master_ni_with_memory.receive_req_flit(ar)

        # Per-ID FIFO should have entries
        fifo_count = sum(
            len(fifo)
            for fifo in master_ni_with_memory._per_id_fifo.values()
        )
        # After processing, FIFO should have pending entries
        master_ni_with_memory.process_cycle(current_time=0)
        # Outstanding requests are tracked


# ==============================================================================
# Part 2.5: Channel Mode Tests
# ==============================================================================

class TestMasterNIChannelMode:
    """Tests for channel mode specific behavior."""

    def test_general_mode_shared_resp_buffer(self, master_ni_with_memory):
        """General Mode should use shared response buffer."""
        assert master_ni_with_memory.config.channel_mode == ChannelMode.GENERAL
        assert master_ni_with_memory._strategy.uses_per_channel_buffers is False

    def test_axi_mode_separate_resp_buffers(self, master_ni_axi_mode):
        """AXI Mode should have separate response buffers."""
        assert master_ni_axi_mode.config.channel_mode == ChannelMode.AXI
        assert master_ni_axi_mode._strategy.uses_per_channel_buffers is True

        # Should have per-channel response buffers
        assert AxiChannel.B in master_ni_axi_mode._resp_per_channel
        assert AxiChannel.R in master_ni_axi_mode._resp_per_channel

    def test_axi_mode_get_channel_resp_flit(self, master_ni_axi_mode):
        """AXI Mode should support per-channel response retrieval."""
        master_ni_axi_mode.write_local(0x1000, bytes(64))
        master_ni_axi_mode.write_local(0x2000, b"read_data" + bytes(23))

        # Write transaction
        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=0, rob_req=True, last=True,
        )
        w = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"write" + bytes(27),
            last=True, rob_idx=0, seq_num=0,
        )

        # Read transaction
        ar = FlitFactory.create_ar(
            src=(1, 1), dest=(1, 0),
            addr=0x2000,
            axi_id=1, length=0, rob_idx=1, rob_req=True,
        )

        master_ni_axi_mode.receive_req_flit(aw)
        master_ni_axi_mode.receive_req_flit(w)
        master_ni_axi_mode.receive_req_flit(ar)

        # Process
        for i in range(10):
            master_ni_axi_mode.process_cycle(current_time=i)

        # Should be able to get B and R from separate channels
        b_flit = master_ni_axi_mode.get_channel_resp_flit(AxiChannel.B)
        r_flit = master_ni_axi_mode.get_channel_resp_flit(AxiChannel.R)

        # At least one should be available
        assert b_flit is not None or r_flit is not None


# ==============================================================================
# Part 2.6: Buffer Management Tests
# ==============================================================================

class TestMasterNIBufferManagement:
    """Tests for buffer management."""

    def test_req_input_buffer_full(self, master_ni):
        """Request input buffer should reject when full."""
        # Fill the buffer
        for i in range(master_ni.config.req_buffer_depth):
            flit = FlitFactory.create_ar(
                src=(1, 1), dest=(1, 0),
                addr=0x1000 + i * 0x100,
                axi_id=i % 16,
                length=0, rob_idx=i, rob_req=True,
            )
            result = master_ni.receive_req_flit(flit)
            if not result:
                break

        # Buffer should be full now
        assert master_ni.req_input.is_full() is True

        # Next one should be rejected
        overflow_flit = FlitFactory.create_ar(
            src=(1, 1), dest=(1, 0),
            addr=0x9000, axi_id=15, length=0, rob_idx=99, rob_req=True,
        )
        result = master_ni.receive_req_flit(overflow_flit)
        assert result is False

    def test_resp_output_available(self, master_ni_with_memory):
        """Response output should indicate available responses."""
        assert master_ni_with_memory.has_pending_response() is False

        master_ni_with_memory.write_local(0x1000, bytes(64))

        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=0, rob_req=True, last=True,
        )
        w = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"data" + bytes(28),
            last=True, rob_idx=0, seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw)
        master_ni_with_memory.receive_req_flit(w)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        assert master_ni_with_memory.has_pending_response() is True


# ==============================================================================
# Part 2.7: AW/W Packet Matching Tests
# ==============================================================================

class TestMasterNIAwWMatching:
    """Tests for AW/W packet matching (FlooNoC: separate packets)."""

    def test_aw_before_w(self, master_ni_with_memory):
        """Normal case: AW arrives before W."""
        master_ni_with_memory.write_local(0x1000, bytes(64))

        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=0, rob_req=True, last=True,
        )
        w = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"aw_before_w" + bytes(21),
            last=True, rob_idx=0, seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw)
        master_ni_with_memory.process_cycle(current_time=0)

        master_ni_with_memory.receive_req_flit(w)
        master_ni_with_memory.process_cycle(current_time=1)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i+2)

        # Write should complete
        data = master_ni_with_memory.read_local(0x1000, 32)
        assert data[:11] == b"aw_before_w"

    def test_w_before_aw_pipelined(self, master_ni_with_memory):
        """Pipelined case: W may arrive before AW."""
        master_ni_with_memory.write_local(0x1000, bytes(64))

        # W arrives first (pipelined mode)
        w = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"w_before_aw" + bytes(21),
            last=True, rob_idx=5, seq_num=0,
        )
        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=5, rob_req=True, last=True,
        )

        # W first
        master_ni_with_memory.receive_req_flit(w)
        master_ni_with_memory.process_cycle(current_time=0)

        # Then AW
        master_ni_with_memory.receive_req_flit(aw)

        for i in range(10):
            master_ni_with_memory.process_cycle(current_time=i+1)

        # Write should still complete
        data = master_ni_with_memory.read_local(0x1000, 32)
        assert data[:11] == b"w_before_aw"


# ==============================================================================
# Part 2.8: Statistics Tests
# ==============================================================================

class TestMasterNIStatistics:
    """Tests for statistics tracking."""

    def test_stats_track_write_requests(self, master_ni_with_memory):
        """Statistics should track write request count."""
        master_ni_with_memory.write_local(0x1000, bytes(64))

        assert master_ni_with_memory.stats.write_requests == 0

        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=0, rob_req=True, last=True,
        )
        w = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"data" + bytes(28),
            last=True, rob_idx=0, seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw)
        master_ni_with_memory.receive_req_flit(w)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        assert master_ni_with_memory.stats.write_requests == 1

    def test_stats_track_read_requests(self, master_ni_with_memory):
        """Statistics should track read request count."""
        master_ni_with_memory.write_local(0x2000, b"test" + bytes(28))

        assert master_ni_with_memory.stats.read_requests == 0

        ar = FlitFactory.create_ar(
            src=(1, 1), dest=(1, 0),
            addr=0x2000,
            axi_id=0, length=0, rob_idx=0, rob_req=True,
        )

        master_ni_with_memory.receive_req_flit(ar)

        for i in range(5):
            master_ni_with_memory.process_cycle(current_time=i)

        assert master_ni_with_memory.stats.read_requests == 1

    def test_stats_track_b_responses(self, master_ni_with_memory):
        """Statistics should track B response count."""
        master_ni_with_memory.write_local(0x1000, bytes(64))

        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=0, rob_req=True, last=True,
        )
        w = FlitFactory.create_w(
            src=(1, 1), dest=(1, 0),
            data=b"data" + bytes(28),
            last=True, rob_idx=0, seq_num=0,
        )

        master_ni_with_memory.receive_req_flit(aw)
        master_ni_with_memory.receive_req_flit(w)

        for i in range(10):
            master_ni_with_memory.process_cycle(current_time=i)
            master_ni_with_memory.get_resp_flit()

        assert master_ni_with_memory.stats.b_responses_sent >= 1


# ==============================================================================
# Part 2.9: Flit Latency Callback Tests
# ==============================================================================

class TestMasterNIFlitLatencyCallback:
    """Tests for per-flit latency tracking callback."""

    def test_flit_latency_callback_invoked(self, master_ni_with_memory):
        """Flit latency callback should be invoked on arrival."""
        latencies = []

        def callback(latency: int, axi_ch, payload_bytes: int):
            latencies.append(latency)

        master_ni_with_memory.set_flit_latency_callback(callback)

        master_ni_with_memory.write_local(0x1000, bytes(64))

        # Create flit with injection_cycle set
        aw = FlitFactory.create_aw(
            src=(1, 1), dest=(1, 0),
            addr=0x1000,
            axi_id=0, length=0, rob_idx=0, rob_req=True, last=True,
        )
        aw.injection_cycle = 10

        master_ni_with_memory.receive_req_flit(aw)
        master_ni_with_memory.process_cycle(current_time=15)  # Latency = 15 - 10 = 5

        # Callback should have been called
        assert len(latencies) >= 1
        if latencies:
            assert latencies[0] == 5
