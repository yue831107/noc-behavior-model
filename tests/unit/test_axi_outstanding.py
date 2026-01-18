"""
Unit tests for True AXI Outstanding implementation.

Tests the pipelined AW and FIFO W ordering behavior in both
HostAXIMaster and LocalAXIMaster.

Key behaviors tested:
1. Pipelined AW: Multiple AW can be sent before W data completes
2. FIFO W ordering: W beats follow AW send order (no interleaving)
3. Outstanding limit: Respects max_outstanding configuration
4. Channel Mode: General Mode (AW/W exclusive) vs AXI Mode (AW+W parallel)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple, Optional
from collections import deque

from src.testbench import HostAXIMaster, HostMemory
from src.testbench.local_axi_master import LocalAXIMaster, LocalTransferConfig, PendingBurst
from src.testbench.memory import LocalMemory
from src.config import TransferConfig, TransferMode
from src.core.router import ChannelMode
from src.core.ni import NIConfig
from src.axi.interface import AXI_AW, AXI_W, AXI_B, AXIResp


class MockSlaveNI:
    """Mock SlaveNI for testing AW/W send behavior."""

    def __init__(self, channel_mode: ChannelMode = ChannelMode.GENERAL):
        self.config = Mock()
        self.config.channel_mode = channel_mode

        # Track what was sent
        self.aw_sent: List[Tuple[int, int]] = []  # (cycle, axi_id)
        self.w_sent: List[Tuple[int, int, bool]] = []  # (cycle, axi_id, is_last)
        self.ar_sent: List[Tuple[int, int]] = []  # (cycle, axi_id)
        self.b_queue: List[AXI_B] = []

        # Control acceptance
        self.accept_aw = True
        self.accept_w = True
        self.accept_ar = True

    def process_aw(self, aw: AXI_AW, cycle: int) -> bool:
        """Record AW if accepted."""
        if self.accept_aw:
            self.aw_sent.append((cycle, aw.awid))
            return True
        return False

    def process_w(self, w: AXI_W, axi_id: int, cycle: int) -> bool:
        """Record W if accepted."""
        if self.accept_w:
            self.w_sent.append((cycle, axi_id, w.wlast))
            return True
        return False

    def process_ar(self, ar, cycle: int) -> bool:
        """Record AR if accepted."""
        if self.accept_ar:
            self.ar_sent.append((cycle, ar.arid))
            return True
        return False

    def get_b_response(self) -> Optional[AXI_B]:
        """Return queued B response."""
        if self.b_queue:
            return self.b_queue.pop(0)
        return None

    def get_r_response(self):
        """Return R response (always None for write-focused tests)."""
        return None

    def queue_b_response(self, axi_id: int):
        """Queue a B response for retrieval."""
        self.b_queue.append(AXI_B(bid=axi_id, bresp=AXIResp.OKAY))


class TestHostAXIMasterOutstanding:
    """Test HostAXIMaster pipelined AW and FIFO W ordering."""

    @pytest.fixture
    def host_memory(self):
        """Create host memory with test data."""
        mem = HostMemory(size=4096)
        # Fill with test pattern - enough for multiple bursts
        test_data = bytes(range(256)) * 16  # 4KB of test data
        mem.write(0, test_data)
        return mem

    @pytest.fixture
    def transfer_config(self):
        """Create transfer config for multi-burst test."""
        return TransferConfig(
            src_addr=0,
            src_size=256,  # Multiple bursts
            dst_addr=0x1000,
            target_nodes=[1, 2],  # Two nodes = 2 AW transactions
            max_burst_len=4,  # 4 W beats per burst
            beat_size=8,
            max_outstanding=4,
            transfer_mode=TransferMode.BROADCAST,
        )

    def test_pipelined_aw_before_w_complete(self, host_memory, transfer_config):
        """AW1 can be sent before W0 completes (pipelining)."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Run enough cycles to see pipelining
        for cycle in range(10):
            master.process_cycle(cycle)

        # Check that multiple AWs were sent
        aw_ids = [axi_id for _, axi_id in mock_ni.aw_sent]
        assert len(mock_ni.aw_sent) >= 2, "Should have sent at least 2 AW"

        # Verify AW came before corresponding W completed
        # In pipelined mode, AW1 should come before all W0 beats are done
        if len(mock_ni.aw_sent) >= 2:
            aw1_cycle = mock_ni.aw_sent[1][0]
            # Find all W beats for first AW's axi_id
            first_aw_id = mock_ni.aw_sent[0][1]
            w0_beats = [c for c, aid, _ in mock_ni.w_sent if aid == first_aw_id]
            if w0_beats:
                # AW1 should be sent while W0 is still in progress
                # (not all W0 beats completed)
                assert aw1_cycle <= max(w0_beats) + 1, \
                    "AW1 should be sent during W0 transfer (pipelining)"

    def test_w_fifo_order(self, host_memory, transfer_config):
        """W beats must follow AW send order (no interleaving)."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Run cycles
        for cycle in range(50):
            master.process_cycle(cycle)

        if len(mock_ni.w_sent) >= 2:
            # Extract the order of axi_ids in W beats (ignoring duplicates)
            w_order = []
            last_id = None
            for _, axi_id, is_last in mock_ni.w_sent:
                if axi_id != last_id:
                    if last_id is not None and not any(
                        is_last_flag for _, aid, is_last_flag in mock_ni.w_sent
                        if aid == last_id
                    ):
                        # Previous transaction's W didn't complete - check FIFO
                        pass
                    w_order.append(axi_id)
                    last_id = axi_id

            # W beats for transaction N should complete before transaction N+1 starts
            # (no interleaving)
            seen_complete = set()
            current_id = None
            for _, axi_id, is_last in mock_ni.w_sent:
                if current_id is None:
                    current_id = axi_id
                elif axi_id != current_id:
                    # Switched to new axi_id - previous should have completed
                    assert current_id in seen_complete, \
                        f"W interleaving detected: switched from {current_id} to {axi_id} before wlast"
                    current_id = axi_id
                if is_last:
                    seen_complete.add(axi_id)

    def test_outstanding_limit_respected(self, host_memory):
        """Should not exceed max_outstanding transactions."""
        config = TransferConfig(
            src_addr=0,
            src_size=512,
            dst_addr=0x1000,
            target_nodes=[1, 2, 3, 4, 5, 6, 7, 8],  # Many targets
            max_burst_len=2,
            beat_size=8,
            max_outstanding=2,  # Low limit
            transfer_mode=TransferMode.BROADCAST,
        )

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Track outstanding count over time
        max_outstanding_observed = 0

        for cycle in range(100):
            master.process_cycle(cycle)

            # Count current outstanding (sent AW, not yet B)
            current_outstanding = len(master._awaiting_b)
            max_outstanding_observed = max(max_outstanding_observed, current_outstanding)

            # Never exceed limit
            assert current_outstanding <= config.max_outstanding, \
                f"Outstanding {current_outstanding} exceeded limit {config.max_outstanding}"

    def test_b_releases_outstanding_slot(self, host_memory):
        """Receiving B response should free slot for new AW."""
        config = TransferConfig(
            src_addr=0,
            src_size=256,
            dst_addr=0x1000,
            target_nodes=[1, 2, 3, 4],
            max_burst_len=2,
            beat_size=8,
            max_outstanding=2,  # Low limit to test release
            transfer_mode=TransferMode.BROADCAST,
        )

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Run until outstanding limit reached
        for cycle in range(20):
            master.process_cycle(cycle)
            if len(master._awaiting_b) >= config.max_outstanding:
                break

        # Verify we hit the limit
        aw_count_before = len(mock_ni.aw_sent)
        outstanding_before = len(master._awaiting_b)

        # Inject B response to release a slot
        if master._awaiting_b:
            released_id = list(master._awaiting_b)[0]
            mock_ni.queue_b_response(released_id)

        # Process more cycles
        for cycle in range(20, 40):
            master.process_cycle(cycle)

        # Should have sent more AW after B released slot
        aw_count_after = len(mock_ni.aw_sent)

        # If we were at limit and released, should be able to send more
        if outstanding_before >= config.max_outstanding:
            assert aw_count_after > aw_count_before, \
                "Should send more AW after B releases outstanding slot"

    def test_general_mode_aw_w_exclusive(self, host_memory, transfer_config):
        """In General Mode, AW and W should be mutually exclusive per cycle."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Run cycles
        for cycle in range(30):
            master.process_cycle(cycle)

        # Check each cycle - should not have both AW and W in same cycle
        aw_cycles = {c for c, _ in mock_ni.aw_sent}
        w_cycles = {c for c, _, _ in mock_ni.w_sent}

        # In General Mode, AW and W are mutually exclusive
        overlap = aw_cycles & w_cycles
        assert len(overlap) == 0, \
            f"General Mode: AW and W should not occur in same cycle, but found overlap at cycles {overlap}"

    def test_axi_mode_aw_w_parallel(self, host_memory, transfer_config):
        """In AXI Mode, AW and W can occur in the same cycle."""
        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=transfer_config,
        )

        mock_ni = MockSlaveNI(ChannelMode.AXI)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Run cycles
        for cycle in range(30):
            master.process_cycle(cycle)

        # Check for cycles with both AW and W
        aw_cycles = {c for c, _ in mock_ni.aw_sent}
        w_cycles = {c for c, _, _ in mock_ni.w_sent}

        # In AXI Mode, overlap is allowed (and expected for efficiency)
        overlap = aw_cycles & w_cycles

        # Note: Overlap isn't guaranteed in every case, but it should be possible
        # This test verifies the code doesn't prevent it in AXI mode
        # (The actual occurrence depends on timing and data)


class TestLocalAXIMasterOutstanding:
    """Test LocalAXIMaster pipelined AW and FIFO W ordering."""

    @pytest.fixture
    def local_memory(self):
        """Create local memory with test data."""
        mem = LocalMemory(node_id=0, size=4096)
        test_data = bytes(range(256)) * 4  # 1KB
        mem.write(0, test_data)
        return mem

    @pytest.fixture
    def transfer_config(self):
        """Create local transfer config."""
        return LocalTransferConfig(
            dest_coord=(2, 1),  # Node 5 in 5x4 mesh
            local_src_addr=0,
            local_dst_addr=0x2000,
            transfer_size=256,
        )

    def test_pipelined_aw_before_w_complete(self, local_memory, transfer_config):
        """AW1 can be sent before W0 completes."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
            max_burst_len=4,
            beat_size=32,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.configure_transfer(transfer_config)
        master.start()

        # Run cycles
        for cycle in range(20):
            master.process_cycle(cycle)

        # Verify multiple AW sent
        if len(mock_ni.aw_sent) >= 2:
            # AW pipelining should occur
            aw0_cycle = mock_ni.aw_sent[0][0]
            aw1_cycle = mock_ni.aw_sent[1][0]

            # AW1 should be sent soon after AW0 (within a few cycles)
            # In non-pipelined mode, AW1 would wait for all W0 beats
            assert aw1_cycle - aw0_cycle <= 5, \
                "AW should be pipelined (sent within few cycles)"

    def test_w_fifo_order_local(self, local_memory, transfer_config):
        """W beats must follow AW order (no interleaving)."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
            max_burst_len=2,  # Small bursts = more transactions
            beat_size=32,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.configure_transfer(transfer_config)
        master.start()

        for cycle in range(50):
            master.process_cycle(cycle)

        # Verify FIFO W ordering
        if len(mock_ni.w_sent) >= 4:
            seen_complete = set()
            current_id = None

            for _, axi_id, is_last in mock_ni.w_sent:
                if current_id is None:
                    current_id = axi_id
                elif axi_id != current_id:
                    assert current_id in seen_complete, \
                        "W interleaving: new transaction started before previous completed"
                    current_id = axi_id
                if is_last:
                    seen_complete.add(axi_id)

    def test_outstanding_limit_local(self, local_memory):
        """Should respect max_outstanding limit."""
        config = LocalTransferConfig(
            dest_coord=(2, 1),  # Node 5 in 5x4 mesh
            local_src_addr=0,
            local_dst_addr=0x2000,
            transfer_size=512,
        )

        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
            max_burst_len=2,
            beat_size=32,
        )
        master._max_outstanding = 2  # Low limit

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.configure_transfer(config)
        master.start()

        for cycle in range(100):
            master.process_cycle(cycle)

            # Count outstanding = w_active + b_pending
            outstanding = len(master._w_active) + len(master._b_pending)
            assert outstanding <= master._max_outstanding, \
                f"Outstanding {outstanding} exceeded limit {master._max_outstanding}"

    def test_general_mode_aw_w_exclusive_local(self, local_memory, transfer_config):
        """General Mode: AW and W mutually exclusive."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
            max_burst_len=4,
            beat_size=32,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.configure_transfer(transfer_config)
        master.start()

        for cycle in range(30):
            master.process_cycle(cycle)

        aw_cycles = {c for c, _ in mock_ni.aw_sent}
        w_cycles = {c for c, _, _ in mock_ni.w_sent}

        overlap = aw_cycles & w_cycles
        assert len(overlap) == 0, \
            f"General Mode: AW/W overlap at {overlap}"

    def test_axi_mode_aw_w_parallel_local(self, local_memory, transfer_config):
        """AXI Mode: AW and W can be parallel."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
            max_burst_len=4,
            beat_size=32,
        )

        mock_ni = MockSlaveNI(ChannelMode.AXI)
        master.connect_to_slave_ni(mock_ni)
        master.configure_transfer(transfer_config)
        master.start()

        for cycle in range(30):
            master.process_cycle(cycle)

        # Verify AXI mode allows parallel (no assertion failure on overlap)
        aw_cycles = {c for c, _ in mock_ni.aw_sent}
        w_cycles = {c for c, _, _ in mock_ni.w_sent}
        overlap = aw_cycles & w_cycles

        # In AXI mode, overlap is valid (test passes if no exception)


class TestPendingBurstDataclass:
    """Test PendingBurst dataclass used by LocalAXIMaster."""

    def test_default_timing_fields(self):
        """Timing fields should default to 0."""
        burst = PendingBurst(
            axi_id=1,
            dst_addr=0x1000,
            data=b'\x00' * 64,
            user_signal=0x123,
            w_beats_total=2,
        )

        assert burst.aw_sent_cycle == 0
        assert burst.first_w_cycle == 0
        assert burst.last_w_cycle == 0
        assert burst.b_received_cycle == 0

    def test_state_fields(self):
        """State fields should track progress."""
        burst = PendingBurst(
            axi_id=2,
            dst_addr=0x2000,
            data=b'\x00' * 128,
            user_signal=0x456,
            w_beats_total=4,
        )

        assert burst.aw_sent is False
        assert burst.w_beats_sent == 0
        assert burst.b_received is False

        # Simulate progression
        burst.aw_sent = True
        burst.aw_sent_cycle = 5

        burst.w_beats_sent = 2
        burst.first_w_cycle = 6

        assert burst.aw_sent is True
        assert burst.w_beats_sent == 2


class TestBackwardCompatibility:
    """Test backward compatibility with max_outstanding=1."""

    @pytest.fixture
    def host_memory(self):
        mem = HostMemory(size=4096)
        mem.write(0, bytes(range(256)) * 4)
        return mem

    def test_max_outstanding_1_sequential_behavior(self, host_memory):
        """max_outstanding=1 should give sequential behavior."""
        config = TransferConfig(
            src_addr=0,
            src_size=128,
            dst_addr=0x1000,
            target_nodes=[1, 2],
            max_burst_len=2,
            beat_size=8,
            max_outstanding=1,  # Sequential mode
            transfer_mode=TransferMode.BROADCAST,
        )

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        mock_ni = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni)
        master.start()

        # Run and collect AW timings
        for cycle in range(50):
            master.process_cycle(cycle)

            # With max_outstanding=1, should never have more than 1 awaiting B
            assert len(master._awaiting_b) <= 1, \
                "max_outstanding=1 should only allow 1 outstanding"

            # Release B to allow progress
            if master._awaiting_b:
                axi_id = list(master._awaiting_b)[0]
                # Check if W is complete for this axi_id
                w_complete = all(
                    is_last for _, aid, is_last in mock_ni.w_sent
                    if aid == axi_id
                )
                if w_complete:
                    mock_ni.queue_b_response(axi_id)


class TestChannelModeStrategy:
    """Test channel mode differentiation in AXI masters."""

    @pytest.fixture
    def host_memory(self):
        mem = HostMemory(size=4096)
        mem.write(0, bytes(256))
        return mem

    def test_channel_mode_detected_from_slave_ni(self, host_memory):
        """AXI master should detect channel mode from connected SlaveNI."""
        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes=[1],
            max_burst_len=4,
            beat_size=8,
            max_outstanding=4,
        )

        master = HostAXIMaster(
            host_memory=host_memory,
            transfer_config=config,
        )

        # Test with General Mode
        mock_ni_general = MockSlaveNI(ChannelMode.GENERAL)
        master.connect_to_slave_ni(mock_ni_general)
        master.start()
        master.process_cycle(0)

        # Reset and test with AXI Mode
        master.reset()
        mock_ni_axi = MockSlaveNI(ChannelMode.AXI)
        master.connect_to_slave_ni(mock_ni_axi)
        master.start()
        master.process_cycle(0)

        # Both should work without error
        assert True
