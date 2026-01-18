"""
Tests for WormholeArbiter packet-level locking mechanism.

Tests verify:
1. Lock acquisition and release
2. Lock conflict handling
3. Arbitration logic (locked paths priority + round-robin)
4. Multiple independent locks
"""

import pytest
from src.core.router import Direction, WormholeArbiter
from src.core.flit import FlitFactory, AxiChannel, Flit


class TestWormholeArbiterInitialState:
    """Test initial state of WormholeArbiter."""

    def test_initial_state_no_locks(self, wormhole_arbiter):
        """All ports should be unlocked initially."""
        for d in Direction:
            assert not wormhole_arbiter.is_output_locked(d)
            assert not wormhole_arbiter.is_input_locked(d)

    def test_initial_lock_status_all_none(self, wormhole_arbiter):
        """Lock status should show all None."""
        status = wormhole_arbiter.get_lock_status()

        assert all(v is None for v in status["output_lock"].values())
        assert all(v is None for v in status["input_lock"].values())


class TestWormholeArbiterLocking:
    """Test lock acquisition and release."""

    def test_lock_acquire_success(self, wormhole_arbiter):
        """Lock acquisition should succeed on free port."""
        success = wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        assert success is True
        assert wormhole_arbiter.is_output_locked(Direction.EAST)
        assert wormhole_arbiter.is_input_locked(Direction.NORTH)
        assert wormhole_arbiter.get_locked_output(Direction.NORTH) == Direction.EAST
        assert wormhole_arbiter.get_lock_holder(Direction.EAST) == Direction.NORTH

    def test_lock_acquire_fail_output_locked(self, wormhole_arbiter):
        """Lock acquisition should fail if output is locked by another input."""
        # First lock: NORTH -> EAST
        wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        # Second lock attempt: SOUTH -> EAST (same output, different input)
        success = wormhole_arbiter.lock(Direction.SOUTH, Direction.EAST)

        assert success is False
        # Original lock should be unchanged
        assert wormhole_arbiter.get_lock_holder(Direction.EAST) == Direction.NORTH

    def test_lock_reacquire_same_input_same_output(self, wormhole_arbiter):
        """Re-acquiring the same lock should succeed (idempotent)."""
        wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        # Same input, same output
        success = wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        assert success is True

    def test_lock_release(self, wormhole_arbiter):
        """Lock release should free both input and output."""
        wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        wormhole_arbiter.release(Direction.NORTH)

        assert not wormhole_arbiter.is_output_locked(Direction.EAST)
        assert not wormhole_arbiter.is_input_locked(Direction.NORTH)
        assert wormhole_arbiter.get_locked_output(Direction.NORTH) is None
        assert wormhole_arbiter.get_lock_holder(Direction.EAST) is None

    def test_release_unlocked_input_no_error(self, wormhole_arbiter):
        """Releasing an unlocked input should not raise error."""
        # Should not raise
        wormhole_arbiter.release(Direction.NORTH)

        # State should remain unchanged
        assert not wormhole_arbiter.is_input_locked(Direction.NORTH)


class TestWormholeArbiterMultipleLocks:
    """Test multiple independent locks."""

    def test_multiple_independent_locks(self, wormhole_arbiter):
        """Multiple non-conflicting locks should succeed."""
        success1 = wormhole_arbiter.lock(Direction.NORTH, Direction.SOUTH)
        success2 = wormhole_arbiter.lock(Direction.EAST, Direction.WEST)
        success3 = wormhole_arbiter.lock(Direction.LOCAL, Direction.NORTH)

        assert success1 and success2 and success3

        # Verify all locks
        assert wormhole_arbiter.get_locked_output(Direction.NORTH) == Direction.SOUTH
        assert wormhole_arbiter.get_locked_output(Direction.EAST) == Direction.WEST
        assert wormhole_arbiter.get_locked_output(Direction.LOCAL) == Direction.NORTH

    def test_partial_release(self, wormhole_arbiter):
        """Releasing one lock should not affect others."""
        wormhole_arbiter.lock(Direction.NORTH, Direction.SOUTH)
        wormhole_arbiter.lock(Direction.EAST, Direction.WEST)

        wormhole_arbiter.release(Direction.NORTH)

        # NORTH -> SOUTH released
        assert not wormhole_arbiter.is_input_locked(Direction.NORTH)
        assert not wormhole_arbiter.is_output_locked(Direction.SOUTH)

        # EAST -> WEST still active
        assert wormhole_arbiter.is_input_locked(Direction.EAST)
        assert wormhole_arbiter.is_output_locked(Direction.WEST)


class TestWormholeArbiterArbitration:
    """Test arbitration logic."""

    def test_arbitrate_single_request(self, wormhole_arbiter, single_flit_factory):
        """Single request should always be granted."""
        flit = single_flit_factory(src=(0, 0), dest=(4, 2))
        requests = {
            Direction.WEST: (flit, Direction.EAST)
        }

        grants = wormhole_arbiter.arbitrate(requests)

        assert len(grants) == 1
        assert grants[0] == (Direction.WEST, Direction.EAST, flit)

    def test_arbitrate_honors_existing_locks(self, wormhole_arbiter, single_flit_factory):
        """Locked paths should be granted immediately (priority)."""
        # Pre-lock NORTH -> EAST
        wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        flit_north = single_flit_factory(src=(0, 0), dest=(4, 2))
        flit_south = single_flit_factory(src=(0, 0), dest=(4, 2))

        requests = {
            Direction.NORTH: (flit_north, Direction.EAST),  # Locked path
            Direction.SOUTH: (flit_south, Direction.EAST),  # Competing for same output
        }

        grants = wormhole_arbiter.arbitrate(requests)

        # Only the locked input should be granted
        assert len(grants) == 1
        assert grants[0][0] == Direction.NORTH
        assert grants[0][1] == Direction.EAST

    def test_arbitrate_multiple_outputs_no_conflict(
        self, wormhole_arbiter, single_flit_factory
    ):
        """Requests for different outputs should all be granted."""
        flit1 = single_flit_factory(src=(0, 0), dest=(4, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 3))

        requests = {
            Direction.WEST: (flit1, Direction.EAST),
            Direction.SOUTH: (flit2, Direction.NORTH),
        }

        grants = wormhole_arbiter.arbitrate(requests)

        assert len(grants) == 2

        # Check both grants are present (order may vary)
        grant_pairs = {(g[0], g[1]) for g in grants}
        assert (Direction.WEST, Direction.EAST) in grant_pairs
        assert (Direction.SOUTH, Direction.NORTH) in grant_pairs

    def test_arbitrate_conflict_uses_round_robin(self, wormhole_arbiter, single_flit_factory):
        """Conflicting requests should use round-robin priority."""
        flit1 = single_flit_factory(src=(0, 0), dest=(4, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(4, 2))

        requests = {
            Direction.NORTH: (flit1, Direction.EAST),
            Direction.SOUTH: (flit2, Direction.EAST),  # Same output
        }

        grants = wormhole_arbiter.arbitrate(requests)

        # Only one should win
        assert len(grants) == 1
        assert grants[0][1] == Direction.EAST

    def test_arbitrate_blocked_by_locked_output(self, wormhole_arbiter, single_flit_factory):
        """Request for locked output (by another input) should be blocked."""
        # NORTH locks EAST
        wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)

        flit = single_flit_factory(src=(0, 0), dest=(4, 2))

        # SOUTH wants EAST but NORTH has it locked
        requests = {
            Direction.SOUTH: (flit, Direction.EAST),
        }

        grants = wormhole_arbiter.arbitrate(requests)

        # SOUTH should be blocked
        assert len(grants) == 0


class TestWormholeArbiterReset:
    """Test reset functionality."""

    def test_reset_clears_all_locks(self, wormhole_arbiter):
        """Reset should clear all locks."""
        wormhole_arbiter.lock(Direction.NORTH, Direction.EAST)
        wormhole_arbiter.lock(Direction.SOUTH, Direction.WEST)
        wormhole_arbiter.lock(Direction.LOCAL, Direction.NORTH)

        wormhole_arbiter.reset()

        for d in Direction:
            assert not wormhole_arbiter.is_input_locked(d)
            assert not wormhole_arbiter.is_output_locked(d)


class TestWormholeArbiterPacketScenarios:
    """Test realistic packet transfer scenarios."""

    def test_head_body_tail_sequence(self, wormhole_arbiter, multi_flit_packet_factory):
        """Test HEAD locks, BODY follows, TAIL releases."""
        flits = multi_flit_packet_factory(src=(0, 0), dest=(4, 2), num_flits=3)
        head, body, tail = flits

        # HEAD arrives and wins arbitration
        requests_head = {Direction.WEST: (head, Direction.EAST)}
        grants_head = wormhole_arbiter.arbitrate(requests_head)
        assert len(grants_head) == 1

        # Lock for HEAD
        wormhole_arbiter.lock(Direction.WEST, Direction.EAST)
        assert wormhole_arbiter.is_input_locked(Direction.WEST)

        # BODY arrives - should follow locked path
        requests_body = {Direction.WEST: (body, Direction.EAST)}
        grants_body = wormhole_arbiter.arbitrate(requests_body)
        assert len(grants_body) == 1
        assert grants_body[0][0] == Direction.WEST

        # TAIL arrives - should follow locked path
        requests_tail = {Direction.WEST: (tail, Direction.EAST)}
        grants_tail = wormhole_arbiter.arbitrate(requests_tail)
        assert len(grants_tail) == 1

        # After TAIL, release lock
        wormhole_arbiter.release(Direction.WEST)
        assert not wormhole_arbiter.is_input_locked(Direction.WEST)
        assert not wormhole_arbiter.is_output_locked(Direction.EAST)

    def test_interleaved_packets_blocked(
        self, wormhole_arbiter, multi_flit_packet_factory
    ):
        """
        Test that interleaved packets are blocked.

        Scenario:
        - Packet A (WEST -> EAST) starts with HEAD
        - Packet B (SOUTH -> EAST) tries to interleave - should be blocked
        """
        packet_a = multi_flit_packet_factory(src=(0, 0), dest=(4, 2), num_flits=3)
        packet_b = multi_flit_packet_factory(src=(0, 0), dest=(4, 2), num_flits=2)

        head_a = packet_a[0]
        head_b = packet_b[0]

        # Packet A HEAD wins
        requests1 = {Direction.WEST: (head_a, Direction.EAST)}
        grants1 = wormhole_arbiter.arbitrate(requests1)
        assert len(grants1) == 1

        # Lock for packet A
        wormhole_arbiter.lock(Direction.WEST, Direction.EAST)

        # Packet B HEAD tries to use same output - should be blocked
        requests2 = {Direction.SOUTH: (head_b, Direction.EAST)}
        grants2 = wormhole_arbiter.arbitrate(requests2)

        # Packet B should be blocked (EAST is locked by WEST)
        assert len(grants2) == 0

    def test_parallel_packets_different_outputs(
        self, wormhole_arbiter, multi_flit_packet_factory
    ):
        """Two packets to different outputs can proceed in parallel."""
        packet_a = multi_flit_packet_factory(src=(0, 0), dest=(4, 2), num_flits=2)
        packet_b = multi_flit_packet_factory(src=(0, 0), dest=(2, 3), num_flits=2)

        head_a = packet_a[0]
        head_b = packet_b[0]

        # Both packets request different outputs
        requests = {
            Direction.WEST: (head_a, Direction.EAST),
            Direction.SOUTH: (head_b, Direction.NORTH),
        }

        grants = wormhole_arbiter.arbitrate(requests)

        # Both should be granted
        assert len(grants) == 2

        # Lock both paths
        wormhole_arbiter.lock(Direction.WEST, Direction.EAST)
        wormhole_arbiter.lock(Direction.SOUTH, Direction.NORTH)

        # Both should be locked
        assert wormhole_arbiter.is_input_locked(Direction.WEST)
        assert wormhole_arbiter.is_input_locked(Direction.SOUTH)
