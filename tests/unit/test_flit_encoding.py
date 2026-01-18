"""
Flit Encoding Unit Tests.

Tests for FlitHeader bit-level encoding/decoding and node ID operations.
Focuses on edge cases, boundary values, and roundtrip preservation.

Usage:
    pytest tests/unit/test_flit_encoding.py -v
"""

import pytest

from src.core.flit import (
    FlitHeader,
    AxiChannel,
    encode_node_id,
    decode_node_id,
    X_BITS,
    Y_BITS,
    NODE_ID_BITS,
    ROB_IDX_BITS,
)


# =============================================================================
# Test: Node ID Encoding
# =============================================================================

class TestNodeIdEncoding:
    """Tests for encode_node_id / decode_node_id functions."""

    def test_encode_origin(self):
        """Encode (0, 0) should produce 0."""
        assert encode_node_id((0, 0)) == 0

    def test_encode_max_x(self):
        """Encode maximum x value."""
        max_x = (1 << X_BITS) - 1  # 7
        result = encode_node_id((max_x, 0))
        assert result == max_x << Y_BITS

    def test_encode_max_y(self):
        """Encode maximum y value."""
        max_y = (1 << Y_BITS) - 1  # 3
        result = encode_node_id((0, max_y))
        assert result == max_y

    def test_encode_max_both(self):
        """Encode maximum x and y values."""
        max_x = (1 << X_BITS) - 1  # 7
        max_y = (1 << Y_BITS) - 1  # 3
        result = encode_node_id((max_x, max_y))
        expected = (max_x << Y_BITS) | max_y  # 31
        assert result == expected

    def test_encode_typical_mesh_corner(self):
        """Encode typical mesh corner (4, 3)."""
        result = encode_node_id((4, 3))
        # (4 << 2) | 3 = 16 | 3 = 19
        assert result == 19

    @pytest.mark.parametrize("x,y", [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (4, 0), (5, 1), (6, 2), (7, 3),
    ])
    def test_roundtrip_preservation(self, x, y):
        """Encode then decode should preserve original values."""
        node_id = encode_node_id((x, y))
        decoded = decode_node_id(node_id)
        assert decoded == (x, y)

    def test_decode_zero(self):
        """Decode 0 should produce (0, 0)."""
        assert decode_node_id(0) == (0, 0)

    def test_decode_max_node_id(self):
        """Decode maximum valid node_id."""
        max_node_id = (1 << NODE_ID_BITS) - 1  # 31
        result = decode_node_id(max_node_id)
        max_x = (1 << X_BITS) - 1  # 7
        max_y = (1 << Y_BITS) - 1  # 3
        assert result == (max_x, max_y)


class TestNodeIdBoundaryErrors:
    """Tests for boundary error handling in node ID encoding."""

    def test_encode_negative_x_raises(self):
        """Encoding negative x should raise ValueError."""
        with pytest.raises(ValueError, match="x coordinate -1 out of range"):
            encode_node_id((-1, 0))

    def test_encode_negative_y_raises(self):
        """Encoding negative y should raise ValueError."""
        with pytest.raises(ValueError, match="y coordinate -1 out of range"):
            encode_node_id((0, -1))

    def test_encode_x_overflow_raises(self):
        """Encoding x > max should raise ValueError."""
        max_x = (1 << X_BITS) - 1  # 7
        with pytest.raises(ValueError, match=f"x coordinate {max_x + 1} out of range"):
            encode_node_id((max_x + 1, 0))

    def test_encode_y_overflow_raises(self):
        """Encoding y > max should raise ValueError."""
        max_y = (1 << Y_BITS) - 1  # 3
        with pytest.raises(ValueError, match=f"y coordinate {max_y + 1} out of range"):
            encode_node_id((0, max_y + 1))

    def test_decode_negative_raises(self):
        """Decoding negative node_id should raise ValueError."""
        with pytest.raises(ValueError, match="node_id -1 out of range"):
            decode_node_id(-1)

    def test_decode_overflow_raises(self):
        """Decoding node_id > max should raise ValueError."""
        max_node_id = (1 << NODE_ID_BITS) - 1  # 31
        with pytest.raises(ValueError, match=f"node_id {max_node_id + 1} out of range"):
            decode_node_id(max_node_id + 1)


# =============================================================================
# Test: FlitHeader Encoding
# =============================================================================

class TestFlitHeaderToInt:
    """Tests for FlitHeader.to_int() method."""

    def test_encode_all_zeros(self):
        """Header with all default values should encode to expected value."""
        hdr = FlitHeader()
        # Default: rob_req=False, rob_idx=0, dst_id=0, src_id=0, last=True, axi_ch=AW
        # last bit at position 16 = 0x10000
        assert hdr.to_int() == 0x10000

    def test_encode_rob_req_bit(self):
        """rob_req should set bit 0."""
        hdr = FlitHeader(rob_req=True, last=False)
        assert hdr.to_int() & 0x1 == 1

    def test_encode_rob_idx_field(self):
        """rob_idx should occupy bits 5:1."""
        hdr = FlitHeader(rob_idx=31, last=False)  # max 5-bit value
        result = hdr.to_int()
        rob_idx_extracted = (result >> 1) & 0x1F
        assert rob_idx_extracted == 31

    def test_encode_dst_id_field(self):
        """dst_id should occupy bits 10:6."""
        hdr = FlitHeader(dst_id=31, last=False)
        result = hdr.to_int()
        dst_id_extracted = (result >> 6) & 0x1F
        assert dst_id_extracted == 31

    def test_encode_src_id_field(self):
        """src_id should occupy bits 15:11."""
        hdr = FlitHeader(src_id=31, last=False)
        result = hdr.to_int()
        src_id_extracted = (result >> 11) & 0x1F
        assert src_id_extracted == 31

    def test_encode_last_bit(self):
        """last should set bit 16."""
        hdr = FlitHeader(last=True)
        assert (hdr.to_int() >> 16) & 0x1 == 1

        hdr_no_last = FlitHeader(last=False)
        assert (hdr_no_last.to_int() >> 16) & 0x1 == 0

    def test_encode_axi_ch_field(self):
        """axi_ch should occupy bits 19:17."""
        for ch in AxiChannel:
            hdr = FlitHeader(axi_ch=ch, last=False)
            result = hdr.to_int()
            axi_ch_extracted = (result >> 17) & 0x7
            assert axi_ch_extracted == int(ch)

    def test_encode_all_ones(self):
        """Header with maximum values."""
        hdr = FlitHeader(
            rob_req=True,
            rob_idx=31,
            dst_id=31,
            src_id=31,
            last=True,
            axi_ch=AxiChannel.R,  # value 4
        )
        result = hdr.to_int()
        # All bits should be set within their ranges
        assert result & 0x1 == 1  # rob_req
        assert (result >> 1) & 0x1F == 31  # rob_idx
        assert (result >> 6) & 0x1F == 31  # dst_id
        assert (result >> 11) & 0x1F == 31  # src_id
        assert (result >> 16) & 0x1 == 1  # last
        assert (result >> 17) & 0x7 == 4  # axi_ch


class TestFlitHeaderFromInt:
    """Tests for FlitHeader.from_int() method."""

    def test_decode_zero(self):
        """Decode zero value."""
        hdr = FlitHeader.from_int(0)
        assert hdr.rob_req is False
        assert hdr.rob_idx == 0
        assert hdr.dst_id == 0
        assert hdr.src_id == 0
        assert hdr.last is False
        assert hdr.axi_ch == AxiChannel.AW

    def test_decode_last_bit_only(self):
        """Decode with only last bit set."""
        hdr = FlitHeader.from_int(0x10000)
        assert hdr.last is True
        assert hdr.rob_req is False

    def test_decode_rob_req_bit(self):
        """Decode with rob_req bit set."""
        hdr = FlitHeader.from_int(0x1)
        assert hdr.rob_req is True

    def test_decode_axi_channels(self):
        """Decode all valid AXI channel values."""
        for ch in AxiChannel:
            value = int(ch) << 17
            hdr = FlitHeader.from_int(value)
            assert hdr.axi_ch == ch

    def test_decode_invalid_axi_ch_raises(self):
        """Decoding invalid axi_ch value should raise ValueError."""
        # Value 5, 6, 7 are invalid (only 0-4 are valid)
        for invalid_ch in [5, 6, 7]:
            value = invalid_ch << 17
            with pytest.raises(ValueError, match=f"Invalid axi_ch value: {invalid_ch}"):
                FlitHeader.from_int(value)


class TestFlitHeaderRoundtrip:
    """Tests for FlitHeader encoding/decoding roundtrip preservation."""

    @pytest.mark.parametrize("rob_req", [False, True])
    @pytest.mark.parametrize("last", [False, True])
    @pytest.mark.parametrize("axi_ch", list(AxiChannel))
    def test_roundtrip_flags(self, rob_req, last, axi_ch):
        """Roundtrip should preserve boolean flags and channel."""
        original = FlitHeader(rob_req=rob_req, last=last, axi_ch=axi_ch)
        encoded = original.to_int()
        decoded = FlitHeader.from_int(encoded)

        assert decoded.rob_req == rob_req
        assert decoded.last == last
        assert decoded.axi_ch == axi_ch

    @pytest.mark.parametrize("rob_idx", [0, 1, 15, 31])
    def test_roundtrip_rob_idx(self, rob_idx):
        """Roundtrip should preserve rob_idx."""
        original = FlitHeader(rob_idx=rob_idx)
        decoded = FlitHeader.from_int(original.to_int())
        assert decoded.rob_idx == rob_idx

    @pytest.mark.parametrize("node_id", [0, 1, 15, 31])
    def test_roundtrip_dst_id(self, node_id):
        """Roundtrip should preserve dst_id."""
        original = FlitHeader(dst_id=node_id)
        decoded = FlitHeader.from_int(original.to_int())
        assert decoded.dst_id == node_id

    @pytest.mark.parametrize("node_id", [0, 1, 15, 31])
    def test_roundtrip_src_id(self, node_id):
        """Roundtrip should preserve src_id."""
        original = FlitHeader(src_id=node_id)
        decoded = FlitHeader.from_int(original.to_int())
        assert decoded.src_id == node_id

    def test_roundtrip_comprehensive(self):
        """Full roundtrip with all fields set."""
        original = FlitHeader(
            rob_req=True,
            rob_idx=17,
            dst_id=23,
            src_id=5,
            last=True,
            axi_ch=AxiChannel.R,
        )
        decoded = FlitHeader.from_int(original.to_int())

        assert decoded.rob_req == original.rob_req
        assert decoded.rob_idx == original.rob_idx
        assert decoded.dst_id == original.dst_id
        assert decoded.src_id == original.src_id
        assert decoded.last == original.last
        assert decoded.axi_ch == original.axi_ch


class TestFlitHeaderFieldIsolation:
    """Tests verifying fields don't affect each other."""

    def test_rob_idx_isolated(self):
        """Changing rob_idx should only affect bits 5:1."""
        base = FlitHeader(dst_id=31, src_id=31, last=True, axi_ch=AxiChannel.R)
        base_val = base.to_int()

        modified = FlitHeader(rob_idx=31, dst_id=31, src_id=31, last=True, axi_ch=AxiChannel.R)
        mod_val = modified.to_int()

        # Only bits 5:1 should change
        mask = 0x1F << 1  # rob_idx mask
        assert (base_val & ~mask) == (mod_val & ~mask)
        assert (mod_val >> 1) & 0x1F == 31

    def test_dst_id_isolated(self):
        """Changing dst_id should only affect bits 10:6."""
        base = FlitHeader(rob_idx=31, src_id=31)
        modified = FlitHeader(rob_idx=31, dst_id=31, src_id=31)

        mask = 0x1F << 6
        assert (base.to_int() & ~mask) == (modified.to_int() & ~mask)

    def test_src_id_isolated(self):
        """Changing src_id should only affect bits 15:11."""
        base = FlitHeader(rob_idx=31, dst_id=31)
        modified = FlitHeader(rob_idx=31, dst_id=31, src_id=31)

        mask = 0x1F << 11
        assert (base.to_int() & ~mask) == (modified.to_int() & ~mask)


# =============================================================================
# Test: FlitHeader Properties
# =============================================================================

class TestFlitHeaderProperties:
    """Tests for FlitHeader computed properties."""

    def test_src_property(self):
        """src property should decode src_id to coordinate."""
        # src_id = 19 = (4 << 2) | 3 = (4, 3)
        hdr = FlitHeader(src_id=19)
        assert hdr.src == (4, 3)

    def test_dest_property(self):
        """dest property should decode dst_id to coordinate."""
        # dst_id = 7 = (1 << 2) | 3 = (1, 3)
        hdr = FlitHeader(dst_id=7)
        assert hdr.dest == (1, 3)

    def test_is_request_aw(self):
        """AW channel should be request."""
        hdr = FlitHeader(axi_ch=AxiChannel.AW)
        assert hdr.is_request() is True
        assert hdr.is_response() is False

    def test_is_request_w(self):
        """W channel should be request."""
        hdr = FlitHeader(axi_ch=AxiChannel.W)
        assert hdr.is_request() is True
        assert hdr.is_response() is False

    def test_is_request_ar(self):
        """AR channel should be request."""
        hdr = FlitHeader(axi_ch=AxiChannel.AR)
        assert hdr.is_request() is True
        assert hdr.is_response() is False

    def test_is_response_b(self):
        """B channel should be response."""
        hdr = FlitHeader(axi_ch=AxiChannel.B)
        assert hdr.is_request() is False
        assert hdr.is_response() is True

    def test_is_response_r(self):
        """R channel should be response."""
        hdr = FlitHeader(axi_ch=AxiChannel.R)
        assert hdr.is_request() is False
        assert hdr.is_response() is True


# =============================================================================
# Test: AxiChannel Enum
# =============================================================================

class TestAxiChannelEnum:
    """Tests for AxiChannel enum behavior."""

    def test_channel_values(self):
        """Channel enum values should be sequential 0-4."""
        assert int(AxiChannel.AW) == 0
        assert int(AxiChannel.W) == 1
        assert int(AxiChannel.AR) == 2
        assert int(AxiChannel.B) == 3
        assert int(AxiChannel.R) == 4

    def test_request_channels(self):
        """Request channels are AW, W, AR."""
        assert AxiChannel.AW.is_request() is True
        assert AxiChannel.W.is_request() is True
        assert AxiChannel.AR.is_request() is True

    def test_response_channels(self):
        """Response channels are B, R."""
        assert AxiChannel.B.is_response() is True
        assert AxiChannel.R.is_response() is True

    def test_mutual_exclusivity(self):
        """Each channel should be either request or response, not both."""
        for ch in AxiChannel:
            is_req = ch.is_request()
            is_resp = ch.is_response()
            assert is_req != is_resp, f"{ch} should be exactly one of request/response"
