---
name: Flit & Packet
description: FlooNoC 風格的 Flit 與 Packet 結構
globs:
  - src/core/flit.py
  - src/core/packet.py
  - src/core/ni.py
---

## FlooNoC Flit 格式

### Header (20 bits)
```
| Bit     | Field   | Width | Description              |
|---------|---------|-------|--------------------------|
| [0]     | rob_req | 1     | RoB request flag         |
| [5:1]   | rob_idx | 5     | RoB index (0-31)         |
| [10:6]  | dst_id  | 5     | Destination {x[2:0], y[1:0]} |
| [15:11] | src_id  | 5     | Source {x[2:0], y[1:0]}  |
| [16]    | last    | 1     | Last flit of packet      |
| [19:17] | axi_ch  | 3     | AXI channel type         |
```

### AXI Channel Types
```python
class AxiChannel(IntEnum):
    AW = 0  # Write Address (53 bits)
    W = 1   # Write Data (288 bits) - 含 strb
    AR = 2  # Read Address (53 bits)
    B = 3   # Write Response (10 bits)
    R = 4   # Read Response (266 bits)
```

### 重要：W Flit 的 strb 欄位

`AxiWPayload.strb` 是 32-bit write strobe，指示哪些 bytes 有效：

```python
# 例如：只有前 10 bytes 有效
strb = (1 << 10) - 1  # = 0x000003FF

# PacketDisassembler 使用 strb 去除 padding
valid_bytes = count_valid_bytes(strb)
payload += data[:valid_bytes]
```

## Packet 結構

### Write Request
```
[AW flit] → [W flit] → [W flit] → ... → [W flit (last=True)]
```

### Read Request
```
[AR flit (last=True)]
```

### Write Response
```
[B flit (last=True)]
```

### Read Response
```
[R flit] → [R flit] → ... → [R flit (last=True)]
```

## PacketAssembler / PacketDisassembler

- **PacketAssembler**: Packet → List[Flit]
  - 設定正確的 `strb` mask 標示有效 bytes

- **PacketDisassembler**: List[Flit] → Packet
  - 使用 `strb` 只取出有效 bytes，去除 padding
