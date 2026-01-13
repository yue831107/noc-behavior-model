---
name: NoC Routing
description: NoC 路由與封包傳輸知識
globs:
  - src/core/router.py
  - src/core/routing_selector.py
  - src/core/mesh.py
---

## XY Routing 演算法

### 規則
1. **X-first**: 先沿 X 軸移動到目標 column
2. **Y-second**: 再沿 Y 軸移動到目標 row
3. **Y→X Turn Prevention**: 禁止從 Y 方向轉向 X 方向（避免 deadlock）

### 方向定義
```python
class Direction(Enum):
    NORTH = 0  # Y+
    EAST = 1   # X+
    SOUTH = 2  # Y-
    WEST = 3   # X-
    LOCAL = 4  # 本地 NI
```

### 路由決策
```python
def compute_output_port(current: tuple, dest: tuple) -> Direction:
    cx, cy = current
    dx, dy = dest

    if dx > cx:
        return Direction.EAST
    elif dx < cx:
        return Direction.WEST
    elif dy > cy:
        return Direction.NORTH
    elif dy < cy:
        return Direction.SOUTH
    else:
        return Direction.LOCAL
```

## Wormhole Switching

- **Packet-level locking**: HEAD flit 鎖定路徑直到 TAIL flit 通過
- **WormholeArbiter**: 管理每個 output port 的 packet 鎖定狀態
- **Credit-based flow control**: 使用 credit 防止 buffer overflow

## Mesh 拓撲

```
(0,3) - (1,3) - (2,3) - (3,3) - (4,3)
  |       |       |       |       |
(0,2) - (1,2) - (2,2) - (3,2) - (4,2)
  |       |       |       |       |
(0,1) - (1,1) - (2,1) - (3,1) - (4,1)
  |       |       |       |       |
(0,0) - (1,0) - (2,0) - (3,0) - (4,0)

Column 0: Edge Routers (連接 Routing Selector)
Columns 1-4: Compute Nodes (Router + NI + Memory)
```
