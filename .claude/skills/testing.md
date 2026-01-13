---
name: Testing Patterns
description: 測試模式與驗證策略
globs:
  - tests/**/*.py
  - tests/conftest.py
---

## 測試架構

### 目錄結構
```
tests/
├── unit/           # 單元測試 - 單一元件
├── integration/    # 整合測試 - 多元件互動
├── performance/    # 效能測試 - 理論值驗證
└── conftest.py     # 共用 fixtures
```

### Cycle-accurate 模擬步驟

```python
def run_multi_router_cycle(routers, cycle):
    # 1. Sample inputs
    for r in routers:
        r.sample_inputs()

    # 2. Clear input signals
    for r in routers:
        r.clear_input_signals()

    # 3. Update ready signals
    for r in routers:
        r.update_ready_signals()

    # 4. Route and forward
    for r in routers:
        r.route_flits(cycle)

    # 5. Propagate wire signals
    propagate_all_wires(routers)

    # 6. Clear accepted outputs
    for r in routers:
        r.clear_accepted_outputs()

    # 7. Credit release
    for r in routers:
        r.propagate_credit_release()
```

### 常用 Fixtures

```python
@pytest.fixture
def router_config():
    return RouterConfig(buffer_depth=4, ...)

@pytest.fixture
def two_routers_horizontal(router_config):
    # 建立水平相連的兩個 router
    ...

@pytest.fixture
def router_chain_horizontal(router_config):
    # 建立 router chain
    ...
```

## 驗證策略

### Golden Verification
- `GoldenManager` 記錄預期資料
- 模擬結束後比對 Memory 內容
- Scatter 模式：每個節點分配不同資料

### Theory Validation
- Throughput ≈ 理論最大值
- Latency ≈ hop_count × cycles_per_hop
- Little's Law: L = λW
