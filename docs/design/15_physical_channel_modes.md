# Physical Channel Architecture Comparison

本文件比較兩種 NoC Physical Channel 架構：**General Mode（方案 A）** 和 **AXI Mode（方案 B）**，提供詳細的 bit-level 分析和 trade-off 指南。

## 目錄

1. [架構概述](#架構概述)
2. [Bit-Level 詳細比較](#bit-level-詳細比較)
3. [優缺點分析](#優缺點分析)
4. [效能影響分析](#效能影響分析)
5. [模擬驗證結果](#模擬驗證結果)
6. [Trade-off 決策指南](#trade-off-決策指南)
7. [建議與結論](#建議與結論)

---

## 架構概述

### General Mode（方案 A）- 2 條 Multiplexed Channel

目前的架構設計，將 5 個 AXI channel 合併到 2 條 physical channel：

```
                    ┌─────────────────────────────────────────┐
                    │            Combined Router              │
                    │                                         │
          N ────────┤  ┌─────────────────────────────────┐   │
          S ────────┤  │         ReqRouter (5×5)         │   │──── 處理 AW, W, AR
          E ────────┤  │    (Request Sub-Router)         │   │
          W ────────┤  └─────────────────────────────────┘   │
          Local ────┤                                         │
                    │  ┌─────────────────────────────────┐   │
                    │  │        RespRouter (5×5)         │   │──── 處理 B, R
                    │  │    (Response Sub-Router)        │   │
                    └──┴─────────────────────────────────┴───┘

Physical Wires per Direction: 4 條 (Req in/out + Resp in/out)
Sub-Router 數量: 2 個
```

**特點**：
- 使用 `axi_ch` 欄位（3 bits）區分 channel 類型
- Payload 對齊到最大 channel（Request: 288 bits, Response: 266 bits）
- 較小的線寬，但存在 Head-of-Line blocking

### AXI Mode（方案 B）- 5 條 Independent Channel

將每個 AXI channel 獨立成一條 physical channel：

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    Combined Router                       │
                    │                                                          │
          N ────────┤  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐     │
          S ────────┤  │  AW   │ │   W   │ │  AR   │ │   B   │ │   R   │     │
          E ────────┤  │Router │ │Router │ │Router │ │Router │ │Router │     │
          W ────────┤  │ (5×5) │ │ (5×5) │ │ (5×5) │ │ (5×5) │ │ (5×5) │     │
          Local ────┤  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘     │
                    └─────────────────────────────────────────────────────────┘

Physical Wires per Direction: 10 條 (5 channels × in/out)
Sub-Router 數量: 5 個
```

**特點**：
- 不需要 `axi_ch` 欄位，channel 本身代表類型
- 每個 channel payload 精確匹配，無浪費
- 無 Head-of-Line blocking，但線寬和複雜度增加

---

## Bit-Level 詳細比較

### Header 結構比較

| 欄位 | 方案 A (General) | 方案 B (AXI) | 說明 |
|------|-----------------|--------------|------|
| rob_req | 1 bit | 1 bit | RoB 請求標誌 |
| rob_idx | 5 bits | 5 bits | RoB 索引 (32 entries) |
| dst_id | 5 bits | 5 bits | 目標節點 {x[2:0], y[1:0]} |
| src_id | 5 bits | 5 bits | 來源節點 {x[2:0], y[1:0]} |
| last | 1 bit | 1 bit | Packet 結束標誌 |
| axi_ch | 3 bits | ~~不需要~~ | Channel 類型識別 |
| **Header 總計** | **20 bits** | **17 bits** | **-15%** |

### 方案 A：General Mode 詳細結構

#### Request Channel (AW/W/AR 共用)

```
┌─────────────────────────────────────────────────────────────┐
│                    Request Channel (310 bits)                │
├──────┬──────┬────────────────┬──────────────────────────────┤
│valid │ready │    header      │         payload              │
│ (1)  │ (1)  │    (20)        │         (288)                │
├──────┴──────┴────────────────┴──────────────────────────────┤
│                                                              │
│  Header (20 bits):                                           │
│  ┌───────┬───────┬───────┬───────┬──────┬────────┐          │
│  │rob_req│rob_idx│dst_id │src_id │ last │ axi_ch │          │
│  │  (1)  │  (5)  │  (5)  │  (5)  │  (1) │  (3)   │          │
│  └───────┴───────┴───────┴───────┴──────┴────────┘          │
│                                                              │
│  Payload (288 bits) - Union of:                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AW: addr(32) + id(8) + len(8) + size(3) + burst(2)  │    │
│  │     = 53 bits [padding: 235 bits]                   │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ W:  data(256) + strb(32) = 288 bits [padding: 0]    │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ AR: addr(32) + id(8) + len(8) + size(3) + burst(2)  │    │
│  │     = 53 bits [padding: 235 bits]                   │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

| 組成 | Bits | 說明 |
|------|------|------|
| valid | 1 | Flit 有效信號 |
| ready | 1 | Credit/Flow control |
| header | 20 | 含 axi_ch 區分類型 |
| payload | 288 | Max(AW=53, W=288, AR=53) |
| **總計** | **310** | |

#### Response Channel (B/R 共用)

```
┌─────────────────────────────────────────────────────────────┐
│                   Response Channel (288 bits)                │
├──────┬──────┬────────────────┬──────────────────────────────┤
│valid │ready │    header      │         payload              │
│ (1)  │ (1)  │    (20)        │         (266)                │
├──────┴──────┴────────────────┴──────────────────────────────┤
│                                                              │
│  Payload (266 bits) - Union of:                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ B: id(8) + resp(2) = 10 bits [padding: 256 bits]    │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ R: data(256) + id(8) + resp(2) = 266 bits           │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

| 組成 | Bits | 說明 |
|------|------|------|
| valid | 1 | Flit 有效信號 |
| ready | 1 | Credit/Flow control |
| header | 20 | 含 axi_ch 區分類型 |
| payload | 266 | Max(B=10, R=266) |
| **總計** | **288** | |

#### 方案 A 每方向總寬度

| Channel | In | Out | 小計 |
|---------|-----|-----|------|
| Request | 310 | 310 | 620 |
| Response | 288 | 288 | 576 |
| **總計** | **598** | **598** | **1,196 bits** |

---

### 方案 B：AXI Mode 詳細結構

#### AW Channel (Write Address)

```
┌────────────────────────────────────────────┐
│           AW Channel (72 bits)             │
├──────┬──────┬─────────┬───────────────────┤
│valid │ready │ header  │     payload       │
│ (1)  │ (1)  │  (17)   │      (53)         │
├──────┴──────┴─────────┴───────────────────┤
│  Header (17 bits): 無 axi_ch               │
│  Payload: addr(32)+id(8)+len(8)+           │
│           size(3)+burst(2) = 53 bits       │
└────────────────────────────────────────────┘
```

| 組成 | Bits |
|------|------|
| valid | 1 |
| ready | 1 |
| header | 17 |
| payload | 53 |
| **總計** | **72** |

#### W Channel (Write Data)

```
┌────────────────────────────────────────────┐
│           W Channel (307 bits)             │
├──────┬──────┬─────────┬───────────────────┤
│valid │ready │ header  │     payload       │
│ (1)  │ (1)  │  (17)   │      (288)        │
├──────┴──────┴─────────┴───────────────────┤
│  Payload: data(256) + strb(32) = 288 bits  │
└────────────────────────────────────────────┘
```

| 組成 | Bits |
|------|------|
| valid | 1 |
| ready | 1 |
| header | 17 |
| payload | 288 |
| **總計** | **307** |

#### AR Channel (Read Address)

```
┌────────────────────────────────────────────┐
│           AR Channel (72 bits)             │
├──────┬──────┬─────────┬───────────────────┤
│valid │ready │ header  │     payload       │
│ (1)  │ (1)  │  (17)   │      (53)         │
└────────────────────────────────────────────┘
```

| 組成 | Bits |
|------|------|
| valid | 1 |
| ready | 1 |
| header | 17 |
| payload | 53 |
| **總計** | **72** |

#### B Channel (Write Response)

```
┌────────────────────────────────────────────┐
│           B Channel (29 bits)              │
├──────┬──────┬─────────┬───────────────────┤
│valid │ready │ header  │     payload       │
│ (1)  │ (1)  │  (17)   │      (10)         │
├──────┴──────┴─────────┴───────────────────┤
│  Payload: id(8) + resp(2) = 10 bits        │
└────────────────────────────────────────────┘
```

| 組成 | Bits |
|------|------|
| valid | 1 |
| ready | 1 |
| header | 17 |
| payload | 10 |
| **總計** | **29** |

#### R Channel (Read Data)

```
┌────────────────────────────────────────────┐
│           R Channel (285 bits)             │
├──────┬──────┬─────────┬───────────────────┤
│valid │ready │ header  │     payload       │
│ (1)  │ (1)  │  (17)   │      (266)        │
├──────┴──────┴─────────┴───────────────────┤
│  Payload: data(256)+id(8)+resp(2)=266 bits │
└────────────────────────────────────────────┘
```

| 組成 | Bits |
|------|------|
| valid | 1 |
| ready | 1 |
| header | 17 |
| payload | 266 |
| **總計** | **285** |

#### 方案 B 每方向總寬度

| Channel | In | Out | 小計 |
|---------|-----|-----|------|
| AW | 72 | 72 | 144 |
| W | 307 | 307 | 614 |
| AR | 72 | 72 | 144 |
| B | 29 | 29 | 58 |
| R | 285 | 285 | 570 |
| **總計** | **765** | **765** | **1,530 bits** |

---

### 總體比較摘要

| 指標 | 方案 A (General) | 方案 B (AXI) | 差異 |
|------|-----------------|--------------|------|
| **Header 大小** | 20 bits | 17 bits | -15% |
| **每方向線寬** | 1,196 bits | 1,530 bits | +28% |
| **5-port Router 總線寬** | 5,980 bits | 7,650 bits | +28% |
| **Wire 數量/方向** | 4 條 | 10 條 | +150% |
| **Sub-Router 數量** | 2 個 (Req+Resp) | 5 個 (AW+W+AR+B+R) | +150% |
| **Arbiter 數量** | 10 個 | 25 個 | +150% |
| **NI→Router Channel** | 2 條 (Req+Resp) | 5 條 (AW+W+AR+B+R) | +150% |
| **NI Output Buffer** | 2 個 (共用) | 5 個 (獨立) | +150% |

### Payload 利用率分析

#### 方案 A Payload 浪費

| Flit Type | 實際 Payload | 分配空間 | 浪費 | 浪費率 |
|-----------|-------------|----------|------|--------|
| AW on Request | 53 bits | 288 bits | 235 bits | 82% |
| W on Request | 288 bits | 288 bits | 0 bits | 0% |
| AR on Request | 53 bits | 288 bits | 235 bits | 82% |
| B on Response | 10 bits | 266 bits | 256 bits | 96% |
| R on Response | 266 bits | 266 bits | 0 bits | 0% |

**平均浪費率**: ~52%

#### 方案 B Payload 浪費

| Channel | 實際 Payload | 分配空間 | 浪費 | 浪費率 |
|---------|-------------|----------|------|--------|
| AW | 53 bits | 53 bits | 0 bits | 0% |
| W | 288 bits | 288 bits | 0 bits | 0% |
| AR | 53 bits | 53 bits | 0 bits | 0% |
| B | 10 bits | 10 bits | 0 bits | 0% |
| R | 266 bits | 266 bits | 0 bits | 0% |

**平均浪費率**: 0%

### Channel 利用率分析

除了 Payload 利用率外，不同工作負載下的 Channel 利用率也是重要的考量因素。

#### Write-Heavy Workload（例如：DMA 寫入）

| Channel | 方案 A | 方案 B | 說明 |
|---------|--------|--------|------|
| AW | Request Channel 共用 | 高利用率 | 每個 write transaction 需要一個 AW flit |
| W | Request Channel 共用 | 極高利用率 | burst write 產生大量 W flits |
| AR | Request Channel 共用 | **閒置** | 無 read 操作 |
| B | Response Channel 共用 | 高利用率 | 每個 write 需要 response |
| R | Response Channel 共用 | **閒置** | 無 read 操作 |

```
方案 A (General Mode) - Write-Heavy:
  Request Channel:  AW → W → W → W → W → AW → W → W → ...  (高利用率)
  Response Channel: B → B → B → ...                         (中利用率)

方案 B (AXI Mode) - Write-Heavy:
  AW: ████████████░░░░  (高利用率)
  W:  ████████████████  (極高利用率)
  AR: ░░░░░░░░░░░░░░░░  (閒置)
  B:  ████████████░░░░  (高利用率)
  R:  ░░░░░░░░░░░░░░░░  (閒置)
```

#### Read-Heavy Workload（例如：DMA 讀取）

| Channel | 方案 A | 方案 B | 說明 |
|---------|--------|--------|------|
| AW | Request Channel 共用 | **閒置** | 無 write 操作 |
| W | Request Channel 共用 | **閒置** | 無 write 操作 |
| AR | Request Channel 共用 | 高利用率 | 每個 read transaction 需要一個 AR flit |
| B | Response Channel 共用 | **閒置** | 無 write 操作 |
| R | Response Channel 共用 | 極高利用率 | burst read 產生大量 R flits |

```
方案 A (General Mode) - Read-Heavy:
  Request Channel:  AR → AR → AR → ...                      (低利用率)
  Response Channel: R → R → R → R → R → R → R → ...         (高利用率)

方案 B (AXI Mode) - Read-Heavy:
  AW: ░░░░░░░░░░░░░░░░  (閒置)
  W:  ░░░░░░░░░░░░░░░░  (閒置)
  AR: ████████████░░░░  (高利用率)
  B:  ░░░░░░░░░░░░░░░░  (閒置)
  R:  ████████████████  (極高利用率)
```

#### Mixed Workload（50/50 讀寫混合）

| Channel | 方案 A | 方案 B |
|---------|--------|--------|
| AW | Request Channel 共用 | 中利用率 |
| W | Request Channel 共用 | 高利用率 |
| AR | Request Channel 共用 | 中利用率 |
| B | Response Channel 共用 | 中利用率 |
| R | Response Channel 共用 | 高利用率 |

#### Channel 利用率 Trade-off 分析

| 工作負載 | 方案 A 特性 | 方案 B 特性 |
|----------|-------------|-------------|
| **Write-Heavy** | Request Channel 高利用率，Response 低利用率 | AR/R Channel 完全閒置（50% 資源浪費） |
| **Read-Heavy** | Request Channel 低利用率，Response 高利用率 | AW/W/B Channel 完全閒置（60% 資源浪費） |
| **Mixed** | 兩個 Channel 都有中高利用率 | 所有 Channel 都有適度利用率 |

**結論**：
- **方案 A** 的共用 Channel 設計在單向工作負載下可更有效利用線寬資源
- **方案 B** 在單向工作負載下會有部分 Channel 閒置，但在 Mixed 工作負載下不會有 HoL blocking
- 選擇時需考慮實際應用的流量模式特性

---

## 優缺點分析

### 方案 A：General Mode

#### 優點

| 優點 | 說明 |
|------|------|
| ✅ **線寬較小** | 每方向 1,196 bits vs 1,530 bits (-28%) |
| ✅ **Router 結構簡單** | 只需 2 個 Sub-Router (ReqRouter + RespRouter) |
| ✅ **Arbiter 較少** | 10 個 vs 25 個 (-60%) |
| ✅ **設計驗證簡單** | 較少的 channel 意味著較少的邊界情況 |
| ✅ **功耗較低** | 較少的 wire 和控制邏輯 |
| ✅ **面積較小** | Router 面積約小 30-40% |

#### 缺點

| 缺點 | 說明 |
|------|------|
| ❌ **Head-of-Line Blocking** | W burst 會阻擋 AW/AR |
| ❌ **Payload 浪費** | AW/AR/B 有大量 padding (52% 平均浪費) |
| ❌ **需要 Mux/Demux** | 需要 channel multiplexing 邏輯 |
| ❌ **延遲變異大** | 因 blocking 導致延遲不可預測 |
| ❌ **頻寬利用率低** | 小 payload 浪費頻寬 |

### 方案 B：AXI Mode

#### 優點

| 優點 | 說明 |
|------|------|
| ✅ **無 HoL Blocking** | 每個 channel 獨立，不互相阻擋 |
| ✅ **零 Payload 浪費** | 每個 channel 精確匹配 payload 大小 |
| ✅ **低延遲** | Address channel (AW/AR) 獨立，快速傳輸 |
| ✅ **延遲可預測** | 無 blocking，延遲變異小 |
| ✅ **符合 AXI 語義** | 天然支援 AXI 獨立 channel 特性 |
| ✅ **設計直觀** | 每個 channel 邏輯獨立，易於理解 |

#### 缺點

| 缺點 | 說明 |
|------|------|
| ❌ **線寬增加** | 每方向 +334 bits (+28%) |
| ❌ **Router 結構複雜** | 需要 5 個 Sub-Router (AW/W/AR/B/R) (+150%) |
| ❌ **Arbiter 增加** | 25 個 vs 10 個 (+150%) |
| ❌ **面積增加** | Router 面積約增加 40-60% |
| ❌ **功耗增加** | 更多 wire 和控制邏輯 |
| ❌ **驗證複雜度** | 5 個獨立 channel 需要更多測試 |

---

## 效能影響分析

### Head-of-Line Blocking 問題

**方案 A 的 HoL Blocking 場景**：

```
時序圖 - 方案 A (General Mode):

Request Channel:
    ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │ AW │ W0 │ W1 │ W2 │ W3 │ AW │ AR │ AR │
    └────┴────┴────┴────┴────┴────┴────┴────┘
         ↑                   ↑
         │                   │
    W burst 開始        新 AW 等待 4 cycles
                        AR 等待 5 cycles

問題: AW/AR 必須等待 W burst 完成
```

**方案 B 無 HoL Blocking**：

```
時序圖 - 方案 B (AXI Mode):

AW Channel: ┌────┬────┬────┐
            │ AW │ AW │ AW │
            └────┴────┴────┘

W Channel:  ┌────┬────┬────┬────┬────┬────┐
            │ W0 │ W1 │ W2 │ W3 │ W0 │ W1 │
            └────┴────┴────┴────┴────┴────┘

AR Channel: ┌────┬────┬────┐
            │ AR │ AR │ AR │
            └────┴────┴────┘

優點: 所有 channel 同時進行，無阻擋
```

### 延遲比較

| 場景 | 方案 A | 方案 B | 差異 |
|------|--------|--------|------|
| **單一 AW flit** | 1 cycle | 1 cycle | 相同 |
| **AW 在 W burst 後** | 1 + burst_len cycles | 1 cycle | 方案 B 大幅優於 A |
| **AR 在 W burst 中** | 等待 burst 完成 | 1 cycle | 方案 B 大幅優於 A |
| **B response** | 可能被 R 阻擋 | 即時 | 方案 B 優於 A |

### 吞吐量比較

| 流量類型 | 方案 A | 方案 B | 說明 |
|----------|--------|--------|------|
| **純 Write** | 100% | 100% | 相同 |
| **純 Read** | 100% | 100% | 相同 |
| **混合 R/W** | ~70-85% | ~95-100% | 方案 B 明顯優於 A |
| **高 burst** | ~60-75% | ~90-100% | 方案 B 大幅優於 A |

---

## 模擬驗證結果

本節展示使用 NoC Behavior Model 進行的實際模擬比較結果。

### 測試配置

| 參數 | 值 | 說明 |
|------|-----|------|
| Mesh Topology | 5×4 (20 routers) | 4 Edge Routers + 16 Compute Nodes |
| Transfer Size | 4 KB | 每個目標節點 4KB |
| Target Nodes | 16 (all) | Broadcast 到所有計算節點 |
| Total Data | 64 KB | 4KB × 16 nodes |
| Beat Size | 8 bytes | AXI data width = 64 bits |
| Max Burst Length | 16 | 每個 AXI transaction |
| Max Outstanding | 8 | 並行 transaction 數量 |
| Buffer Depth | 16 | Router buffer 深度 |

**Flit 數量計算**：
- W flits = 64KB ÷ 8 bytes = **8,192 flits**
- AW flits = 4KB ÷ (16 × 8 bytes) × 16 nodes = **512 flits**
- Total request flits = **8,704 flits**

### 效能比較結果

| 指標 | General Mode | AXI Mode | 差異 |
|------|-------------|----------|------|
| **Total Cycles** | 8,267 | 8,203 | -0.8% |
| **Throughput** | 1.05 flits/cycle | 1.06 flits/cycle | +1.0% |
| **Average Latency** | 36.2 cycles | 4.5 cycles | -87.6% |
| **Min Latency** | 3 cycles | 3 cycles | 0% |
| **Max Latency** | 114 cycles | 6 cycles | -94.7% |
| **Latency Variance** | 111 cycles | 3 cycles | **-97.3%** |
| **L/L0 (Normalized)** | 8.04 | 1.00 | -87.6% |
| **Saturation Status** | SATURATED | NORMAL | - |
| **Verification** | 16 PASS | 16 PASS | - |

### Latency 指標說明

#### 什麼是 Per-Flit Latency？

```
Per-Flit Latency = MasterNI 收到時間 - SlaveNI 發出時間
                 = arrival_cycle - injection_cycle
```

這是每個 flit 從進入網路到抵達目的地的時間。

#### 什麼是 Latency Variance？

```
Latency Variance = Max Latency - Min Latency
                 = 最慢 flit 的延遲 - 最快 flit 的延遲
```

**Latency Variance 代表網路延遲的可預測性**：
- 低 variance = 延遲穩定、可預測（對 QoS 重要）
- 高 variance = 延遲不穩定、難以預測

### Dataflow 比較圖

#### General Mode - HoL Blocking 發生點

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GENERAL MODE DATAFLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Host Memory (64KB)                                                        │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────┐                                                       │
│   │   HostAXIMaster │  產生 512 AW + 8192 W flits                           │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  SlaveNI Output Buffer (共用)                                        │  │
│   │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐         │  │
│   │  │ W │ W │ W │ W │ W │ W │ W │AW │ W │ W │ W │ W │AW │...│         │  │
│   │  └───┴───┴───┴───┴───┴───┴───┴─▲─┴───┴───┴───┴───┴───┴───┘         │  │
│   │                                │                                    │  │
│   │                         HoL Blocking Point #1                       │  │
│   │                         AW 被前面的 W 阻擋                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │ RoutingSelector │  選擇 Edge Router (based on hop count + credits)     │
│   └────────┬────────┘                                                       │
│            │                                                                │
│     ┌──────┴──────┬───────────┬───────────┐                                │
│     ▼             ▼           ▼           ▼                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Edge Router Request Buffer (共用)                                    │  │
│  │  ┌───┬───┬───┬───┬───┬───┬───┐                                       │  │
│  │  │ W │ W │ W │AW │ W │ W │...│   ← HoL Blocking Point #2             │  │
│  │  └───┴───┴───┴─▲─┴───┴───┴───┘                                       │  │
│  │                │                                                      │  │
│  │         AW 再次被 W 阻擋                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│            │                                                                │
│            ▼                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  5×4 Mesh (Compute Router Request Buffers - 共用)                     │  │
│  │                                                                       │  │
│  │  每一跳的 Request Buffer 都可能發生 HoL Blocking                       │  │
│  │  ┌───┬───┬───┬───┬───┐                                               │  │
│  │  │ W │ W │AW │ W │...│   ← HoL Blocking Point #3, #4, ...            │  │
│  │  └───┴───┴─▲─┴───┴───┘                                               │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │    MasterNI     │   ← Latency 測量終點                                  │
│   │  (16 nodes)     │     latency = current_time - injection_cycle         │
│   └─────────────────┘                                                       │
│                                                                             │
│  結果：                                                                      │
│    - 某些 AW flit 幸運地前面沒有 W → Min Latency = 3 cycles                 │
│    - 某些 AW flit 每一跳都被 W 擋住 → Max Latency = 114 cycles              │
│    - Latency Variance = 114 - 3 = 111 cycles                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### AXI Mode - 無 HoL Blocking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AXI MODE DATAFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Host Memory (64KB)                                                        │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────┐                                                       │
│   │   HostAXIMaster │  產生 512 AW + 8192 W flits                           │
│   └────────┬────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  SlaveNI Output Buffers (獨立)                                       │  │
│   │                                                                      │  │
│   │  AW Buffer: ┌───┬───┬───┬───┬───┐   ← AW 獨立排隊，不受 W 影響       │  │
│   │             │AW │AW │AW │AW │...│                                   │  │
│   │             └───┴───┴───┴───┴───┘                                   │  │
│   │                                                                      │  │
│   │  W Buffer:  ┌───┬───┬───┬───┬───┬───┬───┬───┐   ← W 獨立排隊        │  │
│   │             │ W │ W │ W │ W │ W │ W │ W │...│                       │  │
│   │             └───┴───┴───┴───┴───┴───┴───┴───┘                       │  │
│   │                                                                      │  │
│   │  AR Buffer: ┌───┬───┬───┐   ← AR 獨立排隊（本測試未使用）             │  │
│   │             │   │   │   │                                           │  │
│   │             └───┴───┴───┘                                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│            │           │                                                    │
│            ▼           ▼                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  RoutingSelector (AXI Mode)                                          │  │
│   │  AXIModeEdgeRouterPort × 4 rows                                      │  │
│   │  ┌──────────┬──────────┬──────────┐                                 │  │
│   │  │ AW Port  │  W Port  │ AR Port  │  ← 獨立 port，獨立 wire          │  │
│   │  └────┬─────┴────┬─────┴────┬─────┘                                 │  │
│   └───────┼──────────┼──────────┼───────────────────────────────────────┘  │
│           │          │          │                                          │
│           ▼          ▼          ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  AXIModeEdgeRouter (5 Sub-Routers)                                   │  │
│   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │  │
│   │  │   AW   │  │   W    │  │   AR   │  │   B    │  │   R    │        │  │
│   │  │ Router │  │ Router │  │ Router │  │ Router │  │ Router │        │  │
│   │  │┌─┬─┬─┐│  │┌─┬─┬─┐ │  │┌─┬─┬─┐ │  │        │  │        │        │  │
│   │  ││A│A│A││  ││W│W│W│ │  ││ │ │ │ │  │        │  │        │        │  │
│   │  │└─┴─┴─┘│  │└─┴─┴─┘ │  │└─┴─┴─┘ │  │        │  │        │        │  │
│   │  └───┬───┘  └───┬────┘  └───┬────┘  └────────┘  └────────┘        │  │
│   │      │          │           │                                      │  │
│   │      │    獨立路徑，互不干擾  │                                      │  │
│   └──────┼──────────┼───────────┼──────────────────────────────────────┘  │
│          ▼          ▼           ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  5×4 Mesh (AXIModeRouter - 5 Sub-Routers each)                       │  │
│   │                                                                      │  │
│   │  AW_Router: ┌─┬─┬─┐     W_Router: ┌─┬─┬─┬─┬─┐                       │  │
│   │             │A│A│A│               │W│W│W│W│W│                       │  │
│   │             └─┴─┴─┘               └─┴─┴─┴─┴─┘                       │  │
│   │                │                       │                            │  │
│   │                │     獨立 Sub-Router    │                            │  │
│   │                │     無 Cross-Channel   │                            │  │
│   │                │     Blocking          │                            │  │
│   │                ▼                       ▼                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                    │                       │                               │
│                    ▼                       ▼                               │
│   ┌─────────────────┐                                                       │
│   │    MasterNI     │   ← Latency 測量終點                                  │
│   │  (16 nodes)     │     latency = current_time - injection_cycle         │
│   └─────────────────┘                                                       │
│                                                                             │
│  結果：                                                                      │
│    - 所有 AW flit 經歷相似的排隊時間                                         │
│    - 所有 W flit 經歷相似的排隊時間                                          │
│    - Min Latency = 3 cycles, Max Latency = 6 cycles                        │
│    - Latency Variance = 6 - 3 = 3 cycles                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 為何 Total Cycles 相近但 Latency Variance 差異巨大？

這是兩個不同維度的指標：

| 指標 | 測量對象 | 代表意義 |
|------|---------|---------|
| **Total Cycles** | 整個傳輸完成時間 | Network Throughput (總吞吐量) |
| **Latency Variance** | 個別 flit 延遲分布 | Network Fairness / QoS (公平性/可預測性) |

#### 類比說明

想像高速公路的車流：

| 情境 | General Mode | AXI Mode |
|------|-------------|----------|
| **車道配置** | 大卡車(W)和轎車(AW)共用車道 | 大卡車專用道 + 轎車專用道 |
| **總通過時間** | ~8,267 秒（總流量一樣） | ~8,203 秒 |
| **個別車輛時間差異** | 很大（轎車可能被卡車擋很久） | 很小（各走各的道路） |

#### 詳細分析

1. **Total Cycles 相近的原因**：
   - 網路總頻寬（throughput）相同
   - 瓶頸是 flit 總數量（8,704 flits），不是 HoL blocking
   - 兩種 mode 使用相同的 XY routing 和 wormhole switching

2. **Latency Variance 差異巨大的原因**：
   - **General Mode**：AW/W 在每一跳共用 buffer，W burst 阻擋 AW
     - 幸運的 AW：前面沒有 W → 快速通過（3 cycles）
     - 不幸的 AW：每一跳都被 W 擋 → 延遲累積（114 cycles）
   - **AXI Mode**：AW/W 有獨立 Sub-Router，互不干擾
     - 所有 flit 經歷相似的排隊延遲（3-6 cycles）

### 結論

| 結論 | 說明 |
|------|------|
| ✅ **AXI Mode 消除 HoL Blocking** | Latency Variance 降低 97.3% |
| ✅ **Throughput 相近** | 網路頻寬利用率相當 |
| ✅ **AXI Mode 適合 QoS 需求** | 延遲可預測，適合 real-time 應用 |
| ⚠️ **General Mode 延遲不可預測** | 高 variance 對 QoS 敏感應用不利 |

---

## Trade-off 決策指南

### 決策矩陣

| 考量因素 | 權重 | 方案 A 分數 | 方案 B 分數 |
|----------|------|------------|------------|
| 面積成本 | 高 | 9 | 5 |
| 功耗效率 | 高 | 8 | 5 |
| 延遲效能 | 中 | 5 | 9 |
| 吞吐量 | 中 | 6 | 9 |
| 設計複雜度 | 中 | 8 | 5 |
| 驗證難度 | 低 | 8 | 5 |
| AXI 相容性 | 低 | 6 | 10 |

### 選擇指南

#### 選擇方案 A（General Mode）當：

1. **面積是首要考量**
   - 嵌入式系統、IoT 設備
   - 成本敏感的應用

2. **流量模式簡單**
   - 主要是順序存取
   - 低 burst 流量
   - 讀寫比例固定

3. **功耗預算有限**
   - 電池供電設備
   - 熱設計受限

4. **開發資源有限**
   - 需要快速完成設計
   - 驗證團隊規模小

#### 選擇方案 B（AXI Mode）當：

1. **效能是首要考量**
   - 高效能計算
   - 低延遲需求
   - 即時系統

2. **流量模式複雜**
   - 混合讀寫流量
   - 高 burst 傳輸
   - 多個 master 競爭

3. **AXI 相容性重要**
   - 需要完整 AXI 特性
   - 與標準 AXI IP 整合

4. **可預測延遲重要**
   - 即時系統
   - QoS 需求

### 折衷方案：3-Channel Mode（方案 C）

如果兩種方案都無法完全滿足需求，可考慮折衷的 3-Channel 架構：

```
3-Channel Mode:
├── Address Channel (AW + AR) ←→ 合併 address
├── Write Data Channel (W)    ←→ 獨立 write data
└── Response Channel (B + R)  ←→ 合併 response
```

| 指標 | 方案 A | 方案 C | 方案 B |
|------|--------|--------|--------|
| Wire 數量 | 4 | 6 | 10 |
| 線寬增加 | 基準 | +15% | +28% |
| HoL Blocking | 嚴重 | 部分解決 | 無 |
| 複雜度 | 低 | 中 | 高 |

---

## 建議與結論

### 效能導向專案

**推薦：方案 B（AXI Mode）**

- 消除 HoL blocking，最大化吞吐量
- 延遲可預測，適合即時應用
- 面積增加 28-40% 是可接受的代價

### 成本導向專案

**推薦：方案 A（General Mode）**

- 面積和功耗最小化
- 適合簡單流量模式
- 設計和驗證成本較低

### 平衡型專案

**推薦：方案 C（3-Channel Mode）或依流量特性選擇**

- 分析實際流量模式後決定
- 可先用行為模型模擬比較
- 根據模擬結果選擇最佳方案

### 模擬驗證結論

本專案已完成兩種架構的行為模型並進行實際測試，詳見[模擬驗證結果](#模擬驗證結果)章節。

**關鍵發現**：
- **Throughput 差異小** (-0.8%)：瓶頸在網路頻寬，非 HoL blocking
- **Latency Variance 差異大** (-97.3%)：AXI Mode 顯著改善延遲可預測性
- **QoS 需求**：若應用對延遲穩定性敏感，強烈建議 AXI Mode

---

## 參考資料

- [FlooNoC Physical Channel Design](https://github.com/pulp-platform/FlooNoC)
- [AXI4 Protocol Specification](https://developer.arm.com/documentation/ihi0022/latest)
- [NoC Behavior Model - Flit Format](./05_flit.md)
- [NoC Behavior Model - Router Architecture](./02_router.md)

---

*文件版本: 1.3*
*最後更新: 2026-01-15*
*更新內容: 新增模擬驗證結果章節（4KB Broadcast Write 效能比較）*
