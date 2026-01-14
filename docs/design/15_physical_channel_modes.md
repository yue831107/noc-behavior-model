# Physical Channel Architecture Comparison

本文件比較兩種 NoC Physical Channel 架構：**General Mode（方案 A）** 和 **AXI Mode（方案 B）**，提供詳細的 bit-level 分析和 trade-off 指南。

## 目錄

1. [架構概述](#架構概述)
2. [Bit-Level 詳細比較](#bit-level-詳細比較)
3. [優缺點分析](#優缺點分析)
4. [效能影響分析](#效能影響分析)
5. [Trade-off 決策指南](#trade-off-決策指南)
6. [建議與結論](#建議與結論)

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

### 模擬驗證建議

在做最終決策前，建議：

1. **建立兩種架構的行為模型**
2. **使用實際流量 pattern 測試**
3. **比較關鍵指標**：
   - 平均延遲
   - 延遲變異 (jitter)
   - 吞吐量
   - 資源利用率
4. **評估面積/效能比**

---

## 參考資料

- [FlooNoC Physical Channel Design](https://github.com/pulp-platform/FlooNoC)
- [AXI4 Protocol Specification](https://developer.arm.com/documentation/ihi0022/latest)
- [NoC Behavior Model - Flit Format](./05_flit.md)
- [NoC Behavior Model - Router Architecture](./02_router.md)

---

*文件版本: 1.0*
*最後更新: 2025-01-14*
