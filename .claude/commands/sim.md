---
name: sim
description: 執行 NoC 模擬
---

請根據使用者需求執行模擬：

## 可用模式

1. **快速 Demo**
   ```bash
   make quick
   ```

2. **Host-to-NoC 模擬**
   ```bash
   make gen_payload PAYLOAD_SIZE=1024
   make gen_config NUM_TRANSFERS=10
   make sim
   ```

3. **NoC-to-NoC 模擬**
   ```bash
   make gen_noc_payload
   make sim_noc_neighbor
   ```

4. **效能批次測試**
   ```bash
   py -3 tools/run_batch_perf_test.py --mode both --count 100
   ```

請詢問使用者想要執行哪種模擬，並報告結果。
