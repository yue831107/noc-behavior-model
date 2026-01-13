---
name: perf
description: 執行效能驗證測試
---

執行效能驗證測試並分析結果：

1. 執行理論值驗證：
   ```bash
   py -3 -m pytest tests/performance/test_theory_validation.py -v
   ```

2. 分析結果：
   - Throughput 是否符合理論值
   - Latency 是否合理
   - Buffer utilization 狀態

3. 如有失敗，深入分析原因並提供改進建議
