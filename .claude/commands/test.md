---
name: test
description: 執行測試套件
---

請執行以下步驟：

1. 執行完整測試：
   ```bash
   py -3 -m pytest tests/ -v --tb=short
   ```

2. 分析測試結果：
   - 列出所有失敗的測試
   - 說明失敗原因
   - 提供修復建議

3. 如果全部通過，報告測試覆蓋率摘要
