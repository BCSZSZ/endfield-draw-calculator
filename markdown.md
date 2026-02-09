```mermaid
flowchart TD
    A["开始: dp(pity=0)(copies=0)=1"]

    subgraph "每一抽的常规转移"
        B["当前状态: dp(p)(c)"]
        B -->|抽到UP| C1["finish_probs[cost]+=p_up * dp(p)(c) (若c+1=目标)"]
        B -->|抽到UP| C2["new_dp[0][c+1]+=p_up * dp(p)(c) (若c+1<目标)"]
        B -->|抽到歪| D["new_dp[0][c]+=p_other * dp(p)(c)"]
        B -->|未出金| E["new_dp[p+1][c]+=p_fail * dp(p)(c)"]
    end

    subgraph "规则3: cost=30 赠送10连"
        F["new_dp拆分 -> temp_dp"]
        F --> G["对每个k=0..10应用二项分布概率"]
        G --> H["若c+k>=目标: finish_probs[cost]+=prob"]
        G --> I["否则: temp_dp[p][c+k]+=prob"]
    end

    subgraph "规则4: cost=120 首发硬保底"
        J["new_dp[p][0]整体转移"]
        J --> K{目标=1?}
        K -->|是| L["finish_probs[cost]+=sum"]
        K -->|否| M["new_dp[p][1]+=shift_mass"]
    end

    subgraph "规则5: cost为240倍数 里程碑赠送"
        N["所有未毕业状态 copies+1"]
        N --> O["若c+1>=目标: finish_probs[cost]+=sum"]
        N --> P["否则: temp_dp[:,c+1]+=new_dp[:,c]"]
    end

    A --> B
    C2 --> Q["更新dp=new_dp"]
    D --> Q
    E --> Q
    H --> Q
    I --> Q
    L --> Q
    M --> Q
    O --> Q
    P --> Q
    Q --> R{累计概率极小?}
    R -->|否| B
    R -->|是| S["结束: 统计期望/方差/分位点等"]
```

| pity范围 | base(出金率) | p_up | p_other | p_fail |
| --- | --- | --- | --- | --- |
| 0-65 | 0.008 | 0.004 | 0.004 | 0.992 |
| 66-79 | 0.008 + (pity - 65) * 0.05 | base * 0.5 | base * 0.5 | 1 - base |
| 80 | 1.0 | 0.5 | 0.5 | 0.0 |

注：这里的 base 是“出金率”，p_up/p_other/p_fail 分别是抽到UP、抽到歪、没出金的概率。

| 规则 | 触发条件 | 状态变化 | 是否一次性 |
| --- | --- | --- | --- |
| 规则3: 30抽赠送10连 | cost == 30 | 对每个状态追加 10 次独立抽取的二项分布结果，copies += k | 是 |
| 规则4: 120抽首发硬保底 | cost == 120 且 copies == 0 | 若目标=1则直接毕业；否则 copies 从 0 变为 1 | 是 |
| 规则5: 240抽里程碑赠送 | cost % 240 == 0 | 所有未毕业状态 copies += 1 | 否(循环) |