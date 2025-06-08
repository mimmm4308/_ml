# 基於Q-Learning 來遊玩21點 使用chatgpt 看過且懂了 有手動修改參數

## 使用chatgpt 看過且懂了 有手動修改參數

## 目標

實作一個基於 Q-Learning 的 AI，學會在 21 點（Blackjack）遊戲中根據局勢選擇最佳行動（Hit 或 Stand），並提供與玩家互動的模擬器。

---

## 🧠 Q-Learning 原理簡介

Q-Learning 是一種強化學習算法，透過以下方式學習策略：

* 建立 Q-Table：紀錄每個「狀態」下每個「行動」的期望報酬（Q 值）。
* 使用公式更新 Q 值：
  $Q(s,a) \leftarrow Q(s,a) + \alpha (r + \gamma \max_a' Q(s',a') - Q(s,a))$

  * $\alpha$：學習率
  * $\gamma$：未來報酬折扣率
  * $r$：當前獎勵
  * $s$、$a$：當前狀態與動作
  * $s'$：下個狀態

---

## 🃏 遊戲規則

* 玩家與莊家初始都抽兩張牌。
* 玩家可選擇要牌（Hit）或停牌（Stand）。
* A 可作 1 或 11。
* 莊家若小於 17 點必須要牌。
* 遊戲目標為盡量接近 21 點但不能爆牌（超過 21）。

---

## 🏗️ 程式結構與流程

### 1. `draw_card()`

模擬從牌堆抽一張牌（1\~10），其中 J/Q/K 都視為 10。

### 2. `hand_value(hand)`

計算手牌總分，考慮 A 可轉為 11 的加分策略。

### 3. `get_state(player, dealer_card)`

將牌局狀態簡化為 `(玩家總分, 莊家明牌)` 作為 Q-Table 的鍵。

### 4. `train_q_learning(...)`

使用 Q-Learning 演算法訓練 AI：

* 反覆模擬 `episodes` 局隨機牌局。
* 透過 ε-greedy 策略探索/利用行動選擇。
* 依照輸贏或爆牌結果調整 Q 值。

### 5. `save_q_table(...)`

將訓練好的 Q-Table 儲存為 CSV 檔案。

### 6. `suggest_action(Q, player_total, dealer_card)`

給定目前局勢，從 Q-Table 中選擇建議行動（Hit 或 Stand）。若無資料則依簡單規則（小於 17 要牌）。

### 7. `play_game(Q)`

完整模擬一局遊戲：

* 玩家依照 AI 策略行動。
* 莊家根據規則行動。
* 顯示結果與勝負。

### 8. `main`

* 執行 Q-Learning 訓練。
* 進入無限回圈與玩家進行遊戲直到選擇退出。

---

## 🧪 訓練參數說明

* `episodes=500000`：訓練局數越多，策略越成熟。
* `alpha=0.1`：學習率（控制 Q 值更新速度）。
* `gamma=0.95`：未來獎勵的重要性。
* `epsilon=0.1`：10% 機率隨機探索，90% 選最佳行動。

---

## 📌 小結

這個 Blackjack AI 使用 Q-Learning 成功學會了在不同局勢中選擇「要牌」或「停牌」。玩家可透過互動介面實際與 AI 對戰，觀察其學習後的表現。