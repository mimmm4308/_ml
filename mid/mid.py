import random
import numpy as np
import csv
from collections import defaultdict

# 玩家可能的行動：0 = 停牌（Stand），1 = 要牌（Hit）
ACTIONS = [0, 1]

# 抽一張牌（模擬 1~10, J/Q/K 都算 10）
def draw_card():
    return random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

# 計算手牌的總分，並考慮 A 可為 1 或 11 的情況
def hand_value(hand):
    value = sum(hand)
    ace_count = hand.count(1)

    while ace_count > 0 and value + 10 <= 21:
        value += 10
        ace_count -= 1

    return value

# 把手牌狀況轉換為「狀態」，這裡用 (玩家總分, 莊家明牌)
def get_state(player, dealer_card):
    return (hand_value(player), dealer_card)

# Q-Learning 訓練函數
def train_q_learning(episodes=100000, alpha=0.1, gamma=0.95, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))

    for _ in range(episodes):
        player = [draw_card(), draw_card()]
        dealer = [draw_card(), draw_card()]
        state = get_state(player, dealer[0])

        done = False
        while not done:
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = np.argmax(Q[state])

            if action == 1:
                player.append(draw_card())
                next_state = get_state(player, dealer[0])

                if hand_value(player) > 21:
                    reward = -1
                    Q[state][action] += alpha * (reward - Q[state][action])
                    done = True
                else:
                    Q[state][action] += alpha * (0 + gamma * np.max(Q[next_state]) - Q[state][action])
                    state = next_state
            else:
                while hand_value(dealer) < 17:
                    dealer.append(draw_card())

                player_val = hand_value(player)
                dealer_val = hand_value(dealer)

                if dealer_val > 21 or player_val > dealer_val:
                    reward = 1
                elif player_val == dealer_val:
                    reward = 0
                else:
                    reward = -1

                Q[state][action] += alpha * (reward - Q[state][action])
                done = True

    return Q

# 儲存 Q-Table 為 CSV
def save_q_table(Q, filename="q_table.csv"):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["player_total", "dealer_card", "action", "q_value"])
        for state, q_values in Q.items():
            for action, q in enumerate(q_values):
                writer.writerow([state[0], state[1], action, q])

# 根據 Q-table 提供建議
def suggest_action(Q, player_total, dealer_card):
    state = (player_total, dealer_card)
    if state in Q:
        return 1 if np.argmax(Q[state]) == 1 else 0
    else:
        return 1 if player_total < 17 else 0

# 顯示手牌（含 A 的處理）
def display_hand(name, hand):
    print(f"{name} 的手牌: {hand} (總分: {hand_value(hand)})")

# 遊戲主邏輯
def play_game(Q):
    you = [draw_card(), draw_card()]
    dealer = [draw_card(), draw_card()]

    print("--- 新的一局開始 ---")
    display_hand("你", you)
    print(f"莊家的明牌: {dealer[0]}")

    # 玩家回合由你控制，AI 提供建議
    while True:
        action = suggest_action(Q, hand_value(you), dealer[0])
        suggestion = "要牌" if action == 1 else "停牌"
        print(f"AI 建議：{suggestion}")

        decision = input("請輸入你的選擇（h = 要牌, s = 停牌）: ").lower()
        if decision == 'h':
            you.append(draw_card())
            display_hand("你", you)
            if hand_value(you) > 21:
                print("你爆牌了，莊家獲勝！\n")
                return
        elif decision == 's':
            print("你選擇：停牌")
            break
        else:
            print("輸入錯誤，請輸入 h 或 s")

    # 莊家回合
    print("\n--- 莊家回合 ---")
    display_hand("莊家", dealer)
    while hand_value(dealer) < 17:
        print("莊家要牌...")
        dealer.append(draw_card())
        display_hand("莊家", dealer)

    you_total = hand_value(you)
    dealer_total = hand_value(dealer)

    print("\n--- 結果 ---")
    if dealer_total > 21 or you_total > dealer_total:
        print("你獲勝了！\n")
    elif you_total == dealer_total:
        print("平手！\n")
    else:
        print("莊家獲勝！\n")

# 主程式
if __name__ == "__main__":
    print("訓練 Q-Learning Blackjack AI，請稍後...")
    Q = train_q_learning()
    save_q_table(Q)

    # 自動進入遊戲流程
    while True:
        play_game(Q)
        again = input("再玩一局？(y/n): ")
        if again.lower() != 'y':
            break
