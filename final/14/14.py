#使用chatgpt 看過且懂了
import gymnasium as gym

# 固定策略參數
K1 = 2.4   # 竿子的角度（pole_angle）權重
# 竿子往右傾時是正數，往左傾是負數
# 權重越大，越積極修正角度，讓竿子保持直立

K2 = 1.5   # 竿子的角速度（pole_ang_vel）權重
# 表示竿子旋轉的速度，正值代表往右轉，負值往左轉
# 權重越大，越重視減緩旋轉速度，避免過快倒下

K3 = 0.5   # 小車的位置（cart_pos）權重
# 正值表示小車偏右，負值偏左
# 權重中等，有助於將車子拉回中間但不會太激進

K4 = 0.05  # 小車的速度（cart_vel）權重
# 車子移動方向與速度，值較小，只做輕微修正
# 避免車子過快加速，增加系統穩定性

# 建立環境
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

total_steps = 0

while True:
    env.render()

    # 拆解觀測值
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation

    # 計分邏輯
    score = (K1 * pole_angle) + (K2 * pole_ang_vel) + (K3 * cart_pos) + (K4 * cart_vel)
    action = 1 if score > 0 else 0

    # 執行動作
    observation, reward, terminated, truncated, info = env.step(action)
    total_steps += 1

    # 顯示狀態
    print(f"Step {total_steps}: score={score:.3f}, angle={pole_angle:.3f}, ang_vel={pole_ang_vel:.3f}, action={action}")

    if terminated or truncated:
        # 判斷結束原因
        if terminated:
            print(f"❌ 回合結束：竿子倒下或車子出界，共撐了 {total_steps} 步")
        elif truncated:
            print(f"✅ 回合結束：時間到，成功撐滿 {total_steps} 步！")

        # 詢問是否繼續
        while True:
            user_input = input("是否要再開始下一回合？(y/n): ").strip().lower()
            if user_input == 'y':
                observation, info = env.reset()
                total_steps = 0
                break
            elif user_input == 'n':
                print("🚪 結束程式。")
                env.close()
                exit()
            else:
                print("請輸入 y（繼續）或 n（結束）")

env.close()
