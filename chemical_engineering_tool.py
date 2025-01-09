import streamlit as st
import numpy as np
from scipy.optimize import fsolve

# 應用標題
st.title("化工單元操作計算工具")

# 選單
options = [
    "氣體吸收塔填充高度計算",
    "Kremser 理論板數計算",
    "蒸餾塔設計 (Underwood, Fenske, Gilliland)",
    "等溫閃蒸分離計算",
    "昇華速率計算",
    "質量傳遞係數計算"
]
selected_option = st.sidebar.selectbox("選擇計算功能", options)

# 主應用程式
class MassTransferApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("化工單元操作計算工具")
        self.geometry("600x500")
        tk.Label(self, text="選擇計算功能", font=("Helvetica", 18)).pack(pady=20)

        # 功能按鈕
        functions = [
            ("氣體吸收塔填充高度計算", self.absorption_tower),
            ("Kremser 理論板數計算", self.kremser_calculation),
            ("蒸餾塔設計 (Underwood, Fenske, Gilliland)", self.distillation_column),
            ("等溫閃蒸分離計算", self.isothermal_flash),
            ("昇華速率計算", self.sublimation_rate),
            ("質量傳遞係數計算", self.mass_transfer_coefficient)
        ]

        for text, func in functions:
            tk.Button(self, text=text, command=func, width=40).pack(pady=10)

    # (1) 氣體吸收塔填充高度計算
    def absorption_tower(self):
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
        import numpy as np

        # 計算函數
        def calculate_absorption_tower():
            try:
                # 讀取輸入數據
                H_X = float(entry_HX.get())  # 液膜單位傳質高度 (m)
                H_Y = float(entry_HY.get())  # 氣膜單位傳質高度 (m)
                y_top = float(entry_y1.get())  # 塔頂氣相分率
                y_bottom = float(entry_y2.get())  # 塔底氣相分率
                x_top = float(entry_x1.get())  # 塔頂液相分率
                x_bottom = float(entry_x2.get())  # 塔底液相分率

                # 步驟 1：求 L/G 比值
                delta_y = y_bottom - y_top
                delta_x = x_bottom - x_top
                L_over_G = delta_y / delta_x  # L/G 比值
                steps = []
                steps.append(f"步驟 1：求 L/G 比值")
                steps.append(f"Δy = y_bottom - y_top = {y_bottom:.4f} - {y_top:.4f} = {delta_y:.4f}")
                steps.append(f"Δx = x_bottom - x_top = {x_bottom:.4f} - {x_top:.4f} = {delta_x:.4f}")
                steps.append(f"L/G = Δy / Δx = {delta_y:.4f} / {delta_x:.4f} = {L_over_G:.4f}\n")

                # 步驟 2：積分求總體氣相傳質單位數 N_OG
                m = 0.06  # 平衡線斜率
                steps.append("步驟 2：計算總體氣相傳質單位數 N_OG")
                steps.append(f"平衡線斜率 m = {m:.4f}")

                # 使用對數近似公式來計算 N_OG
                z_top = 0.994 * y_top + 6e-6
                z_bottom = 0.994 * y_bottom + 6e-6
                N_OG = (1 / 0.994) * np.log(z_bottom / z_top)
                steps.append(f"z_top = 0.994 * y_top + 6e-6 ≈ {z_top:.6f}")
                steps.append(f"z_bottom = 0.994 * y_bottom + 6e-6 ≈ {z_bottom:.6f}")
                steps.append(f"N_OG ≈ (1 / 0.994) * log(z_bottom / z_top)")
                steps.append(f"N_OG ≈ (1 / 0.994) * log({z_bottom:.6f} / {z_top:.6f})")
                steps.append(f"N_OG ≈ {N_OG:.4f}\n")

                # 步驟 3：計算總體氣相傳質單位高度 H_OG
                inverse_H_OG = (1 / H_Y) + m * L_over_G / H_X
                H_OG = 1 / inverse_H_OG
                steps.append("步驟 3：計算總體氣相傳質單位高度 H_OG")
                steps.append(f"(1 / H_OG) = (1 / H_Y) + m * (L/G) / H_X")
                steps.append(f"(1 / H_OG) = (1 / {H_Y:.4f}) + {m:.4f} * ({L_over_G:.4f}) / {H_X:.4f}")
                steps.append(f"(1 / H_OG) ≈ {1 / H_Y:.4f} + {m * L_over_G / H_X:.4f}")
                steps.append(f"H_OG ≈ {H_OG:.4f} m\n")

                # 步驟 4：計算填充塔高度 Z_tower
                Z_tower = N_OG * H_OG
                steps.append("步驟 4：計算填充塔高度 Z_tower")
                steps.append(f"Z_tower = N_OG * H_OG")
                steps.append(f"Z_tower ≈ {N_OG:.4f} * {H_OG:.4f}")
                steps.append(f"Z_tower ≈ {Z_tower:.4f} m\n")

                # 顯示結果
                result_window = tk.Toplevel()
                result_window.title("吸收塔計算結果")
                scrolled_text = scrolledtext.ScrolledText(result_window, width=80, height=30)
                scrolled_text.pack(padx=10, pady=10)
                scrolled_text.insert(tk.END, "\n".join(steps))
                scrolled_text.insert(tk.END, f"\n\n最終結果：填充塔高度 Z ≈ {Z_tower:.2f} m")
                scrolled_text.config(state='disabled')

            except ValueError:
                messagebox.showerror("輸入錯誤", "請輸入有效的數值！")

        # GUI 設計
        root = tk.Tk()
        root.title("氣體吸收塔填充高度計算器")

        # 標題
        tk.Label(root, text="氣體吸收塔計算工具", font=("Helvetica", 16)).pack(pady=10)

        # 參數輸入框
        fields = [
            ("液膜單位傳質高度 H_X (m)", "0.24"),
            ("氣膜單位傳質高度 H_Y (m)", "0.36"),
            ("塔頂氣相分率 y_top", "0.001"),
            ("塔底氣相分率 y_bottom", "0.009"),
            ("塔頂液相分率 x_top", "0.0"),
            ("塔底液相分率 x_bottom", "0.08")
        ]

        entries = {}
        for label_text, default_value in fields:
            frame = tk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=label_text, width=25).pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, default_value)
            entry.pack(side="left", fill="x", expand=True)
            entries[label_text] = entry

        entry_HX = entries["液膜單位傳質高度 H_X (m)"]
        entry_HY = entries["氣膜單位傳質高度 H_Y (m)"]
        entry_y1 = entries["塔頂氣相分率 y_top"]
        entry_y2 = entries["塔底氣相分率 y_bottom"]
        entry_x1 = entries["塔頂液相分率 x_top"]
        entry_x2 = entries["塔底液相分率 x_bottom"]

        # 計算按鈕
        tk.Button(root, text="計算", command=calculate_absorption_tower).pack(pady=20)

        # 主循環
        root.geometry("600x500")
        root.mainloop()

    # (2) Kremser 方程式
    def kremser_calculation(self):
        import tkinter as tk
        from tkinter import messagebox, scrolledtext

        # 計算函數
        def calculate_kremser():
            try:
                # 讀取輸入參數
                m = float(entry_m.get())  # 平衡常數 m
                efficiency = float(entry_efficiency.get()) / 100  # 總板效率
                stages = int(entry_stages.get())  # 理論板數 N
                removal_target = float(entry_removal_target.get()) / 100  # 目標移除率 (% -> 小數)

                steps = []  # 儲存計算步驟

                # (a) 最小氣液流率比 (L/V)_min
                L_V_min = 1 / m  # 最小流率比
                steps.append(f"(a) 最小氣液流率比 (L/V)_min = 1 / m = 1 / {m:.2f} = {L_V_min:.4f}")

                # (b) 實際流率比 L/V
                actual_L_V = 1.5 * L_V_min  # 實際流率比
                steps.append(f"(b) 實際氣液流率比 L/V = 1.5 × (L/V)_min = 1.5 × {L_V_min:.4f} = {actual_L_V:.4f}")

                # 計算 λ 值
                lambda_val = actual_L_V / m
                steps.append(f"(b) λ = L/V ÷ 平衡常數 m = {actual_L_V:.4f} ÷ {m:.4f} = {lambda_val:.4f}")

                # 使用 Kremser 方程式計算移除效率 E
                kremser_E = 1 - ((1 - lambda_val) ** (stages + 1) / (1 - lambda_val ** (stages + 1)))
                kremser_E_percentage = kremser_E * 100
                steps.append(f"(b) 使用 Kremser 方程式：")
                steps.append(f"    E = 1 - ((1 - λ)^(N + 1)) / (1 - λ^(N + 1))")
                steps.append(f"    E = 1 - ((1 - {lambda_val:.4f})^{stages + 1}) / (1 - {lambda_val:.4f}^{stages + 1})")
                steps.append(f"    E = {kremser_E_percentage:.2f}%")

                # (c) 當氣相流率 V 增加時，λ 減小，E 增加
                new_L_V = actual_L_V / 1.5  # 假設 V 增加到原來的 1.5 倍，L/V 減少
                new_lambda_val = new_L_V / m  # 新的 λ
                new_removal_rate = 1 - ((1 - new_lambda_val) ** (stages + 1) / (1 - new_lambda_val ** (stages + 1)))
                new_removal_rate_percentage = new_removal_rate * 100
                steps.append(f"(c) 當氣相流率增加時：")
                steps.append(f"    新的 λ = {new_L_V:.4f} ÷ {m:.4f} = {new_lambda_val:.4f}")
                steps.append(f"    新的移除效率 E = {new_removal_rate_percentage:.2f}%")

                # 顯示結果
                result_window = tk.Toplevel()
                result_window.title("Kremser 計算結果")
                scrolled_text = scrolledtext.ScrolledText(result_window, width=70, height=25)
                scrolled_text.pack(padx=10, pady=10)
                scrolled_text.insert(tk.END, "\n\n".join(steps))
                scrolled_text.insert(tk.END, f"\n\n最終結果：原始移除率 ≈ {kremser_E_percentage:.2f}%，新移除率 ≈ {new_removal_rate_percentage:.2f}%")
                scrolled_text.config(state='disabled')

            except ValueError:
                messagebox.showerror("輸入錯誤", "請輸入有效的數值！")

        # GUI 設計
        root = tk.Tk()
        root.title("Kremser 方程式計算器")

        # 標題
        tk.Label(root, text="Kremser 方程式計算工具", font=("Helvetica", 16)).pack(pady=10)

        # 參數輸入框
        fields = [
            ("平衡常數 m", "0.8"),
            ("總板效率 (%)", "75"),
            ("理論板數 (N)", "8"),
            ("目標移除率 (%)", "95")
        ]

        entries = {}
        for label_text, default_value in fields:
            frame = tk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=label_text, width=25).pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, default_value)
            entry.pack(side="left", fill="x", expand=True)
            entries[label_text] = entry

        entry_m = entries["平衡常數 m"]
        entry_efficiency = entries["總板效率 (%)"]
        entry_stages = entries["理論板數 (N)"]
        entry_removal_target = entries["目標移除率 (%)"]

        # 計算按鈕
        tk.Button(root, text="計算", command=calculate_kremser).pack(pady=20)

        # 主循環
        root.geometry("500x400")
        root.mainloop()

    # (3) 蒸餾塔設計 (Underwood, Fenske, Gilliland)
    def distillation_column(self):
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
        import numpy as np
        from scipy.optimize import fsolve

        # 計算函數
        def calculate_distillation():
            try:
                # 讀取輸入參數
                alpha = float(entry_alpha.get())  # 相對揮發度
                x_D = float(entry_xD.get())  # 塔頂純度
                x_B = float(entry_xB.get())  # 塔底純度
                z_F = float(entry_zF.get())  # 進料組成

                steps = []  # 儲存步驟

                # (a) 最小回流比 R_min (Underwood 方法)
                theta = alpha * x_D
                R_min = x_D / (theta * (1 - x_D))
                steps.append(f"(a) 最小回流比 R_min 計算：")
                steps.append(f"    Underwood 方法 θ = α * x_D = {alpha:.2f} * {x_D:.2f} = {theta:.4f}")
                steps.append(f"    R_min = x_D / [θ * (1 - x_D)]")
                steps.append(f"    R_min = {x_D:.2f} / [{theta:.4f} * (1 - {x_D:.2f})]")
                steps.append(f"    R_min ≈ {R_min:.4f}\n")

                # (b) 最小理論板數 N_min (Fenske 方程式)
                N_min = np.log((x_D / (1 - x_D)) * ((1 - x_B) / x_B)) / np.log(alpha)
                steps.append(f"(b) 最小理論板數 N_min 計算：")
                steps.append(f"    Fenske 方程式：")
                steps.append(f"    N_min = log[(x_D / (1 - x_D)) * ((1 - x_B) / x_B)] / log(α)")
                steps.append(f"    N_min = log[({x_D:.2f} / (1 - {x_D:.2f})) * ((1 - {x_B:.2f}) / {x_B:.2f})] / log({alpha:.2f})")
                steps.append(f"    N_min ≈ {N_min:.2f}\n")

                # (c) 實際板數 N_actual (Gilliland 方程式)
                R_actual = 2 * R_min
                steps.append(f"(c) 實際板數 N_actual 計算：")
                steps.append(f"    實際回流比 R = 2 * R_min = 2 * {R_min:.4f} = {R_actual:.4f}")

                def gilliland(N_actual):
                    left_side = (N_actual - N_min) / (N_actual + 1)
                    right_side = 0.75 * ((R_actual - R_min) / (R_actual + 1))
                    return left_side - right_side

                N_actual = fsolve(gilliland, N_min + 5)[0]  # 初始猜測值
                steps.append(f"    Gilliland 方程式求解：")
                steps.append(f"    N_actual ≈ {N_actual:.2f}\n")

                # 顯示完整計算過程與結果
                result_window = tk.Toplevel()
                result_window.title("Distillation 計算結果")
                scrolled_text = scrolledtext.ScrolledText(result_window, width=80, height=25)
                scrolled_text.pack(padx=10, pady=10)
                scrolled_text.insert(tk.END, "\n".join(steps))
                scrolled_text.insert(tk.END, f"\n最終結果：\n(a) 最小回流比 R_min ≈ {R_min:.4f}\n(b) 最小理論板數 N_min ≈ {N_min:.2f}\n(c) 實際板數 N_actual ≈ {N_actual:.2f}")
                scrolled_text.config(state='disabled')

            except ValueError:
                messagebox.showerror("輸入錯誤", "請輸入有效的數值！")

        # GUI 設計
        root = tk.Tk()
        root.title("Distillation 計算工具")

        # 標題
        tk.Label(root, text="Distillation Column 計算工具", font=("Helvetica", 16)).pack(pady=10)

        # 參數輸入框
        fields = [
            ("相對揮發度 α", "2"),
            ("塔頂純度 x_D", "0.9"),
            ("塔底純度 x_B", "0.1"),
            ("進料組成 z_F", "0.5")
        ]

        entries = {}
        for label_text, default_value in fields:
            frame = tk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=label_text, width=20).pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, default_value)
            entry.pack(side="left", fill="x", expand=True)
            entries[label_text] = entry

        entry_alpha = entries["相對揮發度 α"]
        entry_xD = entries["塔頂純度 x_D"]
        entry_xB = entries["塔底純度 x_B"]
        entry_zF = entries["進料組成 z_F"]

        # 計算按鈕
        tk.Button(root, text="計算", command=calculate_distillation).pack(pady=20)

        # 主循環
        root.geometry("600x400")
        root.mainloop()


    # (4) 等溫閃蒸分離計算
    def isothermal_flash(self):
        import tkinter as tk
        from tkinter import Toplevel, Text, Scrollbar, END
        import mathe

        # 閃蒸計算函數
        def flash_calculation(F, z, K):
            """
            計算閃蒸氣化率 β 及氣相和液相的組成和流率，並返回詳細計算過程
            """
            # 定義求解 β 的方程
            def equation(beta):
                return sum([z[i] * (K[i] - 1) / ((1 - beta) + beta * K[i]) for i in range(len(z))]) - 1

            # 初始猜測 β = 0.5
            beta_initial = 0.5
            beta_solution = fsolve(equation, beta_initial)[0]

            # 計算液相和氣相組成
            x = [z[i] / ((1 - beta_solution) + beta_solution * K[i]) for i in range(len(z))]
            y = [K[i] * x[i] for i in range(len(z))]

            # 計算氣相與液相流率
            V = beta_solution * F
            L = F - V

            # 詳細計算過程
            step_details = "**詳細計算過程**:\n\n"
            step_details += f"**1. 氣化率 β 的計算：**\n"
            step_details += "根據公式：\n"
            step_details += "∑ [z_i * (K_i - 1) / ((1 - β) + β * K_i)] = 1\n\n"
            for i in range(len(z)):
                numerator = z[i] * (K[i] - 1)
                denominator = (1 - beta_solution) + beta_solution * K[i]
                contribution = numerator / denominator
                step_details += f"組分 {i + 1}: [z_{i + 1} * (K_{i + 1} - 1) / ((1 - β) + β * K_{i + 1})] = {contribution:.4f}\n"
            step_details += f"\n求解得到 β = {beta_solution:.4f}\n\n"

            step_details += "**2. 液相摩爾分率 x_i 的計算：**\n"
            for i in range(len(x)):
                step_details += f"x_{i + 1} = z_{i + 1} / ((1 - β) + β * K_{i + 1}) = {x[i]:.4f}\n"

            step_details += "\n**3. 氣相摩爾分率 y_i 的計算：**\n"
            for i in range(len(y)):
                step_details += f"y_{i + 1} = K_{i + 1} * x_{i + 1} = {y[i]:.4f}\n"

            step_details += "\n**4. 氣相與液相流率的計算：**\n"
            step_details += f"氣相流率 V = β * F = {beta_solution:.4f} * {F} = {V:.2f} kmol/h\n"
            step_details += f"液相流率 L = F - V = {F} - {V:.2f} = {L:.2f} kmol/h\n"

            return beta_solution, V, L, y, x, step_details

        # Streamlit 應用程式
        st.title("等溫閃蒸計算應用程式")

        # 輸入參數
        st.sidebar.header("輸入參數")
        F = st.sidebar.number_input("進料流率 F (kmol/h)", min_value=1, value=1000, step=1)
        num_components = st.sidebar.number_input("組分數量", min_value=2, value=4, step=1)

        st.sidebar.subheader("進料組成 z (摩爾分率)")
        z = []
        for i in range(num_components):
            z_i = st.sidebar.number_input(f"組分 {i + 1} 摩爾分率", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
            z.append(z_i)

        # 驗證摩爾分率總和
        if abs(sum(z) - 1.0) > 0.01:
            st.warning("進料組成的摩爾分率總和必須等於 1。請重新輸入！")
        else:
            st.sidebar.subheader("平衡常數 K 值")
            K = []
            for i in range(num_components):
                K_i = st.sidebar.number_input(f"組分 {i + 1} 平衡常數 K", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                K.append(K_i)

            # 計算閃蒸
            if st.sidebar.button("開始計算"):
                beta, V, L, y, x, steps = flash_calculation(F, z, K)
                
                # 顯示結果
                st.subheader("計算結果")
                st.write(f"**氣化率 β:** {beta:.4f}")
                st.write(f"**氣相流率 V:** {V:.2f} kmol/h")
                st.write(f"**液相流率 L:** {L:.2f} kmol/h")

                # 顯示氣相與液相組成
                st.subheader("氣相組成 y")
                for i, yi in enumerate(y):
                    st.write(f"組分 {i + 1}: {yi:.4f}")
                
                st.subheader("液相組成 x")
                for i, xi in enumerate(x):
                    st.write(f"組分 {i + 1}: {xi:.4f}")

                # 顯示詳細計算過程
                st.subheader("詳細計算過程")
                st.markdown(steps)

    # (5) 昇華速率計算
    def sublimation_rate(self):
        import tkinter as tk
        from tkinter import messagebox, scrolledtext

        # 計算邏輯
        def calculate_and_show_steps(d, u, P_A, T, rho_s, M_A, D_AB, nu, R):
            steps = []  # 儲存計算步驟文字

            # 1. 計算 Reynolds 數
            Re = (u * d) / nu
            steps.append(f"1. 計算 Reynolds 數:\nRe = u * d / ν = {u:.2f} * {d:.4f} / {nu:.2e} = {Re:.2f}")

            # 2. 計算 Schmidt 數
            Sc = nu / D_AB
            steps.append(f"2. 計算 Schmidt 數:\nSc = ν / D_AB = {nu:.2e} / {D_AB:.2e} = {Sc:.2f}")

            # 3. 計算 Sherwood 數
            Sh = 2 + 0.552 * (Re ** 0.5) * (Sc ** (1/3))
            steps.append(f"3. 計算 Sherwood 數:\nSh = 2 + 0.552 * Re^0.5 * Sc^1/3\nSh = 2 + 0.552 * {Re:.2f}^0.5 * {Sc:.2f}^1/3 = {Sh:.2f}")

            # 4. 計算質傳係數 k_c
            k_c = Sh * D_AB / d
            steps.append(f"4. 計算質傳係數 k_c:\nk_c = Sh * D_AB / d = {Sh:.2f} * {D_AB:.2e} / {d:.4f} = {k_c:.6f} m/s")

            # 5. 計算升華速率 N_A
            N_A = k_c * P_A / (R * T)
            steps.append(f"5. 計算升華速率 N_A:\nN_A = k_c * P_A / (R * T)\nN_A = {k_c:.6f} * {P_A:.2f} / ({R:.2f} * {T:.2f}) = {N_A:.6e} mol/m²·s")

            # 顯示結果視窗
            result_window = tk.Toplevel()
            result_window.title("計算結果與過程")
            scrolled_text = scrolledtext.ScrolledText(result_window, width=60, height=20)
            scrolled_text.pack(padx=10, pady=10)
            scrolled_text.insert(tk.END, "\n\n".join(steps))
            scrolled_text.config(state='disabled')

        # 計算按鈕功能
        def on_calculate():
            try:
                # 讀取使用者輸入
                d = float(entry_d.get()) / 100  # cm 轉換為 m
                u = float(entry_u.get())  # 流速 (m/s)
                P_A = float(entry_P_A.get())  # 蒸氣壓 (Pa)
                T = float(entry_T.get())  # 絕對溫度 (K)
                rho_s = float(entry_rho_s.get())  # 密度 (kg/m³)
                M_A = float(entry_M_A.get())  # 分子量 (g/mol)
                D_AB = float(entry_D_AB.get())  # 擴散係數 (m²/s)
                nu = float(entry_nu.get())  # 動力黏度 (m²/s)
                R = float(entry_R.get())  # 氣體常數 (Pa·m³/mol·K)

                # 執行計算
                calculate_and_show_steps(d, u, P_A, T, rho_s, M_A, D_AB, nu, R)
            except ValueError:
                messagebox.showerror("輸入錯誤", "請輸入有效的數值！")

        # GUI 介面設計
        root = tk.Tk()
        root.title("升華速率計算器")

        # 標題
        tk.Label(root, text="升華速率計算工具", font=("Helvetica", 16)).pack(pady=10)

        # 參數輸入框
        fields = [
            ("萘球直徑 d (cm)", "1.75"),
            ("氣流速度 u (m/s)", "1.4"),
            ("蒸氣壓 P_A (Pa)", "2.8"),
            ("溫度 T (K)", "298"),
            ("萘密度 ρ_s (kg/m³)", "1140"),
            ("分子量 M_A (g/mol)", "128"),
            ("擴散係數 D_AB (m²/s)", "5.57e-6"),
            ("動力黏度 ν (m²/s)", "1.38e-5"),
            ("氣體常數 R (Pa·m³/mol·K)", "8.314")
        ]

        entries = {}
        for label_text, default_value in fields:
            frame = tk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=label_text, width=20).pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, default_value)
            entry.pack(side="left", fill="x", expand=True)
            entries[label_text] = entry

        entry_d, entry_u, entry_P_A, entry_T, entry_rho_s, entry_M_A, entry_D_AB, entry_nu, entry_R = entries.values()

        # 計算按鈕
        tk.Button(root, text="計算", command=on_calculate).pack(pady=20)

        # 主循環
        root.geometry("500x500")
        root.mainloop()

    # (6) 質量傳遞係數計算
    def mass_transfer_coefficient(self):
        import tkinter as tk
        from tkinter import messagebox, scrolledtext

        # 計算函數
        def calculate_and_show_steps(u, x, nu, D_AB, alpha, rho, cp):
            steps = []

            # Reynolds 數計算
            Re_x = u * x / nu
            steps.append(f"1. 計算 Reynolds 數:\nRe_x = u * x / ν = {u:.2f} * {x:.2f} / {nu:.2e} = {Re_x:.2e}")

            # Schmidt 數計算
            Sc = nu / D_AB
            steps.append(f"2. 計算 Schmidt 數:\nSc = ν / D_AB = {nu:.2e} / {D_AB:.2e} = {Sc:.2f}")

            # 流態判斷並計算 Sherwood 數
            if Re_x < 2e5:
                Sh = 0.332 * (Re_x ** 0.5) * (Sc ** 0.33)
                flow_type = "層流"
                steps.append(f"3. 屬於層流區域 (Re_x < 2×10⁵)，使用層流公式:\nSh = 0.332 * Re_x^0.5 * Sc^0.33")
            else:
                Sh = 0.0292 * (Re_x ** 0.8) * (Sc ** 0.33)
                flow_type = "湍流"
                steps.append(f"3. 屬於湍流區域 (Re_x >= 2×10⁵)，使用湍流公式:\nSh = 0.0292 * Re_x^0.8 * Sc^0.33")

            steps.append(f"Sh = {Sh:.2f} ({flow_type})")

            # 質傳係數 k 計算
            k = Sh * D_AB / x
            steps.append(f"4. 計算質傳係數 k:\nk = Sh * D_AB / x = {Sh:.2f} * {D_AB:.2e} / {x:.2f} = {k:.6f} ft/s")

            # 熱傳係數 h 計算
            h = k * rho * cp
            steps.append(f"5. 利用 Colburn 類比計算熱傳係數 h:\nh = k * ρ * c_p = {k:.6f} * {rho:.4f} * {cp:.2f} = {h:.4f} Btu/(ft²·s·°F)")

            # 顯示步驟和結果
            result_window = tk.Toplevel()
            result_window.title("計算結果與步驟")
            scrolled_text = scrolledtext.ScrolledText(result_window, width=60, height=20)
            scrolled_text.pack(padx=10, pady=10)
            scrolled_text.insert(tk.END, "\n\n".join(steps))
            scrolled_text.config(state='disabled')  # 禁止編輯

        def on_calculate():
            try:
                # 讀取使用者輸入
                u = float(entry_u.get())
                x = float(entry_x.get())
                nu = float(entry_nu.get())
                D_AB = float(entry_D_AB.get())
                alpha = float(entry_alpha.get())
                rho = float(entry_rho.get())
                cp = float(entry_cp.get())

                calculate_and_show_steps(u, x, nu, D_AB, alpha, rho, cp)
            except ValueError:
                messagebox.showerror("輸入錯誤", "請輸入有效的數值！")

        # GUI 介面
        root = tk.Tk()
        root.title("質量傳遞與熱傳遞計算器")

        # 標題
        tk.Label(root, text="質量與熱傳遞計算工具", font=("Helvetica", 16)).pack(pady=10)

        # 參數輸入框
        fields = [
            ("流速 u (ft/s)", "15"),
            ("距離 x (ft)", "4.5"),
            ("動力黏度 ν (ft²/s)", "1.81e-4"),
            ("擴散係數 D_AB (ft²/s)", "2.81e-4"),
            ("熱擴散係數 α (ft²/s)", "2.37e-4"),
            ("空氣密度 ρ (lbm/ft³)", "0.0735"),
            ("比熱 c_p (Btu/(lbm)(°F))", "0.24")
        ]

        entries = {}
        for label_text, default_value in fields:
            frame = tk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            tk.Label(frame, text=label_text, width=20).pack(side="left")
            entry = tk.Entry(frame)
            entry.insert(0, default_value)
            entry.pack(side="left", fill="x", expand=True)
            entries[label_text] = entry

        entry_u, entry_x, entry_nu, entry_D_AB, entry_alpha, entry_rho, entry_cp = entries.values()

        # 計算按鈕
        tk.Button(root, text="計算", command=on_calculate).pack(pady=20)

        # 主循環
        root.geometry("450x400")
        root.mainloop()

# 啟動應用程式
if __name__ == "__main__":
    app = MassTransferApp()
    app.mainloop()
