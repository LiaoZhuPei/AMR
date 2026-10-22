# 解析信号（Analytic Signal）与希尔伯特变换（Hilbert Transform）详解

解析信号和希尔伯特变换是信号处理、通信、雷达、振动分析以及调制解调中的重要概念。它们的核心思想是：

> 把一个实信号扩展成一个复信号，从而同时描述信号的幅度、相位和瞬时频率。

可以先记住两者的关系：

- 希尔伯特变换：根据原始实信号，构造一个与它相对应的正交分量。
- 解析信号：将原始信号作为实部、希尔伯特变换结果作为虚部，组合成复信号。

下面从直觉、数学公式、频域原理和实际应用逐步讲解。

## 一、先建立直觉：为什么需要解析信号？

假设有一个简单的余弦信号：

\\[ x(t)=A\cos(\omega_0t+\phi) \\]

这是一个实信号，只包含余弦波形。

如果我们希望知道信号在每个时刻的：

- 幅度 \\(A\\)
- 相位 \\(\omega_0t+\phi\\)
- 频率 \\(\omega_0/(2\pi)\\)

就可以借助一个复指数信号：

\\[ z(t)=Ae^{j(\omega_0t+\phi)} \\]

利用欧拉公式：

\\[ e^{j\theta}=\cos\theta+j\sin\theta \\]

展开后得到：

\\[ z(t)=A\cos(\omega_0t+\phi) +jA\sin(\omega_0t+\phi) \\]

因此，原始余弦信号可以看作这个复信号的实部。

### 直观理解：实部与虚部

[Expressions — Lcapy 1.27.dev0 documentation](https://images.openai.com/static-rsc-4/Gd0tvEFRGiGG4hPHY_-LMl8MMuyDhdw9kMacvx_5u_uPLxEQCOgRBU62BG8TLlCkvoTksCwcfvfLXW3prvBay_bmTKKYCQ30Tlc9dSCRtHEpFhtBckYHaCWnk3J3iPgaASno7eSVbEVIr4mYKkFebWlMIwPyeOPBuXGUOOTlxXo?purpose=inline)

实部：余弦分量

\\(x(t)=A\cos\theta(t)\\)

[How a Sine Wave is Calculated, Frequency Peaks Visualized, and Harmonic Distortion Measured Using Calculus and FFT Analysis - HomeTheaterHifi.com](https://images.openai.com/static-rsc-4/2pJRTpsXwwlg3bdOrCp2SaRr3_iBHoXB-2jyW-K77kyaagSoNm0Hzn3XiLVH8xfqVJIyNVpLcxDqA-RD3e2LSwqnUqGdQh14VjvSVZcPY7heoud8Tf9hPKLMMxzSpT2Wuxo8LvX8y4OBZ0zhwjJoiiw41lhtUI6CaDYnypEZ9DQ?purpose=inline)

虚部：正交分量

\\(\hat{x}(t)=A\sin\theta(t)\\)

两个分量相差 \\(90^\circ\\)，可以共同描述信号的旋转相位。

这里的关键是：对于一般的实信号，我们并不知道应该怎样直接构造这个虚部。希尔伯特变换就提供了这样一种方法。

注意，余弦与正弦的这个例子是最简单的情况；对于一般的非正弦信号，解析信号的构造要通过希尔伯特变换来完成。

## 二、什么是希尔伯特变换？

### 1. 数学定义

对于一个实信号 \\(x(t)\\)，其希尔伯特变换通常记为 \\(\hat{x}(t)\\)，定义为：

\\[ \boxed{ \hat{x}(t)=\mathcal{H}\\{x(t)\\} =\frac{1}{\pi}\operatorname{PV} \int\_{-\infty}^{+\infty} \frac{x(\tau)}{t-\tau}\\,d\tau } \\]

其中：

- \\(x(t)\\)：原始信号。
- \\(\hat{x}(t)\\)：希尔伯特变换后的信号。
- \\(\tau\\)：积分变量。
- \\(\operatorname{PV}\\)：柯西主值，用于处理积分在 \\(\tau=t\\) 处的奇异性。

这个公式看起来比较复杂。先不要急着计算积分，可以从它的频域意义理解。

### 2. 最重要的性质：改变相位，不改变幅度谱

采用常见的连续时间傅里叶变换约定：

\\[ X(\omega)=\int\_{-\infty}^{+\infty} x(t)e^{-j\omega t}\\,dt \\]

希尔伯特变换的频域表达式是：

\\[ \boxed{ \hat{X}(\omega)=-j\\,\operatorname{sgn}(\omega)X(\omega) } \\]

其中：

\\[ \operatorname{sgn}(\omega)= \begin{cases} +1,&\omega>0\\\ 0,&\omega=0\\\ -1,&\omega<0 \end{cases} \\]

因此：

\\[ \hat{X}(\omega)= \begin{cases} -jX(\omega),&\omega>0\\\ +jX(\omega),&\omega<0 \end{cases} \\]

这意味着：

| 频率范围 | 希尔伯特变换的效果            |
| ---- | -------------------- |
| 正频率  | 相位偏移 \\(-90^\circ\\) |
| 负频率  | 相位偏移 \\(+90^\circ\\) |
| 幅度谱  | 除零频率等特殊情况外，幅度不变      |

也就是说，希尔伯特变换不是简单地把整个信号延迟四分之一个周期，而是对正、负频率分量施加方向相反的 \\(90^\circ\\) 相移。

这种正负频率的区别非常重要，也是理解解析信号的关键。

### 3. 一个最经典的例子：余弦信号

设：

\\[ x(t)=\cos(\omega_0t),\qquad \omega_0>0 \\]

它的希尔伯特变换为：

\\[ \boxed{\mathcal{H}\\{\cos(\omega_0t)\\} =\sin(\omega_0t)} \\]

同理：

\\[ \boxed{\mathcal{H}\\{\sin(\omega_0t)\\} =-\cos(\omega_0t)} \\]

为什么第一个结果是正弦，而不是负正弦？

因为傅里叶变换中，余弦包含正、负两个频率分量：

\\[ \cos(\omega_0t)= \frac{1}{2}e^{j\omega_0t} +\frac{1}{2}e^{-j\omega_0t} \\]

希尔伯特变换分别处理它们：

\\[ \begin{aligned} \mathcal{H}\\{\cos(\omega_0t)\\} &=\frac{-j}{2}e^{j\omega_0t} +\frac{j}{2}e^{-j\omega_0t}\\\ &=\sin(\omega_0t) \end{aligned} \\]

因此，希尔伯特变换产生了与原信号正交的分量。

## 三、什么是解析信号？

有了希尔伯特变换之后，解析信号的定义就很简单了。

设原始实信号为 \\(x(t)\\)，其希尔伯特变换为 \\(\hat{x}(t)\\)，则解析信号定义为：

\\[ \boxed{ x_a(t)=x(t)+j\hat{x}(t) } \\]

其中：

- 实部是原始信号 \\(x(t)\\)。
- 虚部是希尔伯特变换 \\(\hat{x}(t)\\)。
- \\(j=\sqrt{-1}\\)。

注意：解析信号不是原始信号的另一种叫法，而是由原始信号构造出的复信号。

### 1. 继续使用余弦例子

假设：

\\[ x(t)=A\cos(\omega_0t+\phi) \\]

那么：

\\[ \hat{x}(t)=A\sin(\omega_0t+\phi) \\]

解析信号为：

\\[ \begin{aligned} x_a(t) &=A\cos(\omega_0t+\phi)\\\ &\quad+jA\sin(\omega_0t+\phi) \end{aligned} \\]

根据欧拉公式：

\\[ \boxed{x_a(t)=Ae^{j(\omega_0t+\phi)}} \\]

这是一个非常重要的结果：一个实余弦信号经过解析信号构造后，变成了单边频谱的复指数信号。

### 2. 解析信号的几何意义

可以把复信号看成复平面上的一个旋转向量。

[Sinusoids, Elemental signals, By OpenStax | Jobilize](https://images.openai.com/static-rsc-4/vWtOhADaVaMWF3rlna8MTRgQaEgnJ7-umBFGEt20YkB0VuvQNsHPEZTwycBkvuifAN9uohaJcKH5lOXEs8O45TxcaHilRitMn0CXt0FohxczwvJPVeVrZ5g26N7QOS9Sy-lB38wU5IVJOyomIbutYUkS1iDn-ZopQvAREa7YgR0?purpose=inline)

[jobilize.com](https://www.jobilize.com/course/section/sinusoids-elemental-signals-by-openstax?utm_source=chatgpt.com)

对于 \\(x_a(t)=Ae^{j\theta(t)}\\)：

- 向量长度是 \\(A\\)。
- 向量与实轴的夹角是 \\(\theta(t)\\)。
- 实部是 \\(A\cos\theta(t)\\)。
- 虚部是 \\(A\sin\theta(t)\\)。

因此，解析信号把实信号的幅度和相位统一编码在一个复数中。

对于一般信号，向量长度和旋转速度都可以随时间变化，这就引出了两个非常有用的概念：瞬时幅度与瞬时频率。

## 四、解析信号的三个核心用途

### 1. 提取包络（Envelope）

解析信号的模定义为瞬时幅度或包络：

\\[ \boxed{ A(t)=|x_a(t)| =\sqrt{x^2(t)+\hat{x}^2(t)} } \\]

例如，假设信号是一个幅度调制波：

\\[ x(t)=\bigl[1+0.5\cos(2\pi f_mt)\bigr] \cos(2\pi f_ct) \\]

当载波频率 \\(f_c\\) 远高于调制频率 \\(f_m\\) 时，其解析信号的幅度近似为：

\\[ A(t)=1+0.5\cos(2\pi f_mt) \\]

也就是原始信号外面的慢变化轮廓。

[Wind Farm Noise—Modulation of the Amplitude | MDPI](https://images.openai.com/static-rsc-4/J-SRZJ2w5jA1Myo1Ed5C5hki6vYoZGtySVgv2nO1T7LdJqqSeoKRlEOMTu_O5lI4v8U2j3XH-unqaYWUHH8QqZ7xAI2v152rM9hnx0ri8Z5QNfJJU97gJParXd-5JGUyg44XXVbivArJP4XgZrxHexXAIihRc9EXcduG8xxVcVU?purpose=inline)

[mdpi.com](https://www.mdpi.com/2624-599X/3/2/25?utm_source=chatgpt.com)

实际应用包括：

- AM 调幅信号的包络检测。
- 振动信号的冲击与故障分析。
- 生物电信号的幅度变化分析。
- 地震信号的能量分布分析。

需要注意：对于多分量信号，包络可能发生明显起伏，甚至产生零点；它不一定等同于某个物理系统中直观意义上的振幅。

### 2. 提取瞬时相位（Instantaneous Phase）

解析信号的相位定义为：

\\[ \boxed{ \phi(t)=\arg[x_a(t)] =\operatorname{atan2}\bigl(\hat{x}(t),x(t)\bigr) } \\]

其中 \\(\operatorname{atan2}\\) 可以正确区分不同象限。

对于单频信号：

\\[ x_a(t)=Ae^{j(\omega_0t+\phi_0)} \\]

它的相位就是：

\\[ \phi(t)=\omega_0t+\phi_0 \\]

对于一般信号，相位通常需要进行解缠绕（phase unwrapping），否则相位会在 \\(-\pi\\) 和 \\(\pi\\) 之间跳变。

### 3. 提取瞬时频率（Instantaneous Frequency）

瞬时频率由瞬时相位对时间求导得到：

\\[ \boxed{ f_i(t)=\frac{1}{2\pi}\frac{d\phi(t)}{dt} } \\]

如果：

\\[ x_a(t)=A(t)e^{j\phi(t)} \\]

那么：

\\[ \boxed{ f_i(t)=\frac{1}{2\pi}\frac{d}{dt} \arg[x_a(t)] } \\]

当相位随时间线性增加时，瞬时频率就是常数；当相位的变化速度随时间变化时，瞬时频率也会变化。

例如，对于线性调频信号：

\\[ x(t)=\cos\bigl(2\pi f_0t+\pi kt^2\bigr) \\]

其相位为：

\\[ \phi(t)=2\pi f_0t+\pi kt^2 \\]

求导可得：

\\[ \boxed{f_i(t)=f_0+kt} \\]

这就是雷达、声呐和调频信号分析中经常使用的瞬时频率概念。

但有一个重要限制：当解析信号的幅度接近零，或者信号包含多个相互干扰的频率分量时，瞬时相位和瞬时频率可能变得不稳定，解释时需要谨慎。

## 五、为什么解析信号只有正频率？

这是理解整个理论最值得掌握的部分。

设 \\(x(t)\\) 的傅里叶变换为 \\(X(\omega)\\)，解析信号为：

\\[ x_a(t)=x(t)+j\hat{x}(t) \\]

根据希尔伯特变换的频域性质：

\\[ \hat{X}(\omega) =-j\operatorname{sgn}(\omega)X(\omega) \\]

因此：

\\[ X_a(\omega) =X(\omega)+j\hat{X}(\omega) \\]

代入得到：

\\[ \boxed{ X_a(\omega)= \begin{cases} 2X(\omega),&\omega>0\\\ X(0),&\omega=0\\\ 0,&\omega<0 \end{cases} } \\]

这里的零频分量按通常的连续频谱表达处理；对于离散频谱或特殊信号，直流与奈奎斯特频率等分量需单独考虑。

观察这个结果：

- 正频率：幅度加倍。
- 负频率：完全消除。
- 零频率：保留。

为什么正频率会加倍？

因为原始实信号的正、负频率成对出现。构造解析信号时，希尔伯特变换产生的虚部与原始信号组合，使负频率相互抵消，而正频率相互叠加。

### 频谱变化示意

实余弦信号

解析信号

示意图只展示正、负频率的变化，不代表具体信号的完整频谱。

这也是为什么解析信号被称为 analytic signal：它在频域上具有单边频谱的特征，并与复变量理论中的解析性相联系。

## 六、实际计算：如何在 Python 中得到解析信号？

在工程中，通常不直接计算希尔伯特变换的积分，而是使用 FFT 或现成的数字信号处理函数。

例如，Python 的 SciPy 库提供了 `scipy.signal.hilbert`。

```
import numpy as npimport matplotlib.pyplot as pltfrom scipy.signal import hilbert# 采样参数fs = 1000t = np.arange(0, 1, 1 / fs)# 生成一个幅度调制信号x = (1 + 0.5 * np.cos(2 * np.pi * 3 * t)) * \    np.cos(2 * np.pi * 100 * t)# 解析信号xa = hilbert(x)# 提取包络、相位和瞬时频率envelope = np.abs(xa)phase = np.unwrap(np.angle(xa))instantaneous_frequency = np.gradient(phase, t) / (2 * np.pi)# 绘图plt.figure(figsize=(10, 7))plt.subplot(3, 1, 1)plt.plot(t, x)plt.title("Original Signal")plt.subplot(3, 1, 2)plt.plot(t, envelope)plt.title("Analytic Signal Envelope")plt.subplot(3, 1, 3)plt.plot(t, instantaneous_frequency)plt.title("Instantaneous Frequency")plt.tight_layout()plt.show()
```

这段代码完成三个步骤：

1. `hilbert(x)`：构造解析信号。
2. `np.abs(xa)`：提取包络。
3. `np.unwrap(np.angle(xa))`：提取连续相位，再通过数值微分计算瞬时频率。

实际处理有限长度的采样数据时，边界效应、噪声以及相位微分会影响结果。对于瞬时频率，通常还需要考虑滤波、平滑或适当的信号分解。

## 七、最后总结：两者到底有什么区别？

| 对比项  | 希尔伯特变换                                | 解析信号                          |
| ---- | ------------------------------------- | ----------------------------- |
| 数学形式 | \\(\hat{x}(t)=\mathcal H\\{x(t)\\}\\) | \\(x_a(t)=x(t)+j\hat{x}(t)\\) |
| 本质   | 一种信号变换                                | 一种复信号表示                       |
| 主要作用 | 构造正交分量                                | 联合表示幅度与相位                     |
| 频域特征 | 正、负频率分别相移 \\(\mp90^\circ\\)           | 消除负频率，保留正频率                   |
| 常见用途 | 正交分量生成、相位关系分析                         | 包络、瞬时相位、瞬时频率分析                |

最重要的三个公式是：

\\[ \boxed{\hat{x}(t)=\mathcal H\\{x(t)\\}} \\]

\\[ \boxed{x_a(t)=x(t)+j\hat{x}(t)} \\]

\\[ \boxed{ A(t)=|x_a(t)|,\qquad f_i(t)=\frac{1}{2\pi}\frac{d}{dt}\arg[x_a(t)] } \\]

一句话记忆：希尔伯特变换负责构造正交分量，解析信号负责把原信号和正交分量组合起来，从而让幅度、相位和瞬时频率的分析变得更加方便。