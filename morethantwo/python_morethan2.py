import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.float_format', '{:.10f}'.format)

# 数据
#A = [20, 22, 29, 21, 18]
#B = [15, 14, 16, 13, 17]
#C = [10, 12, 11, 9, 13]

A = np.array([20, 22, 29, 21, 18])  # 替换为你的数据
B = np.array([15, 14, 16, 13, 17])  # 替换为你的数据
C = np.array([10, 12, 11, 9, 13]) 


# 将数据组合成一个列表
data = [A, B, C]
# 将组标签组合成一个列表
groups = ['A', 'B', 'C']

# 将数据转换为 DataFrame
df = pd.DataFrame({
    'value': np.concatenate([A, B, C]),
    'group': np.repeat(groups, [len(A), len(B), len(C)])
})

# 使用 statsmodels 进行单因素方差分析
model = ols('value ~ group', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)  # typ=2 表示使用 Type II 平方和

# 输出 ANOVA 表格
print("ANOVA Table:")
print(anova_table)

# 提取 F 值和 p 值
F = anova_table['F']['group']
p = anova_table['PR(>F)']['group']

# 判断是否存在显著差异
if p < 0.05:
    print("存在显著差异，进行Tukey检验")
    # 第二步：Tukey检验
    tukey_result = stats.tukey_hsd(A, B, C)
    print(tukey_result)

  
    print("Statistics:", tukey_result.statistic)
    print("P-values:", tukey_result.pvalue)
    print("Confidence Intervals (Low):", tukey_result.confidence_interval)


else:
    print("不存在显著差异")

# 绘制柱状图
means = [np.mean(x) for x in data]
errors = [np.std(x) / np.sqrt(len(x)) for x in data]
x = np.arange(len(groups))

plt.bar(x, means, yerr=errors, capsize=5, label='Mean ± SEM')
plt.xticks(x, groups)

# 标记显著性差异
if p < 0.05:
    # 提取组间比较的结果
    group_indices = list(range(len(groups)))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            # 提取Tukey检验的p值
            pval = tukey_result.pvalue[i, j]
            if pval < 0.05:
                # 计算标记位置
                x1, x2 = i, j
                y = max(means[x1], means[x2]) + max(errors[x1], errors[x2]) + 1
                plt.plot([x1, x1, x2, x2], [y - 0.2, y, y, y - 0.2], lw=1.5, color='black')
                plt.text((x1 + x2) * 0.5, y, '*' if pval > 0.001 else '**', ha='center', va='bottom', color='black')

plt.ylabel('Mean Value')
plt.title('Comparison of Means among Groups')
plt.legend()
plt.show()

print(dir(tukey_result))
