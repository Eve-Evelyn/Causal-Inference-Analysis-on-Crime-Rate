import pandas as pd
import matplotlib.pyplot as plt
import linearmodels as lm
import seaborn as sns

data = pd.read_csv("Crime.csv")

# check that there's no null values in data, invalid data, and the numbers of data in region and years match each other
print(data.isna().sum())
print(data.value_counts("region"))
print(data.value_counts("year"))

# scatter plot to visualize how crime rate varies based on year, region, and police per capita
fig, ax = plt.subplots(1, 2)
ax1 = sns.scatterplot(data=data, x="polpc", y="crmrte", hue="region", ax=ax[0])
ax1.set_xlim(0, 0.0175)
ax1.set_ylim(0, 0.14)
ax1.set_xlabel("Police per Capita", fontsize=12)
ax1.set_ylabel("Crime Rate", fontsize=12)
ax1.set_title("Crime Rate VS Police per Capita")
ax2 = sns.scatterplot(data=data, x="year", y="crmrte", hue="region", ax=ax[1])
ax2.set_ylim(0, 0.14)
ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("Crime Rate", fontsize=12)
ax2.set_title("Crime Rate VS Year")

# plt.show()

# causal inference using two way fixed effect with region and year as Entity Effects and Time Effects
data = data.set_index(['region', 'year'])
model = lm.PanelOLS.from_formula('crmrte ~ polpc + EntityEffects + TimeEffects', data=data)
res = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
print(res)
