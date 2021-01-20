# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Empirical Analysis of Cuckoo Hashing Insert Complexity
# This notebook attempts to analyze the number of disk reads required for different formats of a hash table that uses the Cuckoo format for collison handling. 
#
# Disk reads are calculated theorectically by assuming that each time a (key, value) pair is moved, that this new section of our data structure will need to be read from disk storage.

# %%
import math
import numpy as np
import pandas as pd
from itertools import product
from functools import partial
from scipy.stats import poisson, geom, chisquare, chi2

# %%
import dask.dataframe as dd
from dask.distributed import Client, progress
client = Client(n_workers=2, threads_per_worker=2, memory_limit='4GB')

# %%
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.resources import INLINE

# %%
output_notebook(INLINE)

# %%
pd.set_option('display.max_rows', 500)

# %% [markdown]
# ## Schema Information
# Here we are getting the schema and setting up our data for the analysis

# %%
filenames = [f"/mnt/mybook/MonteCarloCuckoo{i}.csv" for i in range(2, 5)]
data_types = {
    "Unnamed: 0": "int64", 
    "Util": "float64", 
    "NumReads": "int64", 
    "NumTables": "float64", 
    "LenTables": "int64",
    "NumElem": "int64", 
    "Hash1": "object",
    "Hash2": "object",
    "Hash3": "object",
    "Hash4": "object"
}

# %%
df2 = dd.read_csv(filenames[0], dtype=data_types)

# %%
df3 = dd.read_csv(filenames[1], dtype=data_types)

# %%
df4 = dd.read_csv(filenames[2], dtype=data_types)

# %% [markdown]
# # EDA 
# We want to understand some basic information about our dataset. This will be a breif analysis of number of rows and distribution

# %% [markdown]
# ## Descriptive Statistics

# %%
# number of data points for 2 table variations
df2.shape[0].compute()

# %%
# number of data points for 3 table variations
df3.shape[0].compute()

# %%
# number of data points for 4 table variations
df4.shape[0].compute()

# %%
# max load factor for 2 table variations
df2["Util"].max().compute()

# %%
# max load factor for 3 table variations
df3["Util"].max().compute()

# %%
# max load factor for 4 table variations
df4["Util"].max().compute()

# %% [markdown]
# ## Basic Plots
# Here we look at a couple of lots that show a 1% sample of our data in order to understand the results better. This is used to inform the direct of a regression or other metric for empirically defining the number of disk reads a Hash Table would require if it was not stored in main memory.
#
# We see the number of disk reads required appears to grow exponentially with the load factor (also called the utilization ratio here). Additionally, the growth rate of this hypothetical function is different between the number of tables. This follows from the theoretical likelihood for the number of operations required for insertion given the load factor and number of tables.

# %%
merged_df = dd.concat([df2.sample(frac=0.05), df3.sample(frac=0.05), df4.sample(frac=0.05)]).compute()
merged_df.head(10)

# %%
merged_df[["Util", "NumReads"]].plot(x="Util", y="NumReads", kind="scatter")

# %%
merged_df[merged_df["NumTables"] == 2][["Util", "NumReads"]].plot(x="Util", y="NumReads", kind="scatter")

# %%
merged_df[merged_df["NumTables"] == 3][["Util", "NumReads"]].plot(x="Util", y="NumReads", kind="scatter")

# %%
merged_df[merged_df["NumTables"] == 4][["Util", "NumReads"]].plot(x="Util", y="NumReads", kind="scatter")

# %% [markdown]
# ## Distribution Analysis
# If we think about how we might express a function for the information found in this data, we should look at the log transform of the data in order to possibly have a linear regression; however, you will immediately notice that the data does not follow a linear pattern and that even a fitted regression to this data expresed by some unknown function would attempt to reduce the error (most commonly by RSS). Such a function would still poorly describe our data except by the fact that we have a concept of the expected case. Rather since we know our data structure has a worst-case ammoratized constant time search complexity, we we consider ourselves with a probablistic worst-case complexity by finding the 90%, 95%, and 99% thresholds for insertion complexity. To find the probablistic complexity we will look for a distribution that matches our underlying data.

# %%
plot = figure(plot_width=400, plot_height=400)
plot.scatter(
    x=merged_df[merged_df["NumTables"] == 2]["Util"], 
    y=np.log(merged_df[merged_df["NumTables"] == 2]["NumReads"])
)
show(plot)

# %% [markdown]
# If we consider the underlying events that the data is representing, we see that the number of disk reads that an insertion requires direct corresponds to the number of times that a (key, value) pair is inserted into a filled table slot. For the ideal hash function the maps to a uniform probability distribution across its range, our event is that the table slot fails to be empty which has a probability of $p(x) = 1 - \frac{\textrm{load factor}}{\textrm{number of tables} \ \cdot \ \textrm{length of tables}}$. This event is repeated every time that we have a theorectial disk read. Since we are repeating independent events with a technically infinite domain, the discrete probability distribution is either poisson (average number of events occuring for a load factor) or geometric (rate at which our event occurs for a load factor). 
#
# This is further supported by a histogram of number of disk reads for a load factor of 50% on two tables. We also consider these two distributions for the case with two tables.

# %%
merged_df[
    (merged_df["NumTables"] == 2) & 
    (merged_df["Util"] >= 0.5) & 
    (merged_df["Util"] < 0.51) &
    (merged_df["NumReads"] < 50)
  ]["NumReads"].hist(density=True, bins=100)

# %% [markdown]
# ### Remove outliers
# If we bin the data we find that most of our points fall into the very low range. We want to generate the average of our data so that we can find the theoretical distribution without having the outliers skew the data too much. We consider the poisson distribution first which is defined by $p(x, \lambda) = \frac{e^{-\lambda}\lambda^x}{x!}$, where $\lambda = \mu = \sigma^2$. Thus we will plot the distribution using the $\lambda$ derived from the mean and standard deviation considering data points less than the outlier cutoffs. The geometric distribution is defined by $p(x, r) = (1 - r)^x(r)$, where $r = \frac{1}{1+\mu}$.
#
# The outlier thresholds that we consider are 2, 3, and 5. 

# %%
bins = pd.qcut(
    merged_df[
        (merged_df["NumTables"] == 2) &
        (merged_df["Util"] >= 0.5) & 
        (merged_df["Util"] < 0.51) 
    ]["NumReads"], 
    1000, 
    duplicates="drop", 
    retbins=True
)
(pd.DataFrame(bins[0]).reset_index().groupby("NumReads").count() /
     merged_df[merged_df["NumTables"] == 2].shape[0]).transpose()

# %%
outlier_thresholds = [2, 3, 5]


# %%
def get_lambda_from_mean(outlier_threshold):
    # lambda = mean
    return merged_df[(merged_df["NumTables"] == 2) & 
       (merged_df["Util"] >= 0.5) & (merged_df["Util"] < 0.51) &
       (merged_df["NumReads"] < outlier_threshold)
      ]["NumReads"].mean()

def get_lambda_from_std(outlier_threshold):
    # lambda = std^2
    return merged_df[(merged_df["NumTables"] == 2) & 
       (merged_df["Util"] >= 0.5) & (merged_df["Util"] < 0.51) &
       (merged_df["NumReads"] < outlier_threshold)
      ]["NumReads"].std() ** 2

def get_r_from_mean(outlier_threshold):
    # r = 1/(mean + 1)
    return 1/(1 + merged_df[(merged_df["NumTables"] == 2) & 
       (merged_df["Util"] >= 0.5) & (merged_df["Util"] < 0.51) &
       (merged_df["NumReads"] < outlier_threshold)
      ]["NumReads"].mean())

cand_lambda_mean = [x for x in map(get_lambda_from_mean, outlier_thresholds)]
cand_lambda_std = [x for x in map(get_lambda_from_std, outlier_thresholds)]
cand_r = [x for x in map(get_r_from_mean, outlier_thresholds)]
print(cand_lambda_mean)
print(cand_lambda_std)
print(cand_r)

# %%
plots = []
x = np.arange(1, 20)

# plot the histogram of our actual data
plot = figure(
    plot_width=300, 
    plot_height=300,
    title="Base Distribution"
)
hist, edges = np.histogram(
    merged_df[
        (merged_df["NumTables"] == 2) & 
        (merged_df["Util"] >= 0.5) & 
        (merged_df["Util"] < 0.51) &
        (merged_df["NumReads"] < 50)
      ]["NumReads"],
    density=True, 
    bins=100
)
plot.quad(
    top=hist, 
    bottom=0, 
    left=edges[:-1], 
    right=edges[1:]
)
plots.append(plot)

# plot the poisson distributions with lambda generated from the mean
for i in range(len(cand_lambda_mean)):
    l = cand_lambda_mean[i]
    plot = figure(
        plot_width=300, 
        plot_height=300,
        title=f"Mean Lambda={l} (threshold={outlier_thresholds[i]})"
    )
    y = poisson.pmf(x, l)
    plot.vbar(x=x, top=y)
    plots.append(plot)
    
    
# plot the poisson distributions with lambda generated from the standard deviation
for i in range(len(cand_lambda_std)):
    l = cand_lambda_std[i]
    plot = figure(
        plot_width=300, 
        plot_height=300,
        title=f"STD Lambda={l} (threshold={outlier_thresholds[i]})"
    )
    y = poisson.pmf(x, l)
    plot.vbar(x=x, top=y)
    plots.append(plot)

# plot the geometric distributions with r generated from the mean
for i in range(len(cand_lambda_std)):
    r = cand_r[i]
    plot = figure(
        plot_width=300, 
        plot_height=300,
        title=f"Mean r={r} (threshold={outlier_thresholds[i]})"
    )
    y = geom.pmf(x, r)
    plot.vbar(x=x, top=y)
    plots.append(plot)
    
# show all of the plots together
view = gridplot(plots, ncols = 3)
show(view)


# %% [markdown]
# ## Goodness of Fit tests for Poisson and Geometric distributions
# Here we want to identify (across more items than we can look at graphically) whether or not our underlying distribution is poisson or geometric. This will involve a couple of statistical tests. If we are unable to statistically support our assumption of a geometric distribution, we will have to assume this is the theorectical distribution based solely on our reasoning of the problem. 
#
# Useful Articles:
# - https://search.ebscohost.com/login.aspx?direct=true&db=edselc&AN=edselc.2-52.0-84884931915&site=eds-live&scope=site
# - https://www.jstor.org/stable/1403470
# - https://doi.org/10.1081/SAC-120023878
# - https://doi.org/10.1080/00949655.2011.563740
#
# A modified version of the Smooth Goodness of Fit Test $S_1$ (Best and Rayner) had the highest power for testing if the distribution is Geometric or Poisson (ÖZonur et al.) - first article above. This is defined by:
#
# $C(n, k) = k - \textrm{combinations of} \ n \ \textrm{elements} = \frac{n!}{(n-k)!k!}$  
#
# $a = \frac{q}{1-q}$
#
# $K = \frac{1}{r!(a^2 + a)^{r/2})}$
#
# $h_r(X_j, \hat{q}) = K\Sigma_{i=0}^{r} C(r-i, x) * (C(i, r))^2 * i! (r-i)! (-a)i$
#
# $h_r$ is a special case of Meixner polynomials defined by a recurrance relation in Rayer and Best (1989). We need not concern ourseleves too much with generation of these polynomials since the modified version uses only $S_1 = U^2_2$ meaning we only need $h_2(X_j, \hat{q})$
#
# $h_2(X_j, \hat{q}) = x(x-1)-4ax+2a^2$
#
# $U_r = \Sigma_{j=1}^n h_r(X_j, \hat{q}) / \sqrt{n}$
#
# $S_c = U^2_2 + \cdots + U^2_{c+1}$
#
# $S^*_1 = nS_1 / \Sigma_{j=1}^{n} h^2_2(X_j, \hat{q})$
#
# If $S^*_1 > \chi_1^2$ then $H_0$ is rejected.
#
# Thus, all of these together give us the large formula below
#
# \begin{equation} S_1^* = \frac{n \Sigma_{j=1}^{n} h_r(X_j, \hat{q}) / \sqrt{n}}{\Sigma_{j=1}^{n} h_2^2(X_j, \hat{q})} \end{equation}

# %%
def h_2(x, q):
    a = float(q / (1 - q))
    return x * (x - 1) - (4 * a * x) + np.power(2, a)

def U_2(x, n, q):
    return sum(map(lambda x: h_2(x, q), x)) / math.sqrt(n)

def S_1_star(x):
    n = float(x.shape[0])
    mu = x.mean()
    p = 1 / (1 + mu)
    q = 1 - p
    return n * (np.power(U_2(x, n, q), 2)) / sum(map(lambda x: np.power(h_2(x, q), 2), x))


# %% [markdown]
# We want to now calculate this test for geometric goodness of fit for each percentile of load factor in our data. This will be calculating the chi-squared test and the defined $S_1^*$ test above for each load factor percentile and number of tables (100 * 3 = 300 times). We will see what passes our assumption of geometric, and what does not.

# %%
test_df = dd.concat([df2, df3, df4])

# %%
test_df["LF"] = (test_df["Util"].round(2) * 100).astype("int64")

# %%
metrics = test_df[["NumTables", "LF"]].drop_duplicates().compute().set_index(["NumTables", "LF"])

# %%
metrics["S_STAR"] = (test_df[["NumReads", "NumTables", "LF"]]
                     .groupby(["NumTables", "LF"])
                     .apply(lambda x: float(S_1_star(x["NumReads"])), meta=("NumReads", "float64")))

# %%
metrics["CHI"] = (test_df[["NumReads", "NumTables", "LF"]]
                  .groupby(["NumTables", "LF"])
                  .apply(lambda x: chisquare(x["NumReads"])[0], meta=("NumReads", "float64")))

# %%
metrics["GEO"] = metrics["S_STAR"] - metrics["CHI"]

# %%
metrics.shape

# %%
# find failures
print([i for i in metrics[metrics["GEO"] > 0].index])

# %%
metrics[metrics["GEO"] > 0].shape

# %% [markdown]
# We will also test if the data passes a test to be poisson distributed. This will be the index of dispersion (or Variance to Mean Ratio) test (https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/ind_disp.htm)
#
# This found the that the index of dispersion shows a ratio of variance to mean, and if data is Poisson then this ration should be 1. If the data is geometric, then the dispersion is >1.
#
# \begin{equation} I = \Sigma_{i=1}^{N} \frac{(X_i - \bar{X})^2}{\bar{X}} = \frac{\sigma^2}{\bar{x}} \end{equation}

# %%
var = pd.DataFrame(
        test_df[["NumReads", "NumTables", "LF"]]
        .groupby(["NumTables", "LF"])["NumReads"]
        .std()
        .rpow(2)
        .compute()).rename(columns={"NumReads":"VAR"})
mu = pd.DataFrame(
        test_df[["NumReads", "NumTables", "LF"]]
        .groupby(["NumTables", "LF"])["NumReads"]
        .mean()
        .compute()).rename(columns={"NumReads":"MEAN"})
temp = var.merge(mu, on=["NumTables", "LF"])
out = temp["VAR"] / temp["MEAN"]

# %%
out[out <= 1]

# %%
out[out < 2].plot(figsize=(12,6))

# %% [markdown]
# # Bootstrapping an Accurate Representation of Our Data
# Here we randomly sample our data multiple times in order to find a bootstrapped average to use in our theorectical distribution.
#
# We want to find the mean for number of reads at each of the percentiles this will be done for each combination of hashes. The best hash combination will be the one with the lowest average mean num reads.

# %%
num_samples = 100

# %%
df2["LF"] = (df2["Util"].round(2) * 100).astype("int64")

metrics_list = []
for i in range(num_samples):
    m = df2.sample(frac=0.1, replace=True)[["Hash1", "Hash2", "LF", "NumReads"]].groupby(["Hash1", "Hash2", "LF"]).mean()
    metrics_list.append(m.compute())

temp = pd.concat(metrics_list)
sd2 = temp.groupby(["Hash1", "Hash2", "LF"]).std()
mu2 = temp.groupby(["Hash1", "Hash2", "LF"]).mean()

# %%
df3["LF"] = (df3["Util"].round(2) * 100).astype("int64")

metrics_list = []
for i in range(num_samples):
    m = (df3.sample(frac=0.1, replace=True)[["Hash1", "Hash2", "Hash3", "LF", "NumReads"]]
            .groupby(["Hash1", "Hash2", "Hash3", "LF"])
            .mean())
    metrics_list.append(m.compute())

temp = pd.concat(metrics_list)
sd3 = temp.groupby(["Hash1", "Hash2", "Hash3", "LF"]).std()
mu3 = temp.groupby(["Hash1", "Hash2", "Hash3",  "LF"]).mean()

# %%
df4["LF"] = (df4["Util"].round(2) * 100).astype("int64")

metrics_list = []
for i in range(num_samples):
    m = (df4.sample(frac=0.1, replace=True)[["Hash1", "Hash2", "Hash3", "Hash4", "LF", "NumReads"]]
            .groupby(["Hash1", "Hash2", "Hash3", "Hash4", "LF"])
            .mean())
    metrics_list.append(m.compute())

temp = pd.concat(metrics_list)
sd4 = temp.groupby(["Hash1", "Hash2", "Hash3", "Hash4",  "LF"]).std()
mu4 = temp.groupby(["Hash1", "Hash2", "Hash3", "Hash4", "LF"]).mean()

# %%
# save our bootstrapped means and variances
mu2.to_csv("/mnt/mybook/mu2.csv")
sd2.to_csv("/mnt/mybook/sd2.csv")
mu3.to_csv("/mnt/mybook/mu3.csv")
sd3.to_csv("/mnt/mybook/sd3.csv")
mu4.to_csv("/mnt/mybook/mu4.csv")
sd4.to_csv("/mnt/mybook/sd4.csv")

# %% [markdown]
# Let's look at the best combination of the hash functions for our cuckoo hashing based on the lowest mean number of reads. We look at the lower and upper bounds for our confidence intervals for the mean number of reads below the theoretical thresholds of 50% for two tables, 91% for three tables, and 97% for four tables (http://www.eecs.harvard.edu/~michaelm/postscripts/esa2009.pdf, https://hal.inria.fr/hal-01184689/document)

# %%
mu2 = mu2.reset_index()
sd2 = sd2.reset_index()

# %%
threshold = mu2.reset_index()["LF"] <= 50
lb2 = (mu2[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2"]).mean() 
       - 2*sd2[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2"]).mean())
lb2.sort_values("NumReads")

# %%
ub2 = (mu2[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2"]).mean() 
       + 2*sd2[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2"]).mean())
ub2.sort_values("NumReads")

# %%
mu3 = mu3.reset_index()
sd3 = sd3.reset_index()

# %%
threshold = mu3.reset_index()["LF"] <= 91
lb3 = (mu3[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3"]).mean() 
       - 2*sd3[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3"]).mean())
lb3.sort_values("NumReads")

# %%
ub3 = (mu3[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3"]).mean() 
       + 2*sd3[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3"]).mean())
ub3.sort_values("NumReads")

# %%
mu4 = mu4.reset_index()
sd4 = sd4.reset_index()

# %%
threshold = mu3.reset_index()["LF"] <= 97
lb4 = (mu4[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3", "Hash4"]).mean() 
       - 2*sd4[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3", "Hash4"]).mean())
lb4.sort_values("NumReads")

# %%
ub4 = (mu4[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3", "Hash4"]).mean() 
       + 2*sd4[threshold].set_index("LF").dropna().groupby(["Hash1", "Hash2", "Hash3", "Hash4"]).mean())
ub4.sort_values("NumReads")

# %%
plot = figure(y_range=(0, 300), plot_width=600, plot_height=400)
plot.line(
    x=[50,50],
    y=[0,300],
    color="blue"
)
plot.line(
    x=mu2.reset_index()["LF"].unique().tolist(),
    y=mu2.groupby("LF").mean().apply(lambda x: geom.ppf(0.95, 1/(1+x)))["NumReads"].tolist(),
    color="blue",
    legend_label="Two Tables"
)

plot.line(
    x=[91,91],
    y=[0,300],
    color="red"
)
plot.line(
    x=mu3.reset_index()["LF"].unique().tolist(),
    y=mu3.groupby("LF").mean().apply(lambda x: geom.ppf(0.95, 1/(1+x)))["NumReads"].tolist(),
    color="red",
    legend_label="Three Tables"
)

plot.line(
    x=[97,97],
    y=[0,300],
    color="green"
)
plot.line(
    x=mu4.reset_index()["LF"].unique().tolist(),
    y=mu4.groupby("LF").mean().apply(lambda x: geom.ppf(0.95, 1/(1+x)))["NumReads"].tolist(),
    color="green",
    legend_label="Four Tables"
)
show(plot)

# %%
plot = figure(
    y_range=(0, 20), 
    plot_width=1000, 
    plot_height=750, 
    tooltips=[("LOC", "($x, $y)")],
    title="Number of Disk Reads for Insertion to a Cuckoo Hashing Scheme (Average and 90th Percentile)"
)
plot.xaxis.axis_label = "Load Factor"
plot.yaxis.axis_label = "Disk Reads"

plot.line(
    x=[50,50],
    y=[0,20],
    color="blue"
)
plot.line(
    x=mu2.reset_index()["LF"].unique().tolist(),
    y=mu2.groupby("LF").mean().apply(lambda x: geom.ppf(0.90, 1/(1+x)))["NumReads"].tolist(),
    color="blue",
    legend_label="Two Tables"
)
plot.line(
    x=mu2.reset_index()["LF"].unique().tolist(),
    y=mu2.groupby("LF").mean()["NumReads"].tolist(),
    color="blue",
    line_dash="dashed"
)

plot.line(
    x=[91,91],
    y=[0,20],
    color="red"
)
plot.line(
    x=mu3.reset_index()["LF"].unique().tolist(),
    y=mu3.groupby("LF").mean().apply(lambda x: geom.ppf(0.90, 1/(1+x)))["NumReads"].tolist(),
    color="red",
    legend_label="Three Tables"
)
plot.line(
    x=mu3.reset_index()["LF"].unique().tolist(),
    y=mu3.groupby("LF").mean()["NumReads"].tolist(),
    color="red",
    line_dash="dashed"
)

plot.line(
    x=[97,97],
    y=[0,20],
    color="green"
)
plot.line(
    x=mu4.reset_index()["LF"].unique().tolist(),
    y=mu4.groupby("LF").mean().apply(lambda x: geom.ppf(0.90, 1/(1+x)))["NumReads"].tolist(),
    color="green",
    legend_label="Four Tables"
)
plot.line(
    x=mu4.reset_index()["LF"].unique().tolist(),
    y=mu4.groupby("LF").mean()["NumReads"].tolist(),
    color="green",
    line_dash="dashed"
)

show(plot)

# %%
plot = figure(y_range=(0, 20), plot_width=600, plot_height=400)
plot.line(
    x=mu2.reset_index()["LF"].unique().tolist(),
    y=df2[["NumReads", "LF"]].groupby("LF").mean().compute()["NumReads"].tolist(),
    color="blue",
    legend_label="Two Tables"
)
show(plot)

# %%
