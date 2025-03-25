# TimeSeries_Workforce_Optimization
Worked with group of 5 using hybrid time-series forecasting models to optimize part-time staffing in retail.
Please scroll down to see the Python codes.
### Final Delivery:
https://www.canva.com/design/DAGiv6WoJvo/NMFB0ncDZHvOPiuUsBlAOw/edit?utm_content=DAGiv6WoJvo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton 

### Product Overview:
### Target Buyer: 
Regional Store Managers like Mark, who oversee operations across multiple retail locations and are responsible for staffing, cost control, and store performance.
### Business Problem:
Mark is stuck juggling outdated workforce tools that can't forecast demand, leading to overstaffing during slow hours and understaffing during peak times. The result? Wasted labor costs, burnout among frontline staff, poor customer experiences, and missed revenue targets—plus traditional solutions are either too expensive or too complex to implement.
### Value Proposition:
The platform delivers intelligent labor forecasting and scheduling that integrates with existing HR systems, reduces turnover, and boosts revenue by aligning staffing with real-time and forecasted demand. It’s plug-and-play for ops teams and directly improves both the customer experience and employee retention—no technical degree required.
### ML Solution:
The solution uses a hybrid forecasting model (SVD-SARIMAX + ETS) to predict weekly sales trends with high accuracy, enabling smarter, data-driven staffing decisions. By integrating historical sales and economic indicators, it balances long-term patterns with real-time adaptability—automating the scheduling process across stores with minimal technical overhead.
### Result & Output of the Solution:
The model delivers a 12% cut in labor costs by reducing unnecessary part-time hires and a 15% boost in customer service efficiency by solving peak-hour understaffing. It scales effortlessly across 100+ retail locations, proving it's not just a pilot—it’s plug-and-play at enterprise scale.
### Business Model:
This is a SaaS-based subscription model with tiered pricing—$300/month for small stores, scaling up to $500/month for enterprise clients—designed to flex with business size and needs. Revenue is diversified through API licensing and consulting services for large-scale retailers, but the real growth engine is the subscription tier, built to scale.
### Investment Cycle:
With projected ROI hitting 70.53% by Year 3, the investment thesis is strong—subscription accounts are scaling fast: 40% YoY growth from enterprise, 20% from small stores, and 17% from midsize. Consulting and customization are strategic upsell levers, growing at 16% annually but kept secondary to maintain product-first scalability.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas_datareader import data as pdr #read data from yahoo finance api
import matplotlib.pyplot as plt #viz #GUI manager
import seaborn as sns #viz #plotly is another package
import datetime
from pandas import Grouper #groupby
#statistical data exploration, conducting statistical tests, and estimation of different statistical models
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf #autocorrelation plot
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # double and triple exponential smoothing
from pandas.plotting import autocorrelation_plot #autocorrelation plot
from statsmodels.graphics.gofplots import qqplot #residual diagnostics
from sklearn.metrics import mean_squared_error #accuracy metrics
from math import sqrt
from sklearn.metrics import mean_absolute_error #accuracy metrics

from random import gauss #create gaussian white noise
from random import seed
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.api import VAR
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

#import shap #explainable AI - XAI
from sklearn.ensemble import RandomForestRegressor

import itertools
```
