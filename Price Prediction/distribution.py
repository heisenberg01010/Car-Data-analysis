from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

lm = LinearRegression()

df = pd.read_csv (r'/home/heisenberg/Downloads/CarPrice_Assignment.csv')

X = df[['enginesize']]
Y = df['price']
lm.fit(X, Y)
print(lm.intercept_)
print(lm.coef_)

Yhat = lm.predict(X)
print(Yhat[0:5])

ax1 = sns.distplot(df['price'], hist = False, color = "r", label = "Actual value")
sns.distplot(Yhat, hist = False, color = "b", label = "Fitted Value", ax = ax1)
plt.title("Fitted Value v/s Actual Value")
plt.show()
plt.close()
