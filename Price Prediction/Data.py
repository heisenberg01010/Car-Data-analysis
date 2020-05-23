import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv (r'/home/heisenberg/Downloads/CarPrice_Assignment.csv')
print(df.columns)
print(df['price'])

sns.regplot(x = "highwaympg", y="price", data=df)
plt.ylim(0,)
plt.show()
print(df[['symboling', 'CarName', 'fueltype', 'aspiration',
       'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
       'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
       'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']].corr())
