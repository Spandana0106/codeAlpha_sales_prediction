import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------
df = pd.read_csv("Advertising.csv")  # place the downloaded file in the same folder
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(df.shape)
print(df.head())
print(df.describe())

# ---------------------------------------------------------
# STEP 2: Explore relationships
# ---------------------------------------------------------
corr = df.corr(numeric_only=True)
print("\nCorrelation with Sales:\n", corr["Sales"].sort_values(ascending=False))

sns.pairplot(df, x_vars=[c for c in df.columns if c != "Sales"], y_vars="Sales", kind="reg")
plt.savefig("sales_relationships.png")
plt.close()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("sales_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# STEP 3: Prepare features and target
# ---------------------------------------------------------
X = df.drop(columns=["Sales"])
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# STEP 4: Train models (compare linear vs. tree ensemble)
# ---------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 5: Evaluate both
# ---------------------------------------------------------
for name, model in [("Linear Regression", lin_model), ("Random Forest", rf_model)]:
    pred = model.predict(X_test)
    print(f"\n{name}")
    print(" R2:", r2_score(y_test, pred))
    print(" MAE:", mean_absolute_error(y_test, pred))
    print(" RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# ---------------------------------------------------------
# STEP 6: Interpret coefficients (business insight)
# ---------------------------------------------------------
coef_df = pd.Series(lin_model.coef_, index=X.columns).sort_values(ascending=False)
print("\nLinear model coefficients (impact per $1000 spend):\n", coef_df)
print(
    "\nInsight: the channel with the highest coefficient gives the most "
    "sales lift per unit of spend -- prioritize budget there, but check "
    "for diminishing returns at high spend levels before scaling further."
)