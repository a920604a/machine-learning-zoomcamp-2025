import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error

# ------------------------------
# Load dataset
# ------------------------------
url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
df = pd.read_csv(url)

# ------------------------------
# Filter relevant columns
# ------------------------------
columns = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df = df[columns]

# ------------------------------
# Q1: Missing value column
# ------------------------------
missing_cols = df.columns[df.isnull().any()].tolist()
print("Column with missing values:", missing_cols)

# ------------------------------
# Q2: Median horsepower
# ------------------------------
median_hp = df['horsepower'].median()
print("Median horsepower:", median_hp)

# ------------------------------
# Shuffle and split dataset
# ------------------------------
def split_data(df, seed=42):
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    df_train = df_shuffled[:n_train]
    df_val = df_shuffled[n_train:n_train+n_val]
    df_test = df_shuffled[n_train+n_val:]
    return df_train, df_val, df_test

# ------------------------------
# Q3: Missing value treatment & RMSE
# ------------------------------
def prepare_features(df, fill_value):
    df_filled = df.copy()
    df_filled['horsepower'] = df_filled['horsepower'].fillna(fill_value)
    X = df_filled.drop(columns=['fuel_efficiency_mpg']).values
    y = df_filled['fuel_efficiency_mpg'].values
    return X, y

df_train, df_val, df_test = split_data(df, seed=42)

# Option 1: fill with 0
X_train_0, y_train_0 = prepare_features(df_train, 0)
X_val_0, y_val_0 = prepare_features(df_val, 0)
model_0 = LinearRegression().fit(X_train_0, y_train_0)
y_pred_0 = model_0.predict(X_val_0)
rmse_0 = round(root_mean_squared_error(y_val_0, y_pred_0), 2)

# Option 2: fill with mean (from train only)
hp_mean = df_train['horsepower'].mean()
X_train_mean, y_train_mean = prepare_features(df_train, hp_mean)
X_val_mean, y_val_mean = prepare_features(df_val, hp_mean)
model_mean = LinearRegression().fit(X_train_mean, y_train_mean)
y_pred_mean = model_mean.predict(X_val_mean)
rmse_mean = round(root_mean_squared_error(y_val_mean, y_pred_mean), 2)

print("RMSE with 0:", rmse_0)
print("RMSE with mean:", rmse_mean)

# ------------------------------
# Q4: Regularized linear regression
# ------------------------------
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]
rmse_r = {}
for r in r_values:
    model_r = Ridge(alpha=r)
    X_train, y_train = prepare_features(df_train, 0)
    X_val, y_val = prepare_features(df_val, 0)
    model_r.fit(X_train, y_train)
    y_pred = model_r.predict(X_val)
    rmse_r[r] = round(root_mean_squared_error(y_val, y_pred), 2)

best_r = min(rmse_r, key=rmse_r.get)
print("RMSE by r:", rmse_r)
print("Best r:", best_r)

# ------------------------------
# Q5: Influence of random seed
# ------------------------------
seeds = list(range(10))
rmse_scores = []
for seed in seeds:
    df_train, df_val, df_test = split_data(df, seed=seed)
    X_train, y_train = prepare_features(df_train, 0)
    X_val, y_val = prepare_features(df_val, 0)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse_scores.append(root_mean_squared_error(y_val, y_pred))

std_rmse = round(np.std(rmse_scores), 3)
print("Std of RMSE across seeds:", std_rmse)

# ------------------------------
# Q6: Test RMSE with seed 9 and r=0.001
# ------------------------------
df_train, df_val, df_test = split_data(df, seed=9)
df_train_val = pd.concat([df_train, df_val])
X_train_val, y_train_val = prepare_features(df_train_val, 0)
X_test, y_test = prepare_features(df_test, 0)

model_test = Ridge(alpha=0.001).fit(X_train_val, y_train_val)
y_pred_test = model_test.predict(X_test)
rmse_test = round(root_mean_squared_error(y_test, y_pred_test), 3)
print("Test RMSE:", rmse_test)
