import pandas as pd

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("train.csv")

print("===== Original Dataset Info =====")
print("Shape:", df.shape)

print("\n===== Missing Values Before Cleaning =====")
print(df.isnull().sum())

# ==========================
# Handle Missing Values
# ==========================

# Fill numeric columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# ==========================
# Remove Duplicate Rows
# ==========================

duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows Before:", duplicates_before)

df.drop_duplicates(inplace=True)

duplicates_after = df.duplicated().sum()
print("Duplicate Rows After:", duplicates_after)

# ==========================
# Convert Data Types
# ==========================

# Convert date columns from DD/MM/YYYY format
date_columns = ['Order Date', 'Ship Date']

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(
            df[col],
            format='%d/%m/%Y',
            errors='coerce'
        )

# Fill any invalid dates if found
for col in date_columns:
    if col in df.columns:
        df[col] = df[col].fillna(method='ffill')

# Convert Postal Code to integer
if 'Postal Code' in df.columns:
    df['Postal Code'] = df['Postal Code'].fillna(
        df['Postal Code'].median()
    )
    df['Postal Code'] = df['Postal Code'].astype('Int64')

# ==========================
# Final Verification
# ==========================

print("\n===== Data Types After Conversion =====")
print(df.dtypes)

print("\n===== Missing Values After Cleaning =====")
print(df.isnull().sum())

print("\nFinal Shape:", df.shape)

# ==========================
# Export Cleaned Dataset
# ==========================

df.to_csv("cleaned_data.csv", index=False)

print("\n✅ Data Cleaning Completed Successfully!")
print("✅ Cleaned dataset saved as: cleaned_data.csv")