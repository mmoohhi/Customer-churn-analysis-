import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Load Data ──────────────────────────────────────
df = pd.read_csv(r"C:\Telco Customer Churn project\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nChurn Distribution:\n", df["Churn"].value_counts())

# ── Clean Data ─────────────────────────────────────
# TotalCharges has spaces — convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)

print("\nAfter cleaning — Shape:", df.shape)

# ── Analysis 1: Churn Rate ─────────────────────────
churn_counts = df["Churn"].value_counts()
colors = ["#4CAF50", "#E87040"]

plt.figure(figsize=(6, 4))
churn_counts.plot(kind="bar", color=colors)
plt.title("Churn vs Non-Churn Customers")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Telco Customer Churn project\churn_rate.png")
plt.show()

# ── Analysis 2: Churn by Contract Type ────────────
contract_churn = df.groupby(["Contract", "Churn"]).size().unstack()

plt.figure(figsize=(7, 4))
contract_churn.plot(kind="bar", color=["#4CAF50", "#E87040"], figsize=(7, 4))
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(["No Churn", "Churn"])
plt.tight_layout()
plt.savefig(r"C:\Telco Customer Churn project\churn_by_contract.png")
plt.show()

# ── Analysis 3: Monthly Charges vs Churn ──────────
plt.figure(figsize=(7, 4))
df.boxplot(column="MonthlyCharges", by="Churn", 
           color=dict(boxes="#4A90D9", whiskers="#4A90D9", medians="#E87040", caps="#4A90D9"))
plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(r"C:\Telco Customer Churn project\charges_vs_churn.png")
plt.show()

# ── Analysis 4: Tenure vs Churn ───────────────────
plt.figure(figsize=(7, 4))
df[df["Churn"] == "Yes"]["tenure"].hist(bins=20, alpha=0.7, color="#E87040", label="Churn")
df[df["Churn"] == "No"]["tenure"].hist(bins=20, alpha=0.7, color="#4CAF50", label="No Churn")
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Telco Customer Churn project\tenure_vs_churn.png")
plt.show()

# ── ML Model: Logistic Regression ─────────────────
print("\n--- Building ML Model ---")

# Encode all categorical columns
le = LabelEncoder()
df_model = df.copy()
for col in df_model.select_dtypes(include="object").columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix (Accuracy: {accuracy*100:.1f}%)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(r"C:\Telco Customer Churn project\confusion_matrix.png")
plt.show()

# ── Export Cleaned Data ────────────────────────────
df.to_csv(r"C:\Telco Customer Churn project\churn_cleaned.csv", index=False)

print("\n✓ Cleaned file saved: churn_cleaned.csv")
print("✓ All charts saved!")
print(f"\nKey Insights:")
print(f"  - Total Customers: {len(df)}")
print(f"  - Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")
print(f"  - Model Accuracy: {accuracy*100:.2f}%")