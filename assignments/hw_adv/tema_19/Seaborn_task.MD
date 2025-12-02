# 📊 Seaborn Data Visualization – Task

---

## 📚 Introduction

Learn how to use Seaborn for advanced data visualization. You will work with built-in datasets and with synthetic data you generate yourself.

---

## 🎯 What datasets does Seaborn include?

- **tips**: restaurant tips (total_bill, tip, sex, smoker, day, time, size)
- **iris**: iris flower measurements (sepal_length, sepal_width, petal_length, petal_width, species)
- **penguins**: penguin measurements (species, island, bill_length_mm, flipper_length_mm, etc.)
- **titanic**: Titanic passengers (survived, pclass, sex, age, fare, etc.)
- **diamonds**: diamond characteristics (carat, cut, color, clarity, price, etc.)

### 📌 Load a dataset
```python
df = sns.load_dataset("tips")  # replace "tips" with any available name
```

### 📌 See available datasets
```python
print(sns.get_dataset_names())
```

---

## 🚀 Starter Code

Run this first:
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example load
df = sns.load_dataset("tips")
print("🔍 First 5 rows of 'tips':")
print(df.head())
print("\n📋 Column info:")
print(df.info())
```

---

# PART 1: Exploring Seaborn Datasets

---

## 📝 EXERCISE 1: Histogram with a new dataset

**Goal:** Plot the age distribution of Titanic passengers.

### 📖 New concept

**`hue`** colors the plot by a categorical column (e.g., split by `"survived"`).

**TODO**
1. Load the `"titanic"` dataset with `sns.load_dataset()`.
2. Use `sns.histplot()` to show the `age` distribution.
3. Add a title and axis labels.
4. **BONUS:** Split by survival status with `hue="survived"`.

### 💻 Starter code
```python
# Load dataset
df_titanic = sns.load_dataset("titanic")

# Build histogram
sns.histplot(data=df_titanic, x="age")
plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Passenger Count")
plt.show()

# BONUS: add hue="survived" to separate survivors vs. non-survivors
```

**💡 Bonus hint:** Add `hue="survived"` in `sns.histplot()` and compare the shapes.

---

## 📝 EXERCISE 2: Pairplot for Iris

**Goal:** Explore relationships between iris flower measurements.

### 📖 New concept

**`pairplot`** builds a grid of plots showing relationships between every pair of numeric variables.

**TODO**
1. Load the `"iris"` dataset.
2. Create a pairplot with `sns.pairplot()`.
3. Color points by `species` using `hue`.
4. **Observe:** What interesting relationships do you see?

### 💻 Starter code
```python
# Load dataset
df_iris = sns.load_dataset("iris")
```

**❓ Reflection:** Which variable pairs look most strongly correlated?

_Your answer:_ _______________________________________________

---

## 📝 EXERCISE 3: Correlation Heatmap for Penguins

**Goal:** Visualize correlations between penguin measurements.

### 📖 New concepts

**`heatmap`** shows values in a matrix with color intensity (perfect for correlations)  
**`annot=True`** shows the numeric values in each cell  
**`cmap`** sets the color palette (e.g., `"coolwarm"` for blue–red)

**TODO**
1. Load the `"penguins"` dataset.
2. Compute the correlation matrix: `df.corr(numeric_only=True)`.
3. Plot a heatmap with `sns.heatmap()`.
4. Add `annot=True` to see the numbers.
5. **BONUS:** Use `cmap="coolwarm"` for clearer contrast.

### 💻 Starter code
```python
# Load dataset
df_penguins = sns.load_dataset("penguins")

# Compute correlations
correlation_matrix = df_penguins.corr(numeric_only=True)
```

---

# PART 2: Working with Generated Data

---

## 🎲 Generate a Synthetic Dataset

Run this to create employee data:
```python
# Build a synthetic dataset with pandas
np.random.seed(42)
df_employees = pd.DataFrame({
    "sex": np.random.choice(["Male", "Female"], size=100),
    "age": np.random.normal(35, 10, size=100),
    "salary": np.random.normal(5000, 1500, size=100),
    "department": np.random.choice(["IT", "HR", "Marketing", "Sales"], size=100),
    "experience": np.random.randint(1, 15, 100),
})

print("🔍 First 5 rows of the generated dataset:")
print(df_employees.head())
print("\n📊 Descriptive statistics:")
print(df_employees.describe())
```

---

## 📝 EXERCISE 4: Violinplot for salary vs. experience

**Goal:** Analyze salary distributions by experience and sex.

### 📖 New concepts

**`violinplot`** blends a boxplot with a density curve (shows distribution shape)  
**`split=True`** splits each violin by `hue` to compare groups

**TODO**
1. Create experience categories (Junior: 1–5 years, Mid: 6–10 years, Senior: 11–15 years).
2. Build a violinplot with `sns.violinplot()`.
3. X-axis: experience categories.
4. Y-axis: `salary`.
5. Use `hue="sex"` and `split=True` to compare by gender.

### 💻 Starter code
```python
# Build experience categories
df_employees["exp_category"] = pd.cut(
    df_employees["experience"],
    bins=[0, 5, 10, 15],
    labels=["Junior (1-5)", "Mid (6-10)", "Senior (11-15)"]
)

# Create violinplot

# Add hue="sex" and split=True to compare Male vs Female
```

---

## 📝 EXERCISE 5: FacetGrid by department and sex

**Goal:** Compare salary distributions across departments and genders.

### 📖 New concepts

**`FacetGrid`** builds multiple subplots based on categorical values  
**`col`** creates a column for each category value  
**`.map()`** applies a plotting function to every facet

**TODO**
1. Create a `FacetGrid` with `col="department"`, `hue="sex"`.
2. Map a `histplot` for `salary`.
3. Add a legend with `.add_legend()`.
4. **Observe:** Do salaries differ by department?

### 💻 Starter code
```python
# Create FacetGrid
g = sns.FacetGrid(df_employees, col="department", height=4)
g.map(sns.histplot, "salary")
plt.show()

# TODO: add hue="sex" to the FacetGrid and .add_legend() at the end
```

**❓ Reflection:** Which department seems to pay the most?

_Your answer:_ _______________________________________________

---

## 📝 EXERCISE 6: Barplot to compare salaries

**Goal:** Compare average salary between younger and older employees.

### 📖 New concepts

**`barplot`** shows the mean (or other statistic) for each category  
**`apply()`** applies a function to each element of a column

**TODO**
1. Create a new `"age_group"` column:
   - `"Under 30"` if `age < 30`
   - `"30 and over"` if `age >= 30`
2. Build a barplot with `sns.barplot()`.
3. X-axis: `age_group`.
3. Y-axis: `salary` (mean is automatic).
5. **BONUS:** Add `hue="sex"` to compare genders.

### 💻 Starter code
```python
# Build age groups
df_employees["age_group"] = df_employees["age"].apply(
    lambda x: "Under 30" if x < 30 else "30 and over"
)

# Create barplot
sns.barplot(data=df_employees, x="age_group", y="salary")
plt.title("Average Salary by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Salary")
plt.show()

# TODO BONUS: add hue="sex" to compare Male vs Female
```

---

# 🎓 Reflection Questions

After completing the exercises, answer:

### 1. What differences did you notice between histplot and violinplot?

_Your answer:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

### 2. What relationships are easier to see with a pairplot?

_Your answer:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

### 3. Why use a FacetGrid instead of a single plot?

_Your answer:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

### 4. What did you learn from the synthetic employee dataset?

_Your answer:_  
_______________________________________________________________  
_______________________________________________________________  
_______________________________________________________________

---

# 💡 Useful Resources

- 📚 **Seaborn documentation:** https://seaborn.pydata.org/
- 📊 **Example gallery:** https://seaborn.pydata.org/examples/index.html
- 🎨 **Color palettes:** https://seaborn.pydata.org/tutorial/color_palettes.html
- 📖 **Official tutorial:** https://seaborn.pydata.org/tutorial.html

---

# 🔥 Extra Challenge (optional)

If you finish early and want more practice:

1. **Combine visuals:** Create subplots that mix multiple plot types.
2. **Explore styles:** Try different themes with `sns.set_style()` (darkgrid, whitegrid, dark, white, ticks).
3. **Save your plots:** Use `plt.savefig("file_name.png", dpi=300, bbox_inches="tight")`.
4. **Analyze complex relationships:** Try `sns.jointplot()` to combine scatter plots with marginal distributions.
5. **Customize colors:** Explore custom palettes with `sns.color_palette()`.

---

## ✅ Final Checklist

Before submitting, verify:

- [ ] All 6 exercises are complete
- [ ] Titles and axis labels are set on every plot
- [ ] Reflection questions are answered
- [ ] Code runs without errors
- [ ] You understand what each visualization shows
- [ ] Bonus steps are tested (if time allowed)

---

**🎉 Good luck with the task! 🎉**

> **Reminder:** If you get stuck, revisit the Seaborn docs and course examples. Experiment with different parameters!
