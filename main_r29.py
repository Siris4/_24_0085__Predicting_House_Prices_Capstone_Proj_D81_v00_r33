import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Create a histogram and KDE for the original price data using Seaborn
plt.figure(figsize=(10, 6))
sns.displot(data['PRICE'], kde=True, height=6, aspect=1.5, bins=30)
plt.title('Histogram and KDE of Original PRICE Data')
plt.xlabel('PRICE')
plt.ylabel('Density')

# Save the plot as a file
plt.savefig(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\price_histogram_kde.png')
