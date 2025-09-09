Core Formula



1. R² Adjusted Score = avg\_score × linear\_r2



2\. Slope Adjusted Score = r2\_adjusted\_score × (0.5 + sigmoid\_factor)



Where sigmoid\_factor = 1 / (1 + exp(-sensitivity × linear\_slope))



3\. Stability Adjusted Score = slope\_adjusted\_score × stability\_factor



Where stability\_factor = 1 / (1 + score\_std)

