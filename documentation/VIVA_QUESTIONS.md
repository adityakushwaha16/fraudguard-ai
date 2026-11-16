# Viva preparation
1. Why is accuracy a poor metric on PaySim? (fraud is ~0.1% of rows; predicting "legit" always gives ~99.9% accuracy.)
2. What are precision, recall, F1, ROC-AUC, PR-AUC? Why PR-AUC for imbalanced data?
3. Explain false positives vs false negatives in fraud. How did you pick the threshold?
4. How did you prevent data leakage? (Pipeline fitted on training data only; stratified split; dropped IDs/flag columns.)
5. Why only TRANSFER and CASH_OUT? Why drop nameOrig, nameDest, isFlaggedFraud, newbalanceOrig and newbalanceDest? (post-transaction balances = target leakage)
6. What engineered features did you add and why (amount-to-balance ratio, would-drain flag, zero-balance flags, hour of day)? Why are they safe to compute before a transaction?
7. How is class imbalance handled? Why not SMOTE on millions of rows?
8. Compare Logistic Regression, Random Forest, XGBoost. Which was selected, using which metric, and why?
9. What is SHAP? Difference between global and local explanations? Why TreeExplainer vs LinearExplainer?
10. How does a SHAP value become the sentence shown to the user?
11. How does the website avoid retraining on each start? (Joblib artifacts loaded once.)
12. How are passwords stored? How do you stop user A reading user B's transactions? (404 for non-owned IDs.)
13. What happens if the model file is missing? (503; no fake predictions.)
14. What are the limitations? (simulated data, concept drift, no real bank integration, demographic bias unknown.)
15. How would you deploy and scale this? What is the future scope?
