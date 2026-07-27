# train - validation - test split
Training:
2023-01-01 → 2024-12-31
approximately 5 million rows

Validation:
2025-01-01 → 2025-06-30
approximately 1.4 million rows

Test:
2025-07-01 → 2025-12-31
approximately 1.5 million rows

Final test:
2026-01-01 → 2026-04-14
approximately 780,000 rows

# first baseline (time only)

Baseline results
Validation MAE: 10.773844411678375
Validation RMSE: 15.499079513934674

# second baseline (time + day of week)

Baseline results
Validation MAE: 10.459917606689025
Validation RMSE: 15.195470914044426

# third baseline (time + day of week + month)

Month-aware baseline results
Validation MAE: 10.738193255434883
Validation RMSE: 15.572181380192578

# Catboost v1

Features: IDs + calendar + departure time
Iterations: 1000
MAE: 10.7731
RMSE: 15.6135
Result: 2.99% worse than weekday baseline

# CatBoost v2
Validation MAE:  9.994428
Validation RMSE: 14.652154

# CatBoost v3
Validation MAE:  9.935900
Validation RMSE: 14.601808
Test MAE:        10.165392
Test RMSE:       17.432818
Final MAE:       9.770397
Final RMSE:      13.851111

# CatBoost v4.1
Validation MAE:  9.855638
Validation RMSE: 14.618278
Test MAE:        9.911966
Test RMSE:       17.078030
Final MAE:       9.761558
Final RMSE:      13.990810

# CatBoost v4.2 hybrid
Validation MAE:  9.853707
Validation RMSE: 14.590293
Test MAE:        9.870427
Test RMSE:       16.933563
Final MAE:       9.739149
Final RMSE:      13.926252

# CatBoost v4.3 business distribution
Underlying design: v4.1 plus 8 company-route-time-weekday distribution features
Loss: MAE
Saved trees: 3647

Validation MAE 10-40: 7.863636
Test MAE 10-40:       7.097154
Final MAE 10-40:      8.287627

Validation overall MAE:  9.856505
Validation overall RMSE: 14.626180
Test overall MAE:        9.931674
Test overall RMSE:       17.113043
Final overall MAE:       9.758819
Final overall RMSE:      13.973663

Decision: do not promote. v4.3 consistently improves the primary 10-40 slice,
but the 2025 H2 test worsens false-negative rates at 20/30/43, overall MAE,
and RMSE for actual 40+ versus frozen v4.2. v4.2 remains the production candidate.
