# 📊 Telecom Customer Churn Prediction

![Intro](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/customer%20churn.jpeg?raw=true)

## 🔎 What is Customer Churn?

Customer churn occurs when customers stop using a company's services. In the highly competitive telecom industry, churn rates range from 15% to 25% annually. Retaining customers is cheaper than acquiring new ones, making churn prediction crucial for business success. By predicting churn, companies can focus on retaining high-risk customers.

---

## 🎯 Objectives

- Identify the percentage of churned and retained customers.
- Analyze data to uncover factors influencing churn.
- Build and evaluate machine learning models to predict churn.

---

## 📂 Dataset

- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/bhartiprasad17/customer-churn-prediction/data)
- **Details:** Includes service subscriptions, demographic information, and churn status.

---

## 🛠️ Technologies Used

- **Libraries:** `sklearn`, `matplotlib`, `pandas`, `seaborn`, `numpy`
- **Machine Learning Models:** Logistic Regression, KNN, Naive Bayes, Decision Trees, Random Forest, AdaBoost, Gradient Boosting, Voting Classifier

---

## 📈 Exploratory Data Analysis (EDA)

### Churn Distribution
![Churn Distribution](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/Churn%20Distribution.png?raw=true)

### Gender-based Churn Distribution
![Churn Distribution wrt Gender](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/distributionWRTGender.PNG?raw=true)

### Customer Contract Distribution
![Contract Distribution](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/Contract%20distribution.png?raw=true)

### Payment Methods
![Payment Methods](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/payment%20methods.png?raw=true)
![Churn by Payment Method](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/payment%20ethods%20with%20respectto%20churn.PNG?raw=true)

### Internet Services
![Internet Services](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/internet%20services.PNG?raw=true)

### Dependent Status
![Dependents](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/dependents.PNG?raw=true)

### Online Security
![Online Security](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/onlineSecurity.PNG?raw=true)

### Senior Citizens
![Senior Citizens](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/seniorCitzen.PNG?raw=true)

### Paperless Billing
![Paperless Billing](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/billing.PNG?raw=true)

### Tech Support
![Tech Support](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/techSupport.PNG?raw=true)

### Charges and Tenure
![Monthly Charges](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/carges%20distribution.PNG?raw=true)
![Total Charges](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/total%20charges.PNG?raw=true)
![Tenure](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/tenure%20and%20churn.PNG?raw=true)

---

## 🤖 Model Performance

### Algorithms Evaluated:
- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- Voting Classifier (final model)

![Model Evaluation](https://github.com/HarshRaj00789/Customer-Churn-Prediction/blob/main/output/Model%20evaluation.PNG?raw=true)

### Final Model - Voting Classifier:
Combines Gradient Boosting, Logistic Regression, and AdaBoost for best results.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()

voting_clf = VotingClassifier(estimators=[
    ('gbc', clf1), 
    ('lr', clf2), 
    ('abc', clf3)
], voting='soft')

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, predictions))
```
```
Final Score 
{'LogisticRegression': [0.841331397558646, 0.010495252078550477],
 'KNeighborsClassifier': [0.7913242024807321, 0.008198993337848612],
 'GaussianNB': [0.8232386881685605, 0.00741678015498337],
 'DecisionTreeClassifier': [0.6470213137060805, 0.02196953973039052],
 'RandomForestClassifier': [0.8197874155380965, 0.011556155864106703],
 'AdaBoostClassifier': [0.8445838813774079, 0.01125665302188384],
 'GradientBoostingClassifier': [0.844630629931458, 0.010723107447558198],
 'VotingClassifier': [0.8468096379573085, 0.010887508320460332]}

```
* Final confusion matrix we got:
<img src= "https://github.com/Pradnya1208/Telecom-Customer-Churn-prediction/blob/main/output/confusion%20matrix.PNG?raw=true" width = "425" />

>From the confusion matrix we can see that: There are total 1383+166=1549 actual non-churn values and the algorithm predicts 1400 of them as non churn and 149 of them as churn. While there are 280+280=561 actual churn values and the algorithm predicts 280 of them as non churn values and 281 of them as churn values.
## Optimizations

We could use Hyperparamete Tuning or Feature enginnering methods to improve the accuracy further.

