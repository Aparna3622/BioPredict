{
  "cell_type": "code",
  "metadata": {
    "language": "python"
  },
  "source": [
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.svm import SVC",
    "from sklearn.tree import DecisionTreeClassifier",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier",
    "from sklearn.neighbors import KNeighborsClassifier",
    "from sklearn.metrics import accuracy_score, classification_report",
    "",
    "models = {",
    "    'Logistic Regression': LogisticRegression(max_iter=1000),",
    "    'SVM': SVC(),",
    "    'Decision Tree': DecisionTreeClassifier(),",
    "    'Random Forest': RandomForestClassifier(),",
    "    'Gradient Boosting': GradientBoostingClassifier(),",
    "    'KNN': KNeighborsClassifier()",
    "}",
    "",
    "results = {}",
    "for name, model in models.items():",
    "    model.fit(X_train, y_train)",
    "    y_pred = model.predict(X_test)",
    "    acc = accuracy_score(y_test, y_pred)",
    "    results[name] = acc",
    "    print(f'\\n{name} Accuracy: {acc:.4f}')",
    "    print(classification_report(y_test, y_pred))",
    "",
    "best_model = max(results, key=results.get)",
    "print(f'\\nBest model: {best_model} with accuracy {results[best_model]:.4f}')"
  ]
}