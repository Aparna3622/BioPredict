{
  "cell_type": "code",
  "metadata": {
    "language": "python"
  },
  "source": [
    "# Add a synthetic 'risk' label based on simple rules",
    "def assign_risk(row):",
    "    risk_score = 0",
    "    if row['age'] >= 50: risk_score += 2",
    "    if 'heartDisease' in str(row['familyHistory']) or 'stroke' in str(row['familyHistory']) or 'diabetes' in str(row['familyHistory']): risk_score += 2",
    "    if row['smoking'] == 'current': risk_score += 2",
    "    if row['alcohol'] == 'heavy': risk_score += 1",
    "    if row['exercise'] == 'none': risk_score += 1",
    "    if row['diet'] in ['poor', 'fair']: risk_score += 1",
    "    if row['stress'] == 'high': risk_score += 1",
    "    if row['sleep'] == 'poor': risk_score += 1",
    "    if pd.notnull(row['conditions']) and len(str(row['conditions'])) > 0: risk_score += 1",
    "    if risk_score >= 5:",
    "        return 'High'",
    "    elif risk_score >= 3:",
    "        return 'Medium'",
    "    else:",
    "        return 'Low'",
    "",
    "df['risk'] = df.apply(assign_risk, axis=1)",
    "",
    "# Preview the updated dataframe",
    "df[['age', 'familyHistory', 'smoking', 'alcohol', 'exercise', 'diet', 'stress', 'sleep', 'conditions', 'risk']].head()",
    "",
    "# Save the updated dataset with the new 'risk' column",
    "df.to_csv('health_user.csv.xls', index=False)",
    "print('✅ Updated dataset saved with risk labels.')"
  ]
}