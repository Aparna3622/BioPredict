
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Heart, Brain, Activity, Users, Clock, Sparkles } from "lucide-react";

interface RiskAssessmentProps {
  currentUser?: any;
}

export const RiskAssessment = ({ currentUser }: RiskAssessmentProps) => {
  // Show different content for new users vs users with completed assessments
  if (currentUser?.isNewUser || !currentUser?.hasCompletedAssessment) {
    return (
      <div className="max-w-6xl mx-auto space-y-8">
        <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent flex items-center space-x-2">
              <Brain className="h-6 w-6 text-blue-600" />
              <span>Complete Your Assessment First</span>
            </CardTitle>
            <CardDescription>
              Take our comprehensive health assessment to receive AI-powered risk predictions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Alert className="bg-blue-50 border-blue-200 mb-6">
              <Clock className="h-4 w-4" />
              <AlertTitle>Ready to Begin?</AlertTitle>
              <AlertDescription>
                Our advanced machine learning algorithm will analyze your health data to provide personalized risk assessments and recommendations.
              </AlertDescription>
            </Alert>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="bg-white/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-blue-600">
                    <Sparkles className="h-5 w-5" />
                    <span>AI-Powered Analysis</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">
                    Our machine learning algorithms analyze over 50 health factors to provide accurate risk predictions tailored to your unique profile.
                  </p>
                </CardContent>
              </Card>
              
              <Card className="bg-white/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-green-600">
                    <CheckCircle className="h-5 w-5" />
                    <span>Comprehensive Report</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">
                    Receive detailed insights on cardiovascular, diabetes, cancer, and mental health risks with personalized recommendations.
                  </p>
                </CardContent>
              </Card>
            </div>
            
            <div className="mt-6 text-center">
              <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                Start Health Assessment
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Use ML-calculated risk results if available
  const riskResults = {
    overall: currentUser?.riskLevel || "Low-Moderate",
    cardiovascular: { 
      score: currentUser?.riskMetrics?.cardiovascular || 25, 
      level: currentUser?.riskMetrics?.cardiovascular > 50 ? "High" : currentUser?.riskMetrics?.cardiovascular > 25 ? "Moderate" : "Low", 
      trend: "stable" 
    },
    diabetes: { 
      score: currentUser?.riskMetrics?.diabetes || 15, 
      level: currentUser?.riskMetrics?.diabetes > 50 ? "High" : currentUser?.riskMetrics?.diabetes > 25 ? "Moderate" : "Low", 
      trend: "improving" 
    },
    cancer: { 
      score: currentUser?.riskMetrics?.cancer || 30, 
      level: currentUser?.riskMetrics?.cancer > 50 ? "High" : currentUser?.riskMetrics?.cancer > 25 ? "Moderate" : "Low", 
      trend: "stable" 
    },
    mental: { 
      score: currentUser?.riskMetrics?.mental || 20, 
      level: currentUser?.riskMetrics?.mental > 50 ? "High" : currentUser?.riskMetrics?.mental > 25 ? "Moderate" : "Low", 
      trend: "improving" 
    }
  };

  const recommendations = [
    {
      category: "Cardiovascular Health",
      priority: riskResults.cardiovascular.score > 50 ? "High" : riskResults.cardiovascular.score > 25 ? "Medium" : "Low",
      suggestions: [
        "Increase aerobic exercise to 150 minutes per week",
        "Reduce sodium intake to less than 2300mg daily",
        "Consider omega-3 supplements after consulting your doctor"
      ]
    },
    {
      category: "Cancer Prevention",
      priority: riskResults.cancer.score > 50 ? "High" : riskResults.cancer.score > 25 ? "Medium" : "Low",
      suggestions: [
        "Schedule regular screening appointments",
        "Increase antioxidant-rich foods in your diet",
        "Limit processed meat consumption"
      ]
    },
    {
      category: "Mental Health",
      priority: riskResults.mental.score > 50 ? "High" : riskResults.mental.score > 25 ? "Medium" : "Low",
      suggestions: [
        "Continue current stress management practices",
        "Maintain social connections",
        "Consider mindfulness or meditation apps"
      ]
    }
  ];

  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "low": return "text-green-600 bg-green-50 border-green-200";
      case "moderate": return "text-yellow-600 bg-yellow-50 border-yellow-200";
      case "high": return "text-red-600 bg-red-50 border-red-200";
      default: return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case "improving": return <TrendingDown className="h-4 w-4 text-green-500" />;
      case "worsening": return <TrendingUp className="h-4 w-4 text-red-500" />;
      default: return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Overall Risk Summary */}
      <Card className="bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent flex items-center space-x-2">
            <Brain className="h-6 w-6 text-blue-600" />
            <span>Your AI-Powered Health Risk Assessment Results</span>
          </CardTitle>
          <CardDescription>
            Based on your comprehensive health profile analysis using advanced machine learning algorithms
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Overall Risk Level</h3>
              <Badge className={`mt-2 ${getRiskColor(riskResults.overall)}`}>
                {riskResults.overall} Risk
              </Badge>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600">Assessment Date</p>
              <p className="font-semibold">{currentUser?.lastAssessment || "Recent"}</p>
            </div>
          </div>
          <Alert className="bg-blue-50 border-blue-200">
            <CheckCircle className="h-4 w-4" />
            <AlertTitle>AI Analysis Complete!</AlertTitle>
            <AlertDescription>
              Our machine learning algorithm has processed your health data. Continue with the recommendations below to maintain and improve your health outcomes.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      {/* Detailed Risk Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="hover:shadow-lg transition-all duration-300">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2 text-red-600">
                <Heart className="h-5 w-5" />
                <span>Cardiovascular Risk</span>
              </CardTitle>
              {getTrendIcon(riskResults.cardiovascular.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{riskResults.cardiovascular.score}%</span>
                <Badge className={getRiskColor(riskResults.cardiovascular.level)}>
                  {riskResults.cardiovascular.level}
                </Badge>
              </div>
              <Progress value={riskResults.cardiovascular.score} className="h-3" />
              <p className="text-sm text-gray-600">
                ML analysis based on family history, lifestyle factors, and current health metrics
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-all duration-300">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2 text-orange-600">
                <Activity className="h-5 w-5" />
                <span>Diabetes Risk</span>
              </CardTitle>
              {getTrendIcon(riskResults.diabetes.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{riskResults.diabetes.score}%</span>
                <Badge className={getRiskColor(riskResults.diabetes.level)}>
                  {riskResults.diabetes.level}
                </Badge>
              </div>
              <Progress value={riskResults.diabetes.score} className="h-3" />
              <p className="text-sm text-gray-600">
                AI assessment of diet, exercise, BMI, and genetic predisposition
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-all duration-300">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2 text-purple-600">
                <Brain className="h-5 w-5" />
                <span>Cancer Risk</span>
              </CardTitle>
              {getTrendIcon(riskResults.cancer.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{riskResults.cancer.score}%</span>
                <Badge className={getRiskColor(riskResults.cancer.level)}>
                  {riskResults.cancer.level}
                </Badge>
              </div>
              <Progress value={riskResults.cancer.score} className="h-3" />
              <p className="text-sm text-gray-600">
                Machine learning analysis of genetic factors, lifestyle, and environmental exposure
              </p>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-all duration-300">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center space-x-2 text-blue-600">
                <Users className="h-5 w-5" />
                <span>Mental Health Risk</span>
              </CardTitle>
              {getTrendIcon(riskResults.mental.trend)}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold">{riskResults.mental.score}%</span>
                <Badge className={getRiskColor(riskResults.mental.level)}>
                  {riskResults.mental.level}
                </Badge>
              </div>
              <Progress value={riskResults.mental.score} className="h-3" />
              <p className="text-sm text-gray-600">
                AI evaluation of stress levels, social support, and lifestyle factors
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Personalized Recommendations */}
      <Card className="bg-white/80 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-xl text-gray-900 flex items-center space-x-2">
            <Sparkles className="h-5 w-5 text-purple-600" />
            <span>AI-Generated Personalized Health Recommendations</span>
          </CardTitle>
          <CardDescription>
            Machine learning-generated suggestions based on your risk profile and latest medical research
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {recommendations.map((rec, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-gray-900">{rec.category}</h4>
                <Badge variant={rec.priority === "High" ? "destructive" : rec.priority === "Medium" ? "default" : "secondary"}>
                  {rec.priority} Priority
                </Badge>
              </div>
              <ul className="space-y-1">
                {rec.suggestions.map((suggestion, idx) => (
                  <li key={idx} className="text-sm text-gray-600 flex items-start space-x-2">
                    <CheckCircle className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <Button className="bg-blue-600 hover:bg-blue-700">
          Schedule Follow-up Assessment
        </Button>
        <Button variant="outline">
          Download Detailed Report
        </Button>
        <Button variant="outline">
          Share with Healthcare Provider
        </Button>
      </div>
    </div>
  );
};
