import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { ChevronRight, ChevronLeft, User, Heart, Activity, Utensils, Brain } from "lucide-react";
import { AssessmentSubmissionNotification } from "./AssessmentSubmissionNotification";

interface HealthAssessmentFormProps {
  currentUser?: any;
  setCurrentUser?: (user: any) => void;
}

export const HealthAssessmentForm = ({ currentUser, setCurrentUser }: HealthAssessmentFormProps) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [submissionData, setSubmissionData] = useState({
    expectedDeliveryTime: "",
    assignedSpecialist: "",
    specialistType: ""
  });
  
  const [formData, setFormData] = useState({
    // Personal Information
    age: "",
    gender: "",
    height: "",
    weight: "",
    // Family History
    familyHistory: {
      heartDisease: false,
      diabetes: false,
      cancer: false,
      stroke: false,
      mentalHealth: false,
    },
    // Lifestyle
    smoking: "",
    alcohol: "",
    exercise: "",
    diet: "",
    stress: "",
    sleep: "",
    // Medical History
    conditions: "",
    medications: "",
    allergies: "",
  });

  const [prediction, setPrediction] = useState<string | null>(null);

  const { toast } = useToast();

  // Machine Learning Risk Calculation Algorithm
  const calculateMLRisk = (data: typeof formData) => {
    console.log("Processing assessment data through ML algorithm...", data);
    
    let cardiovascularRisk = 10; // Base risk
    let diabetesRisk = 5;
    let cancerRisk = 15;
    let mentalHealthRisk = 8;

    // Age factor
    const age = parseInt(data.age) || 0;
    if (age > 50) {
      cardiovascularRisk += 15;
      diabetesRisk += 10;
      cancerRisk += 12;
    } else if (age > 35) {
      cardiovascularRisk += 8;
      diabetesRisk += 5;
      cancerRisk += 6;
    }

    // Family history factors
    if (data.familyHistory.heartDisease) cardiovascularRisk += 20;
    if (data.familyHistory.diabetes) diabetesRisk += 25;
    if (data.familyHistory.cancer) cancerRisk += 18;
    if (data.familyHistory.stroke) cardiovascularRisk += 15;
    if (data.familyHistory.mentalHealth) mentalHealthRisk += 15;

    // Lifestyle factors
    if (data.smoking === "current") {
      cardiovascularRisk += 25;
      cancerRisk += 30;
    } else if (data.smoking === "former") {
      cardiovascularRisk += 10;
      cancerRisk += 12;
    }

    if (data.alcohol === "heavy") {
      cardiovascularRisk += 15;
      cancerRisk += 10;
    }

    if (data.exercise === "none") {
      cardiovascularRisk += 12;
      diabetesRisk += 15;
      mentalHealthRisk += 10;
    } else if (data.exercise === "high") {
      cardiovascularRisk -= 8;
      diabetesRisk -= 10;
      mentalHealthRisk -= 5;
    }

    if (data.diet === "poor") {
      cardiovascularRisk += 10;
      diabetesRisk += 12;
    } else if (data.diet === "excellent") {
      cardiovascularRisk -= 5;
      diabetesRisk -= 8;
    }

    if (data.stress === "high" || data.stress === "severe") {
      cardiovascularRisk += 12;
      mentalHealthRisk += 20;
    }

    if (data.sleep === "poor") {
      cardiovascularRisk += 8;
      mentalHealthRisk += 15;
    }

    // BMI calculation if height and weight provided
    const height = parseFloat(data.height) / 100; // Convert cm to m
    const weight = parseFloat(data.weight);
    if (height && weight) {
      const bmi = weight / (height * height);
      if (bmi > 30) {
        cardiovascularRisk += 15;
        diabetesRisk += 20;
      } else if (bmi > 25) {
        cardiovascularRisk += 8;
        diabetesRisk += 10;
      }
    }

    // Cap risks at reasonable maximums
    cardiovascularRisk = Math.min(cardiovascularRisk, 85);
    diabetesRisk = Math.min(diabetesRisk, 75);
    cancerRisk = Math.min(cancerRisk, 80);
    mentalHealthRisk = Math.min(mentalHealthRisk, 70);

    console.log("ML Risk Calculation Results:", {
      cardiovascular: cardiovascularRisk,
      diabetes: diabetesRisk,
      cancer: cancerRisk,
      mental: mentalHealthRisk
    });

    return {
      cardiovascular: cardiovascularRisk,
      diabetes: diabetesRisk,
      cancer: cancerRisk,
      mental: mentalHealthRisk
    };
  };

  // Function to assign specialist based on risk factors
  const assignSpecialist = (data: typeof formData) => {
    const specialists = [
      { name: "Sarah Johnson", type: "Cardiologist", specialty: "heart" },
      { name: "Michael Chen", type: "Endocrinologist", specialty: "diabetes" },
      { name: "Emily Davis", type: "Oncologist", specialty: "cancer" },
      { name: "Robert Wilson", type: "Internal Medicine", specialty: "general" },
      { name: "Lisa Anderson", type: "Preventive Medicine", specialty: "lifestyle" }
    ];

    // Simple logic to assign specialist based on family history and lifestyle
    if (data.familyHistory.heartDisease || data.smoking === "current") {
      return specialists[0]; // Cardiologist
    } else if (data.familyHistory.diabetes || data.diet === "poor") {
      return specialists[1]; // Endocrinologist
    } else if (data.familyHistory.cancer) {
      return specialists[2]; // Oncologist
    } else if (data.exercise === "none" || data.stress === "high") {
      return specialists[4]; // Preventive Medicine
    } else {
      return specialists[3]; // Internal Medicine
    }
  };

  // Function to calculate expected delivery time (24 hours + processing time)
  const calculateDeliveryTime = () => {
    const now = new Date();
    const deliveryTime = new Date(now.getTime() + 25 * 60 * 60 * 1000); // 25 hours from now
    return deliveryTime.toLocaleDateString('en-US', { 
      weekday: 'long',
      year: 'numeric', 
      month: 'long', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleFamilyHistoryChange = (condition: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      familyHistory: {
        ...prev.familyHistory,
        [condition]: checked
      }
    }));
  };

  const nextStep = () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async (userData: any) => {
    console.log("Assessment Data:", userData);
    // Calculate ML-based risk assessment
    const mlRiskResults = calculateMLRisk(userData);
    // Assign specialist and calculate delivery time
    const specialist = assignSpecialist(userData);
    const deliveryTime = calculateDeliveryTime();
    setSubmissionData({
      expectedDeliveryTime: deliveryTime,
      assignedSpecialist: specialist.name,
      specialistType: specialist.type
    });
    // Update user status if this is their first assessment
    if (currentUser && currentUser.isNewUser && setCurrentUser) {
      const updatedUser = {
        ...currentUser,
        isNewUser: false,
        lastAssessment: new Date().toLocaleDateString(),
        riskLevel: "Moderate", // This would be determined by ML algorithm
        riskMetrics: mlRiskResults,
        hasCompletedAssessment: true
      };
      console.log("Updating user with ML risk results:", updatedUser);
      setCurrentUser(updatedUser);
    }
    await handlePredict(userData); // Get prediction before showing notification

    // --- Send email to user ---
    // You may want to collect the user's email as part of the form. Here, we assume userData.email exists.
    if (userData.email) {
      const emailBody = `Thank you for completing your health risk assessment.\n\nYour responses:\n${Object.entries(userData).map(([k,v]) => `${k}: ${v}`).join("\n")}\n\nA doctor will contact you soon to discuss your results.`;
      try {
        await fetch('https://biopredict.onrender.com/send_email', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            recipient: userData.email,
            subject: 'Your Health Risk Assessment Submission',
            body: emailBody
          })
        });
      } catch (e) {
        // Optionally handle email error
        console.error('Failed to send email:', e);
      }
    }

    setIsSubmitted(true);
    toast({
      title: "Assessment Submitted Successfully!",
      description: `Your AI analysis is complete. Report will be reviewed by Dr. ${specialist.name} and delivered by ${deliveryTime}`,
    });
  };

  // If assessment is submitted, show the notification
  if (isSubmitted) {
    return (
      <div className="max-w-4xl mx-auto">
        <AssessmentSubmissionNotification 
          expectedDeliveryTime={submissionData.expectedDeliveryTime}
          assignedSpecialist={submissionData.assignedSpecialist}
          specialistType={submissionData.specialistType}
        />
        {prediction !== null ? (
          <div className="my-4 p-4 bg-blue-50 rounded">
            <h2 className="text-lg font-semibold text-blue-700">
              Your Health Risk Prediction: {prediction}
            </h2>
          </div>
        ) : (
          <div className="my-4 p-4 bg-yellow-50 rounded">
            <h2 className="text-lg font-semibold text-yellow-700">
              Prediction is being generated...
            </h2>
          </div>
        )}
      </div>
    );
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <div className="flex items-center space-x-2 mb-4">
              <User className="h-5 w-5 text-blue-600" />
              <h3 className="text-xl font-semibold">Personal Information</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  placeholder="Enter your age"
                  value={formData.age}
                  onChange={(e) => handleInputChange("age", e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="gender">Gender</Label>
                <Select value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select gender" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="male">Male</SelectItem>
                    <SelectItem value="female">Female</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                    <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="height">Height (cm)</Label>
                <Input
                  id="height"
                  type="number"
                  placeholder="Enter height in cm"
                  value={formData.height}
                  onChange={(e) => handleInputChange("height", e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="weight">Weight (kg)</Label>
                <Input
                  id="weight"
                  type="number"
                  placeholder="Enter weight in kg"
                  value={formData.weight}
                  onChange={(e) => handleInputChange("weight", e.target.value)}
                />
              </div>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div className="flex items-center space-x-2 mb-4">
              <Heart className="h-5 w-5 text-red-600" />
              <h3 className="text-xl font-semibold">Family Medical History</h3>
            </div>
            <p className="text-gray-600 mb-4">Select any conditions that run in your family:</p>
            <div className="space-y-4">
              {Object.entries(formData.familyHistory).map(([condition, checked]) => (
                <div key={condition} className="flex items-center space-x-2">
                  <Checkbox
                    id={condition}
                    checked={checked}
                    onCheckedChange={(checked) => handleFamilyHistoryChange(condition, checked as boolean)}
                  />
                  <Label htmlFor={condition} className="capitalize">
                    {condition.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                  </Label>
                </div>
              ))}
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="flex items-center space-x-2 mb-4">
              <Activity className="h-5 w-5 text-green-600" />
              <h3 className="text-xl font-semibold">Lifestyle Information</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="smoking">Smoking Status</Label>
                <Select value={formData.smoking} onValueChange={(value) => handleInputChange("smoking", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select smoking status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="never">Never smoked</SelectItem>
                    <SelectItem value="former">Former smoker</SelectItem>
                    <SelectItem value="current">Current smoker</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="alcohol">Alcohol Consumption</Label>
                <Select value={formData.alcohol} onValueChange={(value) => handleInputChange("alcohol", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select alcohol consumption" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="light">Light (1-3 drinks/week)</SelectItem>
                    <SelectItem value="moderate">Moderate (4-7 drinks/week)</SelectItem>
                    <SelectItem value="heavy">Heavy (8+ drinks/week)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="exercise">Exercise Frequency</Label>
                <Select value={formData.exercise} onValueChange={(value) => handleInputChange("exercise", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select exercise frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="light">1-2 times/week</SelectItem>
                    <SelectItem value="moderate">3-4 times/week</SelectItem>
                    <SelectItem value="high">5+ times/week</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="diet">Diet Quality</Label>
                <Select value={formData.diet} onValueChange={(value) => handleInputChange("diet", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select diet quality" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="poor">Poor</SelectItem>
                    <SelectItem value="fair">Fair</SelectItem>
                    <SelectItem value="good">Good</SelectItem>
                    <SelectItem value="excellent">Excellent</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="stress">Stress Level</Label>
                <Select value={formData.stress} onValueChange={(value) => handleInputChange("stress", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select stress level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="moderate">Moderate</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="severe">Severe</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="sleep">Sleep Quality</Label>
                <Select value={formData.sleep} onValueChange={(value) => handleInputChange("sleep", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select sleep quality" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="poor">Poor (4-5 hours)</SelectItem>
                    <SelectItem value="fair">Fair (5-6 hours)</SelectItem>
                    <SelectItem value="good">Good (7-8 hours)</SelectItem>
                    <SelectItem value="excellent">Excellent (8+ hours)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="flex items-center space-x-2 mb-4">
              <Utensils className="h-5 w-5 text-purple-600" />
              <h3 className="text-xl font-semibold">Medical History</h3>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="conditions">Current Medical Conditions</Label>
                <Textarea
                  id="conditions"
                  placeholder="List any current medical conditions..."
                  value={formData.conditions}
                  onChange={(e) => handleInputChange("conditions", e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="medications">Current Medications</Label>
                <Textarea
                  id="medications"
                  placeholder="List current medications and dosages..."
                  value={formData.medications}
                  onChange={(e) => handleInputChange("medications", e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="allergies">Allergies</Label>
                <Textarea
                  id="allergies"
                  placeholder="List any known allergies..."
                  value={formData.allergies}
                  onChange={(e) => handleInputChange("allergies", e.target.value)}
                />
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  const handlePredict = async (formData: any) => {
    try {
      const response = await fetch('https://biopredict.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const result = await response.json();
      setPrediction(result.prediction);
    } catch (error) {
      setPrediction('Error fetching prediction');
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <Card className="bg-white/80 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent flex items-center space-x-2">
            <Brain className="h-6 w-6 text-blue-600" />
            <span>AI-Powered Health Risk Assessment</span>
          </CardTitle>
          <CardDescription>
            Complete this comprehensive assessment to receive personalized health risk predictions powered by machine learning
          </CardDescription>
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-2">
              <span>Step {currentStep} of 4</span>
              <span>{Math.round((currentStep / 4) * 100)}% Complete</span>
            </div>
            <Progress value={(currentStep / 4) * 100} className="h-2" />
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {renderStepContent()}
          
          {prediction && (
            <div className="my-4 p-4 bg-blue-50 rounded">
              <h2 className="text-lg font-semibold text-blue-700">
                Your Health Risk Prediction: {prediction}
              </h2>
            </div>
          )}
          
          <div className="flex justify-between pt-6">
            <Button
              variant="outline"
              onClick={prevStep}
              disabled={currentStep === 1}
              className="flex items-center space-x-2"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Previous</span>
            </Button>
            
            {currentStep === 4 ? (
              <Button
                onClick={async () => {
                  const userData = {
                    age: Number(formData.age),
                    gender: formData.gender,
                    height: Number(formData.height),
                    weight: Number(formData.weight),
                    familyHistory: Object.keys(formData.familyHistory)
                      .filter(key => formData.familyHistory[key])
                      .join(','),
                    smoking: formData.smoking,
                    alcohol: formData.alcohol,
                    exercise: formData.exercise,
                    diet: formData.diet,
                    stress: formData.stress,
                    sleep: formData.sleep,
                    conditions: formData.conditions,
                    medications: formData.medications,
                    allergies: formData.allergies,
                  };
                  await handleSubmit(userData); // Only call this
                }}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 flex items-center space-x-2"
              >
                <Brain className="h-4 w-4" />
                <span>Submit for AI Analysis</span>
                <ChevronRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={nextStep}
                className="flex items-center space-x-2"
              >
                <span>Next</span>
                <ChevronRight className="h-4 w-4" />
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
