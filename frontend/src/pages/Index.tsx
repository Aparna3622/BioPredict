
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Heart, Brain, Activity, Calendar, Users, BookOpen, Shield, TrendingUp, LogIn, UserPlus } from "lucide-react";
import { HealthAssessmentForm } from "@/components/health/HealthAssessmentForm";
import { RiskAssessment } from "@/components/health/RiskAssessment";
import { AppointmentScheduler } from "@/components/health/AppointmentScheduler";
import { HealthTracker } from "@/components/health/HealthTracker";
import { EducationalResources } from "@/components/health/EducationalResources";
import { UserProfile } from "@/components/health/UserProfile";
import { AuthModal } from "@/components/auth/AuthModal";

const Index = () => {
  const [currentUser, setCurrentUser] = useState(null);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");

  // Mock function to simulate user login - replace with real authentication
  const handleUserLogin = (userData) => {
    setCurrentUser(userData);
    setShowAuthModal(false);
  };

  const handleUserLogout = () => {
    setCurrentUser(null);
    setActiveTab("dashboard");
  };

  // If user is not logged in, show the public landing page
  if (!currentUser) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
        {/* Header */}
        <header className="bg-white/80 backdrop-blur-md border-b border-blue-100 sticky top-0 z-50">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-xl">
                  <Heart className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    Bio-Predict
                  </h1>
                  <p className="text-sm text-gray-600">Intelligent Health Risk Assessment</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  <Shield className="h-3 w-3 mr-1" />
                  HIPAA Compliant
                </Badge>
                <Button onClick={() => setShowAuthModal(true)} className="bg-blue-600 hover:bg-blue-700">
                  <LogIn className="h-4 w-4 mr-2" />
                  Sign In
                </Button>
              </div>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <main className="container mx-auto px-4 py-16">
          <div className="text-center max-w-4xl mx-auto mb-16">
            <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Predict Your Health Risks with AI
            </h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Take control of your health journey with our advanced AI-powered risk assessment platform. 
              Get personalized insights based on your family history, lifestyle, and health data.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                size="lg" 
                onClick={() => setShowAuthModal(true)}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-lg px-8 py-4"
              >
                <UserPlus className="h-5 w-5 mr-2" />
                Get Started Free
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                onClick={() => setShowAuthModal(true)}
                className="border-blue-300 text-blue-700 hover:bg-blue-50 text-lg px-8 py-4"
              >
                <LogIn className="h-5 w-5 mr-2" />
                Sign In
              </Button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
            <Card className="bg-white/70 backdrop-blur-sm border-blue-100 hover:shadow-lg transition-all duration-300">
              <CardHeader>
                <div className="bg-blue-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
                <CardTitle className="text-blue-900">AI-Powered Assessment</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Advanced machine learning algorithms analyze your health data to provide accurate risk predictions
                  and personalized recommendations.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-white/70 backdrop-blur-sm border-green-100 hover:shadow-lg transition-all duration-300">
              <CardHeader>
                <div className="bg-green-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                  <Shield className="h-6 w-6 text-green-600" />
                </div>
                <CardTitle className="text-green-900">HIPAA Compliant</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Your health information is protected with enterprise-grade security and full HIPAA compliance
                  for complete peace of mind.
                </p>
              </CardContent>
            </Card>

            <Card className="bg-white/70 backdrop-blur-sm border-purple-100 hover:shadow-lg transition-all duration-300">
              <CardHeader>
                <div className="bg-purple-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                  <Activity className="h-6 w-6 text-purple-600" />
                </div>
                <CardTitle className="text-purple-900">Comprehensive Tracking</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">
                  Monitor your health journey over time with detailed tracking, progress reports, and
                  appointment scheduling integration.
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Statistics */}
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-blue-100">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-blue-600 mb-2">10,000+</div>
                <div className="text-gray-600">Users Assessed</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-600 mb-2">95%</div>
                <div className="text-gray-600">Accuracy Rate</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-purple-600 mb-2">50+</div>
                <div className="text-gray-600">Health Conditions</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-orange-600 mb-2">24/7</div>
                <div className="text-gray-600">Support Available</div>
              </div>
            </div>
          </div>
        </main>

        <AuthModal 
          open={showAuthModal} 
          onClose={() => setShowAuthModal(false)}
          onLogin={handleUserLogin}
        />
      </div>
    );
  }

  // Determine risk metrics based on user status
  const getRiskMetrics = () => {
    // New user - show 0% risk
    if (currentUser.isNewUser) {
      return {
        cardiovascular: 0,
        diabetes: 0,
        cancer: 0,
        mental: 0
      };
    }
    
    // Returning user with completed assessment - show ML-calculated risk
    return currentUser.riskMetrics || {
      cardiovascular: 25,
      diabetes: 15,
      cancer: 30,
      mental: 20
    };
  };

  const riskMetrics = getRiskMetrics();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-blue-100 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-xl">
                <Heart className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Bio-Predict
                </h1>
                <p className="text-sm text-gray-600">Intelligent Health Risk Assessment</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                <Shield className="h-3 w-3 mr-1" />
                HIPAA Compliant
              </Badge>
              <Button variant="outline" onClick={() => setActiveTab("profile")} className="hover:bg-blue-50">
                <Users className="h-4 w-4 mr-2" />
                {currentUser.name}
              </Button>
              <Button variant="outline" onClick={handleUserLogout} className="hover:bg-red-50 text-red-600">
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-7 bg-white/50 backdrop-blur-sm">
            <TabsTrigger value="dashboard" className="flex items-center space-x-2">
              <Activity className="h-4 w-4" />
              <span className="hidden sm:inline">Dashboard</span>
            </TabsTrigger>
            <TabsTrigger value="assessment" className="flex items-center space-x-2">
              <Brain className="h-4 w-4" />
              <span className="hidden sm:inline">Assessment</span>
            </TabsTrigger>
            <TabsTrigger value="results" className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4" />
              <span className="hidden sm:inline">Results</span>
            </TabsTrigger>
            <TabsTrigger value="appointments" className="flex items-center space-x-2">
              <Calendar className="h-4 w-4" />
              <span className="hidden sm:inline">Appointments</span>
            </TabsTrigger>
            <TabsTrigger value="tracking" className="flex items-center space-x-2">
              <Heart className="h-4 w-4" />
              <span className="hidden sm:inline">Tracking</span>
            </TabsTrigger>
            <TabsTrigger value="resources" className="flex items-center space-x-2">
              <BookOpen className="h-4 w-4" />
              <span className="hidden sm:inline">Resources</span>
            </TabsTrigger>
            <TabsTrigger value="profile" className="flex items-center space-x-2">
              <Users className="h-4 w-4" />
              <span className="hidden sm:inline">Profile</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-6">
            {/* Welcome Section */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  {currentUser.isNewUser ? (
                    <>
                      <h2 className="text-3xl font-bold mb-2">Welcome to Bio-Predict, {currentUser.name}!</h2>
                      <p className="text-blue-100 mb-4">Take control of your health journey with AI-powered insights</p>
                      <div className="flex items-center space-x-4 text-sm">
                        <div className="flex items-center space-x-1">
                          <Heart className="h-4 w-4" />
                          <span>Ready to start your first health assessment</span>
                        </div>
                        <Badge variant="secondary" className="bg-white/20 text-white">
                          New User
                        </Badge>
                      </div>
                    </>
                  ) : (
                    <>
                      <h2 className="text-3xl font-bold mb-2">Welcome back, {currentUser.name}!</h2>
                      <p className="text-blue-100 mb-4">Your health journey continues here</p>
                      <div className="flex items-center space-x-4 text-sm">
                        <div className="flex items-center space-x-1">
                          <Calendar className="h-4 w-4" />
                          <span>Last assessment: {currentUser.lastAssessment}</span>
                        </div>
                        <Badge variant="secondary" className="bg-white/20 text-white">
                          Risk Level: {currentUser.riskLevel}
                        </Badge>
                      </div>
                    </>
                  )}
                </div>
                <div className="hidden md:block">
                  <div className="bg-white/20 rounded-full p-4">
                    <Heart className="h-12 w-12" />
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card className="bg-white/70 backdrop-blur-sm border-red-100 hover:shadow-lg transition-all duration-300">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-red-600">Cardiovascular Risk</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl font-bold text-red-700">{riskMetrics.cardiovascular}%</span>
                    <Heart className="h-5 w-5 text-red-500" />
                  </div>
                  <Progress value={riskMetrics.cardiovascular} className="h-2" />
                  <p className="text-xs text-gray-600 mt-2">
                    {currentUser.isNewUser ? "Complete assessment for analysis" : "Based on family history & lifestyle"}
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-orange-100 hover:shadow-lg transition-all duration-300">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-orange-600">Diabetes Risk</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl font-bold text-orange-700">{riskMetrics.diabetes}%</span>
                    <Activity className="h-5 w-5 text-orange-500" />
                  </div>
                  <Progress value={riskMetrics.diabetes} className="h-2" />
                  <p className="text-xs text-gray-600 mt-2">
                    {currentUser.isNewUser ? "Complete assessment for analysis" : "Diet & exercise factors included"}
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-purple-100 hover:shadow-lg transition-all duration-300">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-purple-600">Cancer Risk</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl font-bold text-purple-700">{riskMetrics.cancer}%</span>
                    <Brain className="h-5 w-5 text-purple-500" />
                  </div>
                  <Progress value={riskMetrics.cancer} className="h-2" />
                  <p className="text-xs text-gray-600 mt-2">
                    {currentUser.isNewUser ? "Complete assessment for analysis" : "Genetic & environmental factors"}
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-white/70 backdrop-blur-sm border-blue-100 hover:shadow-lg transition-all duration-300">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-blue-600">Mental Health</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl font-bold text-blue-700">{riskMetrics.mental}%</span>
                    <Users className="h-5 w-5 text-blue-500" />
                  </div>
                  <Progress value={riskMetrics.mental} className="h-2" />
                  <p className="text-xs text-gray-600 mt-2">
                    {currentUser.isNewUser ? "Complete assessment for analysis" : "Stress & lifestyle assessment"}
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200 hover:shadow-lg transition-all duration-300">
                <CardHeader>
                  <CardTitle className="text-green-700">
                    {currentUser.isNewUser ? "Take Your First Assessment" : "Take New Assessment"}
                  </CardTitle>
                  <CardDescription>
                    {currentUser.isNewUser 
                      ? "Start your health journey with our comprehensive AI assessment" 
                      : "Update your health profile for accurate predictions"
                    }
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={() => setActiveTab("assessment")} 
                    className="w-full bg-green-600 hover:bg-green-700"
                  >
                    {currentUser.isNewUser ? "Start First Assessment" : "Start Assessment"}
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 hover:shadow-lg transition-all duration-300">
                <CardHeader>
                  <CardTitle className="text-blue-700">Schedule Appointment</CardTitle>
                  <CardDescription>Book a consultation based on your risk profile</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={() => setActiveTab("appointments")} 
                    variant="outline" 
                    className="w-full border-blue-300 text-blue-700 hover:bg-blue-50"
                  >
                    Book Now
                  </Button>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 hover:shadow-lg transition-all duration-300">
                <CardHeader>
                  <CardTitle className="text-purple-700">View Resources</CardTitle>
                  <CardDescription>Learn about prevention and healthy living</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={() => setActiveTab("resources")} 
                    variant="outline" 
                    className="w-full border-purple-300 text-purple-700 hover:bg-purple-50"
                  >
                    Explore
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="assessment">
            <HealthAssessmentForm currentUser={currentUser} setCurrentUser={setCurrentUser} />
          </TabsContent>

          <TabsContent value="results">
            <RiskAssessment currentUser={currentUser} />
          </TabsContent>

          <TabsContent value="appointments">
            <AppointmentScheduler />
          </TabsContent>

          <TabsContent value="tracking">
            <HealthTracker />
          </TabsContent>

          <TabsContent value="resources">
            <EducationalResources />
          </TabsContent>

          <TabsContent value="profile">
            <UserProfile />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Index;
