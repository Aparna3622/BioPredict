
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { Activity, Heart, Scale, Droplets, Moon, Target, TrendingUp, Calendar } from "lucide-react";

export const HealthTracker = () => {
  const [metrics, setMetrics] = useState({
    weight: "",
    bloodPressure: "",
    heartRate: "",
    bloodSugar: "",
    steps: "",
    sleep: "",
    water: ""
  });

  const { toast } = useToast();

  const weightData = [
    { date: "Jan", weight: 75 },
    { date: "Feb", weight: 74.5 },
    { date: "Mar", weight: 74 },
    { date: "Apr", weight: 73.5 },
    { date: "May", weight: 73.2 },
    { date: "Jun", weight: 72.8 },
  ];

  const activityData = [
    { day: "Mon", steps: 8500, goal: 10000 },
    { day: "Tue", steps: 12000, goal: 10000 },
    { day: "Wed", steps: 9500, goal: 10000 },
    { day: "Thu", steps: 11200, goal: 10000 },
    { day: "Fri", steps: 7800, goal: 10000 },
    { day: "Sat", steps: 15000, goal: 10000 },
    { day: "Sun", steps: 9200, goal: 10000 },
  ];

  const handleMetricUpdate = (metric: string, value: string) => {
    setMetrics(prev => ({ ...prev, [metric]: value }));
  };

  const handleSaveMetrics = () => {
    console.log("Saving metrics:", metrics);
    toast({
      title: "Metrics Updated",
      description: "Your health metrics have been recorded successfully.",
    });
  };

  const currentGoals = {
    weight: { current: 72.8, target: 70, unit: "kg" },
    steps: { current: 9200, target: 10000, unit: "steps" },
    water: { current: 6, target: 8, unit: "glasses" },
    sleep: { current: 7.5, target: 8, unit: "hours" }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <Card className="bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
            Health Progress Tracking
          </CardTitle>
          <CardDescription>
            Monitor your health metrics over time and track progress toward your goals
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Data Input Section */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="h-5 w-5 text-green-600" />
                <span>Log Today's Metrics</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="weight">Weight (kg)</Label>
                <Input
                  id="weight"
                  type="number"
                  placeholder="72.5"
                  value={metrics.weight}
                  onChange={(e) => handleMetricUpdate("weight", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="bloodPressure">Blood Pressure</Label>
                <Input
                  id="bloodPressure"
                  placeholder="120/80"
                  value={metrics.bloodPressure}
                  onChange={(e) => handleMetricUpdate("bloodPressure", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="heartRate">Resting Heart Rate</Label>
                <Input
                  id="heartRate"
                  type="number"
                  placeholder="72"
                  value={metrics.heartRate}
                  onChange={(e) => handleMetricUpdate("heartRate", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="bloodSugar">Blood Sugar (mg/dL)</Label>
                <Input
                  id="bloodSugar"
                  type="number"
                  placeholder="95"
                  value={metrics.bloodSugar}
                  onChange={(e) => handleMetricUpdate("bloodSugar", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="steps">Steps Taken</Label>
                <Input
                  id="steps"
                  type="number"
                  placeholder="9200"
                  value={metrics.steps}
                  onChange={(e) => handleMetricUpdate("steps", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="sleep">Sleep Hours</Label>
                <Input
                  id="sleep"
                  type="number"
                  step="0.5"
                  placeholder="7.5"
                  value={metrics.sleep}
                  onChange={(e) => handleMetricUpdate("sleep", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="water">Water Intake (glasses)</Label>
                <Input
                  id="water"
                  type="number"
                  placeholder="6"
                  value={metrics.water}
                  onChange={(e) => handleMetricUpdate("water", e.target.value)}
                />
              </div>

              <Button onClick={handleSaveMetrics} className="w-full bg-green-600 hover:bg-green-700">
                Save Today's Metrics
              </Button>
            </CardContent>
          </Card>

          {/* Goals Progress */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="h-5 w-5 text-blue-600" />
                <span>Daily Goals</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.entries(currentGoals).map(([key, goal]) => (
                <div key={key} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium capitalize">{key}</span>
                    <Badge variant={goal.current >= goal.target ? "default" : "secondary"}>
                      {goal.current}/{goal.target} {goal.unit}
                    </Badge>
                  </div>
                  <Progress 
                    value={(goal.current / goal.target) * 100} 
                    className="h-2"
                  />
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Charts and Analytics */}
        <div className="lg:col-span-2 space-y-6">
          <Tabs defaultValue="trends" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="trends">Trends</TabsTrigger>
              <TabsTrigger value="activity">Activity</TabsTrigger>
              <TabsTrigger value="summary">Summary</TabsTrigger>
            </TabsList>

            <TabsContent value="trends" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5 text-blue-600" />
                    <span>Weight Progress</span>
                  </CardTitle>
                  <CardDescription>Your weight journey over the past 6 months</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={weightData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={['dataMin - 1', 'dataMax + 1']} />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="weight" 
                        stroke="#3b82f6" 
                        strokeWidth={3}
                        dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="activity" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Activity className="h-5 w-5 text-green-600" />
                    <span>Weekly Activity</span>
                  </CardTitle>
                  <CardDescription>Steps taken this week vs. daily goal</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={activityData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="day" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="steps" fill="#10b981" />
                      <Bar dataKey="goal" fill="#e5e7eb" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="summary" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-red-600">
                      <Heart className="h-5 w-5" />
                      <span>Cardiovascular</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Resting HR</span>
                        <span className="font-medium">72 bpm</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Blood Pressure</span>
                        <span className="font-medium">118/76</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Status</span>
                        <Badge className="bg-green-50 text-green-700">Excellent</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-blue-600">
                      <Droplets className="h-5 w-5" />
                      <span>Metabolic</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Blood Sugar</span>
                        <span className="font-medium">92 mg/dL</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">BMI</span>
                        <span className="font-medium">22.1</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Status</span>
                        <Badge className="bg-green-50 text-green-700">Normal</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-green-600">
                      <Activity className="h-5 w-5" />
                      <span>Activity</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Daily Average</span>
                        <span className="font-medium">9,614 steps</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Goal Achievement</span>
                        <span className="font-medium">85%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Status</span>
                        <Badge className="bg-yellow-50 text-yellow-700">Good</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-purple-600">
                      <Moon className="h-5 w-5" />
                      <span>Sleep</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Average Sleep</span>
                        <span className="font-medium">7.2 hours</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Sleep Quality</span>
                        <span className="font-medium">Good</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Status</span>
                        <Badge className="bg-blue-50 text-blue-700">Improving</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};
