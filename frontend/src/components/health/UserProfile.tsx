
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { User, Shield, Bell, Key, Download, Share2, Settings } from "lucide-react";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

export const UserProfile = () => {
  const [profile, setProfile] = useState({
    name: "Sarah Johnson",
    email: "sarah.johnson@email.com",
    phone: "+1 (555) 123-4567",
    dateOfBirth: "1990-03-15",
    emergencyContact: "John Johnson - +1 (555) 987-6543",
    medicalId: "MRN-789456123",
    notifications: {
      email: true,
      sms: true,
      push: true,
      reminders: true
    }
  });

  const { toast } = useToast();

  const handleProfileUpdate = () => {
    toast({
      title: "Profile Updated",
      description: "Your profile information has been saved successfully.",
    });
  };

  const assessmentHistory = [
    {
      date: "2025-06-20",
      overallRisk: "Low-Moderate",
      cardiovascular: 25,
      diabetes: 15,
      status: "Completed"
    },
    {
      date: "2025-03-15",
      overallRisk: "Moderate",
      cardiovascular: 28,
      diabetes: 18,
      status: "Completed"
    },
    {
      date: "2024-12-10",
      overallRisk: "Moderate",
      cardiovascular: 30,
      diabetes: 22,
      status: "Completed"
    }
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardHeader>
          <div className="flex items-center space-x-4">
            <Avatar className="h-16 w-16">
              <AvatarImage src="/placeholder.svg" />
              <AvatarFallback className="bg-blue-100 text-blue-600 text-xl font-semibold">
                SJ
              </AvatarFallback>
            </Avatar>
            <div>
              <CardTitle className="text-2xl text-gray-900">{profile.name}</CardTitle>
              <CardDescription className="text-lg">{profile.email}</CardDescription>
              <div className="flex items-center space-x-2 mt-2">
                <Badge className="bg-green-100 text-green-700">Active Member</Badge>
                <Badge variant="outline">HIPAA Compliant</Badge>
              </div>
            </div>
          </div>
        </CardHeader>
      </Card>

      <Tabs defaultValue="personal" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="personal" className="flex items-center space-x-2">
            <User className="h-4 w-4" />
            <span>Personal</span>
          </TabsTrigger>
          <TabsTrigger value="privacy" className="flex items-center space-x-2">
            <Shield className="h-4 w-4" />
            <span>Privacy</span>
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center space-x-2">
            <Bell className="h-4 w-4" />
            <span>Notifications</span>
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center space-x-2">
            <Settings className="h-4 w-4" />
            <span>History</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="personal" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Personal Information</CardTitle>
              <CardDescription>
                Update your personal details and contact information
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input
                    id="name"
                    value={profile.name}
                    onChange={(e) => setProfile(prev => ({ ...prev, name: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email">Email Address</Label>
                  <Input
                    id="email"
                    type="email"
                    value={profile.email}
                    onChange={(e) => setProfile(prev => ({ ...prev, email: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <Input
                    id="phone"
                    value={profile.phone}
                    onChange={(e) => setProfile(prev => ({ ...prev, phone: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="dob">Date of Birth</Label>
                  <Input
                    id="dob"
                    type="date"
                    value={profile.dateOfBirth}
                    onChange={(e) => setProfile(prev => ({ ...prev, dateOfBirth: e.target.value }))}
                  />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="emergency">Emergency Contact</Label>
                <Input
                  id="emergency"
                  value={profile.emergencyContact}
                  onChange={(e) => setProfile(prev => ({ ...prev, emergencyContact: e.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="medicalId">Medical Record Number</Label>
                <Input
                  id="medicalId"
                  value={profile.medicalId}
                  disabled
                  className="bg-gray-50"
                />
              </div>
              <Button onClick={handleProfileUpdate} className="w-full md:w-auto">
                Save Changes
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="privacy" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5 text-green-600" />
                <span>Privacy & Security</span>
              </CardTitle>
              <CardDescription>
                Manage your data privacy preferences and security settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Shield className="h-4 w-4 text-green-600" />
                  <span className="font-medium text-green-800">HIPAA Compliance</span>
                </div>
                <p className="text-sm text-green-700">
                  Your health information is protected under HIPAA regulations. We use end-to-end encryption
                  and secure servers to protect your sensitive medical data.
                </p>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Data Sharing with Healthcare Providers</h4>
                    <p className="text-sm text-gray-600">Allow sharing of assessment results with your doctors</p>
                  </div>
                  <Switch defaultChecked />
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Research Participation</h4>
                    <p className="text-sm text-gray-600">Contribute anonymized data to health research studies</p>
                  </div>
                  <Switch />
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Marketing Communications</h4>
                    <p className="text-sm text-gray-600">Receive health tips and product updates</p>
                  </div>
                  <Switch defaultChecked />
                </div>
              </div>

              <div className="space-y-4 pt-4 border-t">
                <h4 className="font-medium">Data Management</h4>
                <div className="flex flex-col sm:flex-row gap-2">
                  <Button variant="outline" className="flex items-center space-x-2">
                    <Download className="h-4 w-4" />
                    <span>Download My Data</span>
                  </Button>
                  <Button variant="outline" className="flex items-center space-x-2">
                    <Share2 className="h-4 w-4" />
                    <span>Transfer Data</span>
                  </Button>
                  <Button variant="destructive" className="flex items-center space-x-2">
                    <Key className="h-4 w-4" />
                    <span>Delete Account</span>
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Bell className="h-5 w-5 text-blue-600" />
                <span>Notification Preferences</span>
              </CardTitle>
              <CardDescription>
                Choose how you'd like to receive updates and reminders
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Email Notifications</h4>
                    <p className="text-sm text-gray-600">Assessment results, appointments, and health tips</p>
                  </div>
                  <Switch 
                    checked={profile.notifications.email}
                    onCheckedChange={(checked) => 
                      setProfile(prev => ({ 
                        ...prev, 
                        notifications: { ...prev.notifications, email: checked }
                      }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">SMS Notifications</h4>
                    <p className="text-sm text-gray-600">Appointment reminders and urgent health alerts</p>
                  </div>
                  <Switch 
                    checked={profile.notifications.sms}
                    onCheckedChange={(checked) => 
                      setProfile(prev => ({ 
                        ...prev, 
                        notifications: { ...prev.notifications, sms: checked }
                      }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Push Notifications</h4>
                    <p className="text-sm text-gray-600">Real-time updates from the mobile app</p>
                  </div>
                  <Switch 
                    checked={profile.notifications.push}
                    onCheckedChange={(checked) => 
                      setProfile(prev => ({ 
                        ...prev, 
                        notifications: { ...prev.notifications, push: checked }
                      }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div>
                    <h4 className="font-medium">Health Reminders</h4>
                    <p className="text-sm text-gray-600">Medication, exercise, and check-up reminders</p>
                  </div>
                  <Switch 
                    checked={profile.notifications.reminders}
                    onCheckedChange={(checked) => 
                      setProfile(prev => ({ 
                        ...prev, 
                        notifications: { ...prev.notifications, reminders: checked }
                      }))
                    }
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Assessment History</CardTitle>
              <CardDescription>
                View your past health risk assessments and track your progress
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {assessmentHistory.map((assessment, index) => (
                  <div key={index} className="p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">{assessment.date}</span>
                        <Badge variant="outline">{assessment.status}</Badge>
                      </div>
                      <Badge 
                        className={
                          assessment.overallRisk === "Low" ? "bg-green-50 text-green-700" :
                          assessment.overallRisk === "Low-Moderate" ? "bg-yellow-50 text-yellow-700" :
                          "bg-orange-50 text-orange-700"
                        }
                      >
                        {assessment.overallRisk} Risk
                      </Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm text-gray-600">
                      <div>Cardiovascular Risk: {assessment.cardiovascular}%</div>
                      <div>Diabetes Risk: {assessment.diabetes}%</div>
                    </div>
                    <Button variant="ghost" size="sm" className="mt-2 p-0 h-auto text-blue-600">
                      View Full Report →
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
