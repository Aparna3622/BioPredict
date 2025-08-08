
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useToast } from "@/hooks/use-toast";
import { Calendar as CalendarIcon, Clock, MapPin, User, Star, Phone, Video } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

export const AppointmentScheduler = () => {
  const [selectedDate, setSelectedDate] = useState<Date>();
  const [selectedDoctor, setSelectedDoctor] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [appointmentType, setAppointmentType] = useState("");
  const [notes, setNotes] = useState("");
  const { toast } = useToast();

  const doctors = [
    {
      id: "1",
      name: "Dr. Sarah Mitchell",
      specialty: "Cardiologist",
      rating: 4.9,
      availability: "Available Today",
      location: "Heart Center",
      experience: "15 years",
      image: "/placeholder.svg"
    },
    {
      id: "2",
      name: "Dr. James Rodriguez",
      specialty: "Internal Medicine",
      rating: 4.8,
      availability: "Next Available: Tomorrow",
      location: "Main Clinic",
      experience: "12 years",
      image: "/placeholder.svg"
    },
    {
      id: "3",
      name: "Dr. Emily Chen",
      specialty: "Endocrinologist",
      rating: 4.9,
      availability: "Available This Week",
      location: "Diabetes Care Center",
      experience: "10 years",
      image: "/placeholder.svg"
    },
    {
      id: "4",
      name: "Dr. Michael Thompson",
      specialty: "Oncologist",
      rating: 4.7,
      availability: "Next Available: Next Week",
      location: "Cancer Care Center",
      experience: "18 years",
      image: "/placeholder.svg"
    }
  ];

  const timeSlots = [
    "09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM",
    "11:00 AM", "11:30 AM", "02:00 PM", "02:30 PM",
    "03:00 PM", "03:30 PM", "04:00 PM", "04:30 PM"
  ];

  const handleBookAppointment = () => {
    if (!selectedDate || !selectedDoctor || !selectedTime || !appointmentType) {
      toast({
        title: "Missing Information",
        description: "Please fill in all required fields to book your appointment.",
        variant: "destructive"
      });
      return;
    }

    const doctor = doctors.find(d => d.id === selectedDoctor);
    toast({
      title: "Appointment Booked!",
      description: `Your appointment with ${doctor?.name} has been scheduled for ${format(selectedDate, "PPP")} at ${selectedTime}.`,
    });
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Schedule Your Health Consultation
          </CardTitle>
          <CardDescription>
            Book appointments with specialists based on your health risk assessment
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Doctor Selection */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <User className="h-5 w-5 text-blue-600" />
                <span>Available Specialists</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {doctors.map((doctor) => (
                <div
                  key={doctor.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-all duration-300 hover:shadow-md ${
                    selectedDoctor === doctor.id 
                      ? "border-blue-500 bg-blue-50" 
                      : "border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => setSelectedDoctor(doctor.id)}
                >
                  <div className="flex items-start space-x-4">
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
                      <User className="h-8 w-8 text-blue-600" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-gray-900">{doctor.name}</h3>
                        <div className="flex items-center space-x-1">
                          <Star className="h-4 w-4 text-yellow-400 fill-current" />
                          <span className="text-sm font-medium">{doctor.rating}</span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-gray-600">{doctor.specialty} • {doctor.experience}</p>
                        <p className="text-sm text-gray-500 flex items-center space-x-1">
                          <MapPin className="h-3 w-3" />
                          <span>{doctor.location}</span>
                        </p>
                        <Badge 
                          variant={doctor.availability.includes("Today") ? "default" : "secondary"}
                          className="text-xs"
                        >
                          {doctor.availability}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Appointment Details */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <CalendarIcon className="h-5 w-5 text-green-600" />
                <span>Book Appointment</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Date Selection */}
              <div className="space-y-2">
                <Label>Select Date</Label>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn(
                        "w-full justify-start text-left font-normal",
                        !selectedDate && "text-muted-foreground"
                      )}
                    >
                      <CalendarIcon className="mr-2 h-4 w-4" />
                      {selectedDate ? format(selectedDate, "PPP") : "Pick a date"}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      selected={selectedDate}
                      onSelect={setSelectedDate}
                      disabled={(date) => date < new Date()}
                      initialFocus
                      className="pointer-events-auto"
                    />
                  </PopoverContent>
                </Popover>
              </div>

              {/* Time Selection */}
              <div className="space-y-2">
                <Label>Select Time</Label>
                <Select value={selectedTime} onValueChange={setSelectedTime}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose time slot" />
                  </SelectTrigger>
                  <SelectContent>
                    {timeSlots.map((time) => (
                      <SelectItem key={time} value={time}>
                        {time}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Appointment Type */}
              <div className="space-y-2">
                <Label>Appointment Type</Label>
                <Select value={appointmentType} onValueChange={setAppointmentType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select appointment type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="consultation">
                      <div className="flex items-center space-x-2">
                        <Phone className="h-4 w-4" />
                        <span>Phone Consultation</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="video">
                      <div className="flex items-center space-x-2">
                        <Video className="h-4 w-4" />
                        <span>Video Call</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="in-person">
                      <div className="flex items-center space-x-2">
                        <MapPin className="h-4 w-4" />
                        <span>In-Person Visit</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Additional Notes */}
              <div className="space-y-2">
                <Label>Additional Notes (Optional)</Label>
                <Textarea
                  placeholder="Any specific concerns or symptoms you'd like to discuss..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={3}
                />
              </div>

              <Button 
                onClick={handleBookAppointment} 
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
              >
                Book Appointment
              </Button>
            </CardContent>
          </Card>

          {/* Upcoming Appointments */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Upcoming Appointments</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm">Dr. Sarah Mitchell</span>
                    <Badge variant="outline" className="text-xs">In 3 days</Badge>
                  </div>
                  <p className="text-xs text-gray-600">June 27, 2025 • 2:00 PM</p>
                  <p className="text-xs text-gray-500">Cardiology Follow-up</p>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm">Annual Check-up</span>
                    <Badge variant="outline" className="text-xs">Next month</Badge>
                  </div>
                  <p className="text-xs text-gray-600">July 15, 2025 • 10:00 AM</p>
                  <p className="text-xs text-gray-500">General Medicine</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
