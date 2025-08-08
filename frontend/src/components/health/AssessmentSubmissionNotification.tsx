
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Clock, UserCheck, FileText, CheckCircle } from "lucide-react";

interface AssessmentSubmissionNotificationProps {
  expectedDeliveryTime: string;
  assignedSpecialist: string;
  specialistType: string;
}

export const AssessmentSubmissionNotification = ({ 
  expectedDeliveryTime, 
  assignedSpecialist, 
  specialistType 
}: AssessmentSubmissionNotificationProps) => {
  return (
    <Card className="bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
      <CardHeader>
        <div className="flex items-center space-x-2">
          <CheckCircle className="h-6 w-6 text-green-600" />
          <CardTitle className="text-green-800">Assessment Submitted Successfully!</CardTitle>
        </div>
        <CardDescription>
          Your health risk assessment has been received and is being processed
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert className="bg-blue-50 border-blue-200">
          <Clock className="h-4 w-4" />
          <AlertTitle className="text-blue-800">Report Delivery Timeline</AlertTitle>
          <AlertDescription className="text-blue-700">
            You will receive your comprehensive health report by <strong>{expectedDeliveryTime}</strong>
          </AlertDescription>
        </Alert>

        <div className="bg-white/70 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-gray-900">Assigned Specialist</h4>
            <Badge variant="secondary" className="bg-purple-100 text-purple-800">
              {specialistType}
            </Badge>
          </div>
          
          <div className="flex items-center space-x-3">
            <UserCheck className="h-5 w-5 text-purple-600" />
            <div>
              <p className="font-medium text-gray-900">Dr. {assignedSpecialist}</p>
              <p className="text-sm text-gray-600">Certified {specialistType}</p>
            </div>
          </div>
        </div>

        <div className="bg-white/70 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <FileText className="h-5 w-5 text-blue-600 mt-0.5" />
            <div className="space-y-2">
              <h4 className="font-semibold text-gray-900">What happens next?</h4>
              <ol className="text-sm text-gray-600 space-y-1 list-decimal list-inside">
                <li>Your assessment is being analyzed by our AI system</li>
                <li>Dr. {assignedSpecialist} will review your risk profile</li>
                <li>A comprehensive report with e-signature will be prepared</li>
                <li>You'll receive the final report via email by {expectedDeliveryTime}</li>
              </ol>
            </div>
          </div>
        </div>

        <Alert className="bg-amber-50 border-amber-200">
          <AlertTitle className="text-amber-800">Important Note</AlertTitle>
          <AlertDescription className="text-amber-700">
            Please check your email regularly. The report will include personalized recommendations 
            and may require follow-up actions based on your risk assessment.
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
};
