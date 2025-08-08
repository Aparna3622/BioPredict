
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BookOpen, Video, FileText, Search, Clock, Star, ArrowRight, Heart, Brain, Activity, Apple } from "lucide-react";
import { useState } from "react";

export const EducationalResources = () => {
  const [searchTerm, setSearchTerm] = useState("");

  const articles = [
    {
      id: 1,
      title: "Understanding Cardiovascular Risk Factors",
      category: "Heart Health",
      readTime: "8 min read",
      difficulty: "Beginner",
      rating: 4.8,
      summary: "Learn about the key factors that contribute to heart disease and how to manage them effectively.",
      tags: ["Prevention", "Heart Disease", "Lifestyle"],
      icon: Heart,
      color: "text-red-600"
    },
    {
      id: 2,
      title: "Diabetes Prevention Through Diet and Exercise",
      category: "Diabetes",
      readTime: "12 min read",
      difficulty: "Intermediate",
      rating: 4.9,
      summary: "Evidence-based strategies for preventing type 2 diabetes through lifestyle modifications.",
      tags: ["Diabetes", "Diet", "Exercise"],
      icon: Activity,
      color: "text-orange-600"
    },
    {
      id: 3,
      title: "Mental Health and Physical Wellness Connection",
      category: "Mental Health",
      readTime: "10 min read",
      difficulty: "Beginner",
      rating: 4.7,
      summary: "Explore the important relationship between mental health and overall physical wellbeing.",
      tags: ["Mental Health", "Wellness", "Stress"],
      icon: Brain,
      color: "text-purple-600"
    },
    {
      id: 4,
      title: "Nutrition Fundamentals for Disease Prevention",
      category: "Nutrition",
      readTime: "15 min read",
      difficulty: "Intermediate",
      rating: 4.6,
      summary: "Comprehensive guide to using nutrition as a tool for preventing chronic diseases.",
      tags: ["Nutrition", "Prevention", "Diet"],
      icon: Apple,
      color: "text-green-600"
    }
  ];

  const videos = [
    {
      id: 1,
      title: "5-Minute Daily Exercises for Heart Health",
      duration: "5:32",
      views: "125K",
      category: "Exercise",
      thumbnail: "/placeholder.svg"
    },
    {
      id: 2,
      title: "Understanding Your Health Risk Assessment",
      duration: "8:45",
      views: "89K",
      category: "Education",
      thumbnail: "/placeholder.svg"
    },
    {
      id: 3,
      title: "Stress Management Techniques",
      duration: "12:15",
      views: "203K",
      category: "Mental Health",
      thumbnail: "/placeholder.svg"
    }
  ];

  const tools = [
    {
      id: 1,
      name: "BMI Calculator",
      description: "Calculate your Body Mass Index and understand what it means for your health",
      category: "Assessment",
      icon: Activity
    },
    {
      id: 2,
      name: "Calorie Tracker",
      description: "Track your daily caloric intake and nutritional information",
      category: "Nutrition",
      icon: Apple
    },
    {
      id: 3,
      name: "Risk Factor Checklist",
      description: "Interactive checklist to identify your personal health risk factors",
      category: "Assessment",
      icon: FileText
    }
  ];

  const filteredArticles = articles.filter(article =>
    article.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    article.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
    article.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <Card className="bg-gradient-to-r from-indigo-50 to-purple-50 border-indigo-200">
        <CardHeader>
          <CardTitle className="text-2xl bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Health Education Center
          </CardTitle>
          <CardDescription>
            Comprehensive resources to help you understand and improve your health
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <Input
              placeholder="Search articles, videos, or topics..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="articles" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 bg-white/50 backdrop-blur-sm">
          <TabsTrigger value="articles" className="flex items-center space-x-2">
            <BookOpen className="h-4 w-4" />
            <span>Articles</span>
          </TabsTrigger>
          <TabsTrigger value="videos" className="flex items-center space-x-2">
            <Video className="h-4 w-4" />
            <span>Videos</span>
          </TabsTrigger>
          <TabsTrigger value="tools" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Tools</span>
          </TabsTrigger>
          <TabsTrigger value="guides" className="flex items-center space-x-2">
            <FileText className="h-4 w-4" />
            <span>Guides</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="articles" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {filteredArticles.map((article) => (
              <Card key={article.id} className="hover:shadow-lg transition-all duration-300 bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-2">
                      <article.icon className={`h-5 w-5 ${article.color}`} />
                      <Badge variant="outline" className="text-xs">
                        {article.category}
                      </Badge>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Star className="h-3 w-3 text-yellow-400 fill-current" />
                      <span className="text-xs text-gray-600">{article.rating}</span>
                    </div>
                  </div>
                  <CardTitle className="text-lg leading-tight">{article.title}</CardTitle>
                  <CardDescription className="text-sm">{article.summary}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-1 mb-4">
                    {article.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4 text-xs text-gray-600">
                      <div className="flex items-center space-x-1">
                        <Clock className="h-3 w-3" />
                        <span>{article.readTime}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {article.difficulty}
                      </Badge>
                    </div>
                    <Button size="sm" variant="ghost" className="text-blue-600 hover:text-blue-700">
                      Read More
                      <ArrowRight className="h-3 w-3 ml-1" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="videos" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {videos.map((video) => (
              <Card key={video.id} className="hover:shadow-lg transition-all duration-300 bg-white/70 backdrop-blur-sm">
                <div className="aspect-video bg-gradient-to-br from-blue-100 to-purple-100 rounded-t-lg flex items-center justify-center">
                  <Video className="h-12 w-12 text-blue-600" />
                </div>
                <CardHeader className="pb-2">
                  <CardTitle className="text-base leading-tight">{video.title}</CardTitle>
                  <div className="flex items-center justify-between text-xs text-gray-600">
                    <Badge variant="outline">{video.category}</Badge>
                    <span>{video.views} views</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">{video.duration}</span>
                    <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                      Watch Now
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="tools" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {tools.map((tool) => (
              <Card key={tool.id} className="hover:shadow-lg transition-all duration-300 bg-white/70 backdrop-blur-sm">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <tool.icon className="h-6 w-6 text-blue-600" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{tool.name}</CardTitle>
                      <Badge variant="outline" className="text-xs mt-1">
                        {tool.category}
                      </Badge>
                    </div>
                  </div>
                  <CardDescription>{tool.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button className="w-full">
                    Use Tool
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="guides" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="bg-gradient-to-br from-red-50 to-pink-50 border-red-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-red-700">
                  <Heart className="h-5 w-5" />
                  <span>Heart Health Guide</span>
                </CardTitle>
                <CardDescription>
                  Complete guide to maintaining cardiovascular health
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Understanding heart disease risk factors</li>
                  <li>• Heart-healthy diet recommendations</li>
                  <li>• Exercise guidelines for heart health</li>
                  <li>• When to see a cardiologist</li>
                </ul>
                <Button className="w-full mt-4 bg-red-600 hover:bg-red-700">
                  Download Guide
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-green-50 to-emerald-50 border-green-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-green-700">
                  <Apple className="h-5 w-5" />
                  <span>Nutrition Guide</span>
                </CardTitle>
                <CardDescription>
                  Evidence-based nutrition for disease prevention
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Macronutrient balance recommendations</li>
                  <li>• Foods that fight inflammation</li>
                  <li>• Meal planning strategies</li>
                  <li>• Supplements and vitamins</li>
                </ul>
                <Button className="w-full mt-4 bg-green-600 hover:bg-green-700">
                  Download Guide
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-purple-50 to-indigo-50 border-purple-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-purple-700">
                  <Brain className="h-5 w-5" />
                  <span>Mental Health Guide</span>
                </CardTitle>
                <CardDescription>
                  Strategies for mental wellness and stress management
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Stress reduction techniques</li>
                  <li>• Building emotional resilience</li>
                  <li>• Sleep hygiene practices</li>
                  <li>• When to seek professional help</li>
                </ul>
                <Button className="w-full mt-4 bg-purple-600 hover:bg-purple-700">
                  Download Guide
                </Button>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-blue-50 to-cyan-50 border-blue-200">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-blue-700">
                  <Activity className="h-5 w-5" />
                  <span>Exercise Guide</span>
                </CardTitle>
                <CardDescription>
                  Safe and effective exercise for all fitness levels
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-gray-700">
                  <li>• Getting started with exercise</li>
                  <li>• Strength training basics</li>
                  <li>• Cardio workout recommendations</li>
                  <li>• Exercise safety and injury prevention</li>
                </ul>
                <Button className="w-full mt-4 bg-blue-600 hover:bg-blue-700">
                  Download Guide
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
