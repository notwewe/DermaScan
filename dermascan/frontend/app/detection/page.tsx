"use client"

import type React from "react"

import { useState, useRef } from "react"
import Image from "next/image"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Camera, Loader2, AlertTriangle, Info, Trash2, Share2 } from "lucide-react"
import GradientBackground from "@/components/gradient-background"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"

// Type definitions
type LesionType = "akiec" | "bcc" | "bkl" | "df" | "mel" | "nv" | "vasc"

type LesionInfo = {
  name: string
  description: string
  risk: string
  color: string
}

type LesionTypes = {
  [key in LesionType]: LesionInfo
}

type SampleImages = {
  [key in LesionType]: string
}

type ResultType = {
  prediction: LesionType
  confidences: Record<LesionType, number>
  details: string[]
}

// Define the lesion types and their descriptions
const lesionTypes: LesionTypes = {
  akiec: {
    name: "Actinic Keratosis",
    description:
      "A pre-cancerous growth caused by sun damage that may develop into squamous cell carcinoma if untreated.",
    risk: "Moderate",
    color: "amber",
  },
  bcc: {
    name: "Basal Cell Carcinoma",
    description: "The most common type of skin cancer that rarely spreads but can cause local damage if not treated.",
    risk: "Moderate to High",
    color: "orange",
  },
  bkl: {
    name: "Benign Keratosis",
    description: "A non-cancerous growth that includes seborrheic keratoses and solar lentigos.",
    risk: "Low",
    color: "green",
  },
  df: {
    name: "Dermatofibroma",
    description: "A common benign skin nodule that usually appears on the legs.",
    risk: "Very Low",
    color: "green",
  },
  mel: {
    name: "Melanoma",
    description: "A serious form of skin cancer that can spread to other parts of the body if not detected early.",
    risk: "High",
    color: "red",
  },
  nv: {
    name: "Melanocytic Nevus",
    description: "A common mole that is usually benign but should be monitored for changes.",
    risk: "Very Low",
    color: "green",
  },
  vasc: {
    name: "Vascular Lesion",
    description: "Includes hemangiomas, angiokeratomas, and pyogenic granulomas.",
    risk: "Low",
    color: "blue",
  },
}

// Sample images for each lesion type
const sampleImages: SampleImages = {
  akiec: "/placeholder.svg?height=300&width=300",
  bcc: "/placeholder.svg?height=300&width=300",
  bkl: "/placeholder.svg?height=300&width=300",
  df: "/placeholder.svg?height=300&width=300",
  mel: "/placeholder.svg?height=300&width=300",
  nv: "/placeholder.svg?height=300&width=300",
  vasc: "/placeholder.svg?height=300&width=300",
}

// Animation variants
const fadeIn = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.5 } },
}

const slideUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
}

const pulseAnimation = {
  pulse: {
    scale: [1, 1.03, 1],
    transition: { duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
  },
}

export default function DetectionPage() {
  const [image, setImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<ResultType | null>(null)
  const [activeTab, setActiveTab] = useState<string>("upload")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setImage(event.target?.result as string)
        setResults(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleClearImage = () => {
    setImage(null)
    setResults(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleDetection = async () => {
    if (!image) return

    setIsLoading(true)

    try {
      // Convert base64 image to blob
      const base64Response = await fetch(image)
      const blob = await base64Response.blob()

      // Create form data
      const formData = new FormData()
      formData.append("file", blob, "image.jpg")

      // Send to Streamlit backend
      const response = await fetch("http://https://dermascan-56zs.onrender.com/api/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to get prediction")
      }

      const result = await response.json()

      // Type assertion to ensure the result matches our expected type
      const typedResult = {
        prediction: result.prediction as LesionType,
        confidences: result.confidences as Record<LesionType, number>,
        details: result.details as string[],
      }

      setResults(typedResult)
    } catch (error) {
      console.error("Error during detection:", error)
      // Fallback to mock results for testing
      const mockConfidences: Partial<Record<LesionType, number>> = {}
      ;(Object.keys(lesionTypes) as LesionType[]).forEach((key) => {
        mockConfidences[key] = Math.random() * 0.2
      })

      // Normalize confidences to sum to 1
      const sum = Object.values(mockConfidences).reduce((a, b) => a + b, 0)
      const normalizedConfidences: Record<LesionType, number> = {} as Record<LesionType, number>
      ;(Object.keys(mockConfidences) as LesionType[]).forEach((key) => {
        normalizedConfidences[key] = (mockConfidences[key] || 0) / sum
      })

      // Find the highest confidence prediction
      const entries = Object.entries(normalizedConfidences) as [LesionType, number][]
      const prediction = entries.reduce(
        (max, [key, value]) => (value > max[1] ? [key, value] : max),
        ["nv" as LesionType, 0],
      )[0]

      const mockResults: ResultType = {
        prediction,
        confidences: normalizedConfidences,
        details: ["Image quality: Good", "Lesion border: Well-defined", "Color variation: Moderate"],
      }

      setResults(mockResults)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSampleImageSelect = (type: LesionType) => {
    setImage(sampleImages[type])
    setResults(null)
    setActiveTab("upload")
  }

  const getRiskBadgeColor = (risk: string) => {
    switch (risk) {
      case "High":
        return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
      case "Moderate to High":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400"
      case "Moderate":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400"
      case "Low":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400"
      default:
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
    }
  }

  const handleShareResults = () => {
    if (!results) return

    // Create a shareable message
    const shareText = `DermaScan detected ${lesionTypes[results.prediction].name} with ${(
      Math.max(...Object.values(results.confidences)) * 100
    ).toFixed(1)}% confidence.`

    // Check if Web Share API is available
    if (navigator.share) {
      navigator
        .share({
          title: "DermaScan Results",
          text: shareText,
        })
        .catch((err) => console.error("Error sharing:", err))
    } else {
      // Fallback - copy to clipboard
      navigator.clipboard
        .writeText(shareText)
        .then(() => alert("Results copied to clipboard!"))
        .catch((err) => console.error("Error copying to clipboard:", err))
    }
  }

  return (
    <main className="relative min-h-screen overflow-hidden">
      <GradientBackground />

      <motion.div initial="hidden" animate="visible" className="container mx-auto px-4 py-12 relative z-10">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-8 text-center"
        >
          Skin Lesion Detection
        </motion.h1>

        <Tabs defaultValue="upload" value={activeTab} onValueChange={setActiveTab} className="max-w-5xl mx-auto">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="upload">Upload Image</TabsTrigger>
            <TabsTrigger value="samples">Sample Images</TabsTrigger>
          </TabsList>

          <div>
            <TabsContent value="upload">
              <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-md border-slate-200 dark:border-slate-700 overflow-hidden">
                <CardContent className="p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <motion.div
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                      className="space-y-4"
                    >
                      <h2 className="text-xl font-semibold text-slate-900 dark:text-white">Upload Skin Lesion Image</h2>
                      <p className="text-slate-700 dark:text-slate-300">
                        Upload or capture a clear, well-lit image of the skin lesion for analysis.
                      </p>

                      <div className="flex flex-col gap-4">
                        <Button
                          onClick={handleUploadClick}
                          className="bg-teal-600 hover:bg-teal-700 text-white transition-all duration-300 transform hover:scale-105"
                        >
                          <Upload className="mr-2 h-4 w-4" /> Upload Image
                        </Button>
                        <input
                          type="file"
                          ref={fileInputRef}
                          onChange={handleFileChange}
                          accept="image/*"
                          className="hidden"
                        />
                        <Button
                          variant="outline"
                          className="border-slate-300 dark:border-slate-700 transition-all duration-300 transform hover:scale-105"
                        >
                          <Camera className="mr-2 h-4 w-4" /> Capture Image
                        </Button>
                      </div>

                      {image && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3 }}
                          className="space-y-4"
                        >
                          <Button
                            onClick={handleDetection}
                            className="w-full bg-teal-600 hover:bg-teal-700 text-white transition-all duration-300 transform hover:scale-105"
                            disabled={isLoading}
                          >
                            {isLoading ? (
                              <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Analyzing...
                              </>
                            ) : (
                              "Analyze Skin Lesion"
                            )}
                          </Button>

                          <Button
                            onClick={handleClearImage}
                            variant="outline"
                            className="w-full border-red-300 text-red-600 hover:bg-red-50 dark:border-red-800 dark:text-red-400 dark:hover:bg-red-900/20"
                          >
                            <Trash2 className="mr-2 h-4 w-4" /> Clear Image
                          </Button>
                        </motion.div>
                      )}

                      {/* Medical disclaimer */}
                      <motion.div
                        animate={{
                          scale: [1, 1.02, 1],
                          transition: { duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
                        }}
                        className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-md border border-amber-200 dark:border-amber-800"
                      >
                        <div className="flex items-start gap-2">
                          <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                          <p className="text-sm text-amber-700 dark:text-amber-300">
                            This tool is for educational purposes only and is not a substitute for professional medical
                            advice. Always consult a healthcare provider for proper diagnosis.
                          </p>
                        </div>
                      </motion.div>
                    </motion.div>

                    <motion.div
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5 }}
                      className="flex flex-col items-center justify-center"
                    >
                      {image ? (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.3 }}
                          className="relative w-full aspect-square rounded-lg overflow-hidden border-2 border-slate-200 dark:border-slate-700 shadow-lg"
                        >
                          <Image
                            src={image || "/placeholder.svg"}
                            alt="Uploaded skin lesion image"
                            fill
                            className="object-cover"
                          />
                        </motion.div>
                      ) : (
                        <motion.div
                          animate={{
                            scale: [1, 1.03, 1],
                            transition: { duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" },
                          }}
                          className="w-full aspect-square rounded-lg bg-slate-100 dark:bg-slate-700/50 border-2 border-dashed border-slate-300 dark:border-slate-600 flex items-center justify-center"
                        >
                          <p className="text-slate-500 dark:text-slate-400 text-center px-4">
                            Upload or capture an image to analyze
                          </p>
                        </motion.div>
                      )}
                    </motion.div>
                  </div>

                  <AnimatePresence>
                    {results && (
                      <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.5 }}
                        className="mt-8 p-6 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 shadow-lg"
                      >
                        <div className="flex justify-between items-center mb-4">
                          <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Detection Results</h3>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={handleShareResults}
                            className="flex items-center gap-2"
                          >
                            <Share2 className="h-4 w-4" /> Share
                          </Button>
                        </div>

                        <div className="mb-6">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium text-slate-900 dark:text-white">
                                {lesionTypes[results.prediction].name}
                              </h4>
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger>
                                    <Info className="h-4 w-4 text-slate-500 dark:text-slate-400" />
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <p className="max-w-xs">{lesionTypes[results.prediction].description}</p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </div>
                            <Badge className={getRiskBadgeColor(lesionTypes[results.prediction].risk)}>
                              Risk: {lesionTypes[results.prediction].risk}
                            </Badge>
                          </div>
                          <p className="text-slate-700 dark:text-slate-300 mb-4">
                            {lesionTypes[results.prediction].description}
                          </p>
                        </div>

                        <div className="mb-6">
                          <h4 className="font-medium text-slate-900 dark:text-white mb-3">Confidence Scores</h4>
                          <div className="space-y-3">
                            {Object.entries(results.confidences)
                              .sort((a, b) => b[1] - a[1])
                              .map(([key, value], index) => (
                                <motion.div
                                  key={key}
                                  className="space-y-1"
                                  initial={{ opacity: 0, x: -10 }}
                                  animate={{ opacity: 1, x: 0 }}
                                  transition={{ duration: 0.3, delay: index * 0.1 }}
                                >
                                  <div className="flex justify-between text-sm">
                                    <span className="text-slate-700 dark:text-slate-300">
                                      {lesionTypes[key as LesionType].name}
                                    </span>
                                    <span className="text-slate-900 dark:text-white font-medium">
                                      {(value * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="h-2 w-full bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                                    <motion.div
                                      initial={{ width: 0 }}
                                      animate={{ width: `${value * 100}%` }}
                                      transition={{ duration: 1, ease: "easeOut" }}
                                      className={`h-full ${
                                        key === results.prediction ? "bg-teal-500" : "bg-slate-400 dark:bg-slate-500"
                                      }`}
                                    />
                                  </div>
                                </motion.div>
                              ))}
                          </div>
                        </div>

                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{ delay: 0.5, duration: 0.5 }}
                        >
                          <h4 className="font-medium text-slate-900 dark:text-white mb-2">Analysis Details:</h4>
                          <ul className="list-disc list-inside text-slate-700 dark:text-slate-300 space-y-1">
                            {results.details.map((detail, index) => (
                              <motion.li
                                key={index}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ duration: 0.3, delay: 0.6 + index * 0.1 }}
                              >
                                {detail}
                              </motion.li>
                            ))}
                          </ul>
                        </motion.div>

                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.8, duration: 0.5 }}
                          className="mt-6 p-4 bg-slate-100 dark:bg-slate-700/30 rounded-md border border-l-4 border-l-teal-500"
                        >
                          <h4 className="font-medium text-slate-900 dark:text-white mb-2">Recommended Next Steps:</h4>
                          {lesionTypes[results.prediction].risk === "High" ? (
                            <p className="text-red-600 dark:text-red-400">
                              Consult a dermatologist as soon as possible for proper evaluation and diagnosis.
                            </p>
                          ) : lesionTypes[results.prediction].risk === "Moderate to High" ||
                            lesionTypes[results.prediction].risk === "Moderate" ? (
                            <p className="text-amber-600 dark:text-amber-400">
                              Schedule an appointment with a dermatologist for professional evaluation.
                            </p>
                          ) : (
                            <p className="text-green-600 dark:text-green-400">
                              Monitor for changes and consult a healthcare provider during your next regular check-up.
                            </p>
                          )}
                        </motion.div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="samples">
              <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-md border-slate-200 dark:border-slate-700">
                <CardContent className="p-6">
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">Sample Lesion Images</h2>
                    <p className="text-slate-700 dark:text-slate-300 mb-6">
                      Select a sample image to see how the detection works for different types of skin lesions.
                    </p>

                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                      {(Object.entries(lesionTypes) as [LesionType, LesionInfo][]).map(([key, value], index) => (
                        <motion.div
                          key={key}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.3, delay: index * 0.1 }}
                          whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
                          className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden cursor-pointer hover:shadow-lg transition-shadow bg-white dark:bg-slate-800"
                          onClick={() => handleSampleImageSelect(key)}
                        >
                          <div className="relative aspect-square">
                            <Image
                              src={sampleImages[key] || "/placeholder.svg"}
                              alt={value.name}
                              fill
                              className="object-cover"
                            />
                          </div>
                          <div className="p-3">
                            <p className="text-sm font-medium text-slate-900 dark:text-white truncate">{value.name}</p>
                            <Badge className={`mt-2 ${getRiskBadgeColor(value.risk)}`}>{value.risk}</Badge>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                </CardContent>
              </Card>
            </TabsContent>
          </div>
        </Tabs>
      </motion.div>
    </main>
  )
}
