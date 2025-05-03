"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export default function ModelLoading() {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState("Initializing...")

  useEffect(() => {
    const statuses = [
      "Loading model architecture...",
      "Loading pre-trained weights...",
      "Initializing EfficientNet-B3...",
      "Setting up classification layers...",
      "Preparing inference pipeline...",
      "Loading normalization parameters...",
      "Optimizing for inference...",
      "Model ready!",
    ]

    let currentStep = 0
    const totalSteps = statuses.length

    const interval = setInterval(() => {
      if (currentStep < totalSteps) {
        setStatus(statuses[currentStep])
        setProgress(Math.min(100, (currentStep + 1) * (100 / totalSteps)))
        currentStep++
      } else {
        clearInterval(interval)
      }
    }, 800)

    return () => clearInterval(interval)
  }, [])

  return (
    <Card className="bg-white/90 dark:bg-slate-800/90 backdrop-blur-md border-slate-200 dark:border-slate-700">
      <CardContent className="p-6">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">Loading Model</h2>
        <div className="space-y-4">
          <Progress value={progress} className="h-2" />
          <p className="text-slate-700 dark:text-slate-300">{status}</p>
        </div>
      </CardContent>
    </Card>
  )
}
