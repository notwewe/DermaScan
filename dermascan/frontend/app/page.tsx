"use client"

import type React from "react"

import Link from "next/link"
import { ArrowRight, Shield, Microscope, FileCheck, Database } from "lucide-react"
import { Button } from "@/components/ui/button"
import GradientBackground from "@/components/gradient-background"
import AnimatedCell from "@/components/animated-cell"
import { motion } from "framer-motion"

// Animation variants
const fadeIn = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.6 } },
}

const slideUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
}

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.2,
    },
  },
}

const cardVariant = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
}

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Medical gradient background */}
      <GradientBackground />

      <div className="container mx-auto px-4 py-12 relative z-10">
        {/* Hero Section */}
        <motion.section
          initial="hidden"
          animate="visible"
          variants={fadeIn}
          className="flex flex-col md:flex-row items-center justify-between py-12 gap-8"
        >
          <motion.div variants={slideUp} className="md:w-1/2 space-y-6">
            <h1 className="text-4xl md:text-6xl font-bold text-slate-900 dark:text-white">
              <span className="text-teal-600 dark:text-teal-400">Derma</span>Scan
            </h1>
            <h2 className="text-2xl md:text-3xl font-semibold text-slate-800 dark:text-slate-200">
              AI-Powered Skin Lesion Detection
            </h2>
            <p className="text-lg text-slate-700 dark:text-slate-300">
              Using advanced deep learning to help identify and classify skin lesions with medical-grade accuracy.
            </p>
            <motion.div
              className="flex flex-wrap gap-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.5 }}
            >
              <Button
                size="lg"
                className="bg-teal-600 hover:bg-teal-700 text-white transition-all duration-300 transform hover:scale-105"
              >
                <Link href="/detection" className="flex items-center gap-2">
                  Start Detection <ArrowRight size={16} />
                </Link>
              </Button>
              <Button
                variant="outline"
                size="lg"
                className="border-slate-300 dark:border-slate-700 transition-all duration-300 transform hover:scale-105"
              >
                <Link href="/about" className="flex items-center gap-2">
                  Learn More
                </Link>
              </Button>
            </motion.div>
          </motion.div>
          <motion.div variants={slideUp} className="md:w-1/2 relative">
            <AnimatedCell />
          </motion.div>
        </motion.section>

        {/* Features Section */}
        <motion.section
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.3 }}
          variants={staggerContainer}
          className="py-16"
        >
          <motion.h2 variants={slideUp} className="text-3xl font-bold text-center text-slate-900 dark:text-white mb-12">
            Key Features
          </motion.h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Shield className="h-10 w-10 text-teal-500" />}
              title="Medical-Grade Accuracy"
              description="Trained on the HAM10000 dataset with over 10,000 dermatoscopic images of skin lesions."
            />
            <FeatureCard
              icon={<Microscope className="h-10 w-10 text-teal-500" />}
              title="Multi-Class Classification"
              description="Detects 7 different types of skin lesions including melanoma, basal cell carcinoma, and more."
            />
            <FeatureCard
              icon={<FileCheck className="h-10 w-10 text-teal-500" />}
              title="Instant Analysis"
              description="Get immediate results with detailed information about the detected skin condition."
            />
            <FeatureCard
              icon={<Database className="h-10 w-10 text-teal-500" />}
              title="Research-Backed"
              description="Built on peer-reviewed research and validated against dermatologist diagnoses."
            />
          </div>
        </motion.section>

        {/* How It Works Section */}
        <motion.section
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.3 }}
          variants={staggerContainer}
          className="py-16"
        >
          <motion.h2 variants={slideUp} className="text-3xl font-bold text-center text-slate-900 dark:text-white mb-12">
            How It Works
          </motion.h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <ProcessCard
              step="1"
              title="Upload Image"
              description="Take a photo of the skin lesion or upload an existing image."
            />
            <ProcessCard
              step="2"
              title="AI Analysis"
              description="Our deep learning model analyzes the image using techniques validated in medical research."
            />
            <ProcessCard
              step="3"
              title="View Results"
              description="Receive a detailed classification with confidence scores and recommended next steps."
            />
          </div>
        </motion.section>

        {/* Disclaimer Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="py-8 px-6 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800"
        >
          <h3 className="text-xl font-semibold text-amber-800 dark:text-amber-400 mb-2">Medical Disclaimer</h3>
          <p className="text-amber-700 dark:text-amber-300">
            DermaScan is designed as an assistive tool and should not replace professional medical advice. Always
            consult with a qualified healthcare provider for proper diagnosis and treatment of skin conditions.
          </p>
        </motion.section>
      </div>
    </main>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <motion.div
      variants={cardVariant}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-xl p-6 hover:shadow-lg transition-all duration-300 border border-slate-200 dark:border-slate-700"
    >
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
      <p className="text-slate-700 dark:text-slate-300">{description}</p>
    </motion.div>
  )
}

function ProcessCard({ step, title, description }: { step: string; title: string; description: string }) {
  return (
    <motion.div
      variants={cardVariant}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-xl p-6 hover:shadow-lg transition-all duration-300 border border-slate-200 dark:border-slate-700 text-center"
    >
      <motion.div
        className="w-12 h-12 rounded-full bg-teal-500 flex items-center justify-center mx-auto mb-4"
        whileHover={{ scale: 1.1, rotate: 5 }}
        transition={{ type: "spring", stiffness: 300 }}
      >
        <span className="text-white font-bold">{step}</span>
      </motion.div>
      <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">{title}</h3>
      <p className="text-slate-700 dark:text-slate-300">{description}</p>
    </motion.div>
  )
}
