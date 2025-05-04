"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowRight, BookOpen, FileText, Database } from "lucide-react"
import GradientBackground from "@/components/gradient-background"
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

export default function AboutPage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <GradientBackground />

      <div className="container mx-auto px-4 py-12 relative z-10">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white mb-8 text-center"
        >
          About DermaScan
        </motion.h1>

        <motion.div
          initial="hidden"
          animate="visible"
          variants={fadeIn}
          className="max-w-4xl mx-auto bg-white/90 dark:bg-slate-800/90 backdrop-blur-md rounded-xl p-8 border border-slate-200 dark:border-slate-700"
        >
          <motion.section variants={slideUp} className="mb-12">
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Project Overview</h2>
            <p className="text-slate-700 dark:text-slate-300 mb-4">
              DermaScan is an AI-powered web application designed to detect and classify skin lesions using advanced
              deep learning techniques. Our project aims to provide an accessible tool for preliminary skin lesion
              assessment, helping users identify potential concerns that may require professional medical attention.
            </p>
            <p className="text-slate-700 dark:text-slate-300">
              The application uses a convolutional neural network trained on the HAM10000 dataset to analyze
              dermatoscopic images and provide detailed classifications across seven different types of skin lesions.
              This technology can help with early detection of potentially serious skin conditions, though it should
              always be used in conjunction with professional medical advice.
            </p>
          </motion.section>

          <motion.section variants={slideUp} className="mb-12">
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">The Dataset</h2>
            <div className="flex items-start gap-4 mb-6">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.5 }}
              >
                <Database className="h-8 w-8 text-teal-500 flex-shrink-0 mt-1" />
              </motion.div>
              <div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">HAM10000 Dataset</h3>
                <p className="text-slate-700 dark:text-slate-300">
                  Our model is trained on the Human Against Machine with 10000 training images (HAM10000) dataset, a
                  large collection of dermatoscopic images of common pigmented skin lesions. This dataset includes over
                  10,000 high-quality images across seven different diagnostic categories:
                </p>
                <motion.ul
                  initial="hidden"
                  animate="visible"
                  variants={staggerContainer}
                  className="list-disc list-inside text-slate-700 dark:text-slate-300 mt-4 space-y-2"
                >
                  {[
                    "Actinic Keratosis (akiec)",
                    "Basal Cell Carcinoma (bcc)",
                    "Benign Keratosis (bkl)",
                    "Dermatofibroma (df)",
                    "Melanoma (mel)",
                    "Melanocytic Nevus (nv)",
                    "Vascular Lesion (vasc)",
                  ].map((item, index) => (
                    <motion.li
                      key={index}
                      variants={{
                        hidden: { opacity: 0, x: -20 },
                        visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
                      }}
                    >
                      {item}
                    </motion.li>
                  ))}
                </motion.ul>
                <p className="text-slate-700 dark:text-slate-300 mt-4">
                  The dataset was collected over 20 years and contains images from different populations, captured with
                  different devices. This diversity helps our model generalize well to various skin types and imaging
                  conditions.
                </p>
              </div>
            </div>
          </motion.section>

          <motion.section variants={slideUp} className="mb-12">
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Technology Stack</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
                className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-lg hover:shadow-md transition-all duration-300"
              >
                <h3 className="text-xl font-semibold text-teal-600 dark:text-teal-400 mb-2">Frontend</h3>
                <ul className="list-disc list-inside text-slate-700 dark:text-slate-300 space-y-1">
                  <li>Next.js for server-side rendering and routing</li>
                  <li>Tailwind CSS for responsive design</li>
                  <li>React for interactive UI components</li>
                  <li>Shadcn/UI for accessible component library</li>
                </ul>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
                className="bg-slate-50 dark:bg-slate-700/50 p-4 rounded-lg hover:shadow-md transition-all duration-300"
              >
                <h3 className="text-xl font-semibold text-teal-600 dark:text-teal-400 mb-2">AI Model</h3>
                <ul className="list-disc list-inside text-slate-700 dark:text-slate-300 space-y-1">
                  <li>EfficientNet architecture</li>
                  <li>PyTorch for model development</li>
                  <li>Transfer learning with ImageNet weights</li>
                  <li>Streamlit for model deployment</li>
                </ul>
              </motion.div>
            </div>
          </motion.section>

          <motion.section variants={slideUp} className="mb-12">
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">The AI Model</h2>
            <p className="text-slate-700 dark:text-slate-300 mb-4">
              Our model uses the EfficientNet-B3 architecture, which provides an excellent balance between accuracy and
              computational efficiency. The model was trained using the following approach:
            </p>
            <motion.ul
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
              className="list-disc list-inside text-slate-700 dark:text-slate-300 space-y-1 mb-4"
            >
              {[
                "Transfer learning from ImageNet pre-trained weights",
                "Data augmentation to improve generalization (rotation, flipping, color jittering)",
                "Class balancing to handle the imbalanced nature of the HAM10000 dataset",
                "Fine-tuning with a learning rate scheduler for optimal convergence",
              ].map((item, index) => (
                <motion.li
                  key={index}
                  variants={{
                    hidden: { opacity: 0, x: -20 },
                    visible: { opacity: 1, x: 0, transition: { duration: 0.3 } },
                  }}
                >
                  {item}
                </motion.li>
              ))}
            </motion.ul>
            <p className="text-slate-700 dark:text-slate-300">
              The model achieves over 85% accuracy on the test set, with particularly high sensitivity for melanoma
              detection, which is critical for early intervention in potentially life-threatening cases.
            </p>
          </motion.section>

          <motion.section variants={slideUp} className="mb-12">
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Medical Disclaimer</h2>
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5, duration: 0.5 }}
              className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-md border border-amber-200 dark:border-amber-800"
            >
              <p className="text-amber-700 dark:text-amber-300">
                DermaScan is designed as an educational and assistive tool only. It is not intended to replace
                professional medical diagnosis, advice, or treatment. The predictions made by this application should be
                verified by qualified healthcare professionals. Early detection and proper medical care are essential
                for skin conditions, particularly for potentially serious conditions like melanoma.
              </p>
            </motion.div>
          </motion.section>

          <motion.section variants={slideUp}>
            <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4">Resources</h2>
            <div className="flex flex-col gap-4">
              <motion.a
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6, duration: 0.3 }}
                href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-teal-600 dark:text-teal-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
                whileHover={{ x: 5, transition: { duration: 0.2 } }}
              >
                <Database size={20} />
                HAM10000 Dataset (Harvard Dataverse)
              </motion.a>
              <motion.a
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7, duration: 0.3 }}
                href="https://www.nature.com/articles/sdata2018161"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-teal-600 dark:text-teal-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
                whileHover={{ x: 5, transition: { duration: 0.2 } }}
              >
                <FileText size={20} />
                HAM10000 Dataset Paper (Scientific Data)
              </motion.a>
              <motion.a
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.3 }}
                href="https://arxiv.org/abs/1905.11946"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-teal-600 dark:text-teal-400 hover:text-teal-700 dark:hover:text-teal-300 transition-colors"
                whileHover={{ x: 5, transition: { duration: 0.2 } }}
              >
                <BookOpen size={20} />
                EfficientNet Paper (arXiv)
              </motion.a>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9, duration: 0.3 }}
                className="mt-4"
              >
                <Button className="bg-teal-600 hover:bg-teal-700 text-white transition-all duration-300 transform hover:scale-105">
                  <Link href="/detection" className="flex items-center gap-2">
                    Try Skin Lesion Detection <ArrowRight size={16} />
                  </Link>
                </Button>
              </motion.div>
            </div>
          </motion.section>
        </motion.div>
      </div>
    </main>
  )
}
