"use client"

import { useEffect, useRef } from "react"
import { motion } from "framer-motion"

export default function GradientBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const handleResize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    window.addEventListener("resize", handleResize)
    handleResize()

    // Create gradient
    let gradientAngle = 0
    const gradientSpeed = 0.0005

    // Animation loop
    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Update gradient angle
      gradientAngle += gradientSpeed
      if (gradientAngle >= Math.PI * 2) {
        gradientAngle = 0
      }

      // Calculate gradient points
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const radius = Math.max(canvas.width, canvas.height)

      const x1 = centerX + Math.cos(gradientAngle) * radius
      const y1 = centerY + Math.sin(gradientAngle) * radius
      const x2 = centerX + Math.cos(gradientAngle + Math.PI) * radius
      const y2 = centerY + Math.sin(gradientAngle + Math.PI) * radius

      // Create gradient
      const gradient = ctx.createLinearGradient(x1, y1, x2, y2)

      // Medical theme colors
      gradient.addColorStop(0, "rgba(240, 253, 250, 0.8)") // Light teal
      gradient.addColorStop(0.3, "rgba(204, 251, 241, 0.7)") // Lighter teal
      gradient.addColorStop(0.6, "rgba(153, 246, 228, 0.6)") // Teal
      gradient.addColorStop(1, "rgba(94, 234, 212, 0.5)") // Darker teal

      // Fill background
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // Add subtle pattern
      ctx.fillStyle = "rgba(255, 255, 255, 0.03)"

      const patternSize = 20
      const patternOffset = (performance.now() * 0.05) % (patternSize * 2)

      for (let x = -patternSize + patternOffset; x < canvas.width; x += patternSize * 2) {
        for (let y = -patternSize + patternOffset; y < canvas.height; y += patternSize * 2) {
          ctx.beginPath()
          ctx.arc(x, y, patternSize / 4, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => {
      window.removeEventListener("resize", handleResize)
      cancelAnimationFrame(animationId)
    }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="fixed inset-0 -z-10"
    >
      <canvas ref={canvasRef} className="fixed inset-0 -z-10" />
    </motion.div>
  )
}
