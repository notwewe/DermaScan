"use client"

import { useEffect, useRef } from "react"
import Image from "next/image"
import { motion } from "framer-motion"

export default function AnimatedCell() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Gentle pulsing animation
    const animate = () => {
      const time = Date.now() * 0.001

      // Apply gentle pulsing movement
      container.style.transform = `
        scale(${1 + Math.sin(time) * 0.02})
        rotate(${Math.sin(time * 0.5) * 2}deg)
      `

      requestAnimationFrame(animate)
    }

    const animationId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(animationId)
    }
  }, [])

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className="relative w-full aspect-square max-w-md mx-auto transition-transform duration-300 ease-in-out"
    >
      <div className="relative w-full h-full">
        <Image
          src="/bg1.png"
          alt="Skin cell visualization"
          width={500}
          height={500}
          className="rounded-lg shadow-lg"
          priority
        />

        {/* Overlay glow effect */}
        <motion.div
          className="absolute inset-0 bg-teal-500 opacity-10 rounded-lg blur-md"
          animate={{
            opacity: [0.1, 0.2, 0.1],
            scale: [1, 1.05, 1],
          }}
          transition={{
            duration: 4,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
        />

        {/* Animated particles */}
        <div className="absolute inset-0 overflow-hidden rounded-lg">
          {Array.from({ length: 15 }).map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-teal-500 rounded-full opacity-60"
              initial={{
                top: `${Math.random() * 100}%`,
                left: `${Math.random() * 100}%`,
                opacity: 0.2,
                scale: 0.8,
              }}
              animate={{
                opacity: [0.2, 0.6, 0.2],
                scale: [0.8, 1.2, 0.8],
              }}
              transition={{
                duration: 3 + Math.random() * 4,
                repeat: Number.POSITIVE_INFINITY,
                delay: Math.random() * 5,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
}
