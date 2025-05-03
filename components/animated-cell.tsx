"use client"

import { useEffect, useRef } from "react"
import Image from "next/image"

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
    <div
      ref={containerRef}
      className="relative w-full aspect-square max-w-md mx-auto transition-transform duration-300 ease-in-out"
    >
      <div className="relative w-full h-full">
        <Image
          src="/placeholder.svg?height=500&width=500"
          alt="Skin cell visualization"
          width={500}
          height={500}
          className="rounded-lg shadow-lg"
          priority
        />

        {/* Overlay glow effect */}
        <div className="absolute inset-0 bg-teal-500 opacity-10 rounded-lg blur-md" />

        {/* Animated particles */}
        <div className="absolute inset-0 overflow-hidden rounded-lg">
          {Array.from({ length: 15 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-teal-500 rounded-full opacity-60"
              style={{
                top: `${Math.random() * 100}%`,
                left: `${Math.random() * 100}%`,
                animation: `pulse ${3 + Math.random() * 4}s infinite ease-in-out ${Math.random() * 5}s`,
              }}
            />
          ))}
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% {
            transform: scale(0.8);
            opacity: 0.2;
          }
          50% {
            transform: scale(1.2);
            opacity: 0.6;
          }
        }
      `}</style>
    </div>
  )
}
