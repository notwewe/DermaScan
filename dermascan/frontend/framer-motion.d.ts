// This file adds TypeScript support for framer-motion with className
import "framer-motion"

declare module "framer-motion" {
  export interface MotionProps {
    className?: string
  }
}
