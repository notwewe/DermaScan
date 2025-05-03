import type React from "react"
import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import Link from "next/link"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "DermaScan - AI-Powered Skin Lesion Detection",
  description: "Using advanced deep learning to help identify and classify skin lesions with medical-grade accuracy.",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
          <div className="flex flex-col min-h-screen">
            <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 sticky top-0 z-50">
              <div className="container mx-auto px-4 py-4">
                <nav className="flex justify-between items-center">
                  <Link href="/" className="text-2xl font-bold text-slate-900 dark:text-white">
                    <span className="text-teal-600 dark:text-teal-400">Derma</span>Scan
                  </Link>
                  <ul className="flex space-x-6">
                    <li>
                      <Link
                        href="/"
                        className="text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                      >
                        Home
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/detection"
                        className="text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                      >
                        Detection
                      </Link>
                    </li>
                    <li>
                      <Link
                        href="/about"
                        className="text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                      >
                        About
                      </Link>
                    </li>
                  </ul>
                </nav>
              </div>
            </header>

            {children}

            <footer className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-t border-slate-200 dark:border-slate-800 mt-auto">
              <div className="container mx-auto px-4 py-6">
                <div className="flex flex-col md:flex-row justify-between items-center">
                  <p className="text-slate-700 dark:text-slate-300 mb-4 md:mb-0">
                    © {new Date().getFullYear()} DermaScan - AI-Powered Skin Lesion Detection
                  </p>
                  <div className="flex space-x-4">
                    <Link
                      href="/about"
                      className="text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                    >
                      About
                    </Link>
                    <a
                      href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-slate-700 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400 transition-colors"
                    >
                      Dataset
                    </a>
                  </div>
                </div>
              </div>
            </footer>
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}
