import { type NextRequest, NextResponse } from "next/server"

// This is a server-side proxy to bypass CORS issues
export async function POST(request: NextRequest) {
  try {
    // Get the request body
    const formData = await request.formData()
    
    // Check if the image is too large and resize it if needed
    const file = formData.get('file') as File
    if (file && file.size > 1000000) { // 1MB
      console.log(`Large image detected (${file.size} bytes), resizing before sending to API`)
      
      // Create a new FormData with the resized image
      const newFormData = new FormData()
      
      try {
        // Convert file to blob
        const arrayBuffer = await file.arrayBuffer()
        const blob = new Blob([arrayBuffer])
        
        // Create an image element to resize
        const img = new Image()
        const blobUrl = URL.createObjectURL(blob)
        
        // Wait for the image to load
        await new Promise((resolve, reject) => {
          img.onload = resolve
          img.onerror = reject
          img.src = blobUrl
        })
        
        // Resize the image
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        
        // Calculate new dimensions (max 1000px on longest side)
        const aspectRatio = img.width / img.height
        let newWidth, newHeight
        
        if (img.width > img.height) {
          newWidth = Math.min(1000, img.width)
          newHeight = newWidth / aspectRatio
        } else {
          newHeight = Math.min(1000, img.height)
          newWidth = newHeight * aspectRatio
        }
        
        canvas.width = newWidth
        canvas.height = newHeight
        ctx?.drawImage(img, 0, 0, newWidth, newHeight)
        
        // Convert to blob
        const resizedBlob = await new Promise<Blob>((resolve) => {
          canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.85)
        })
        
        // Add to form data
        newFormData.append('file', resizedBlob, file.name)
        let formData = newFormData
        
        // Clean up
        URL.revokeObjectURL(blobUrl)
      } catch (error) {
        console.error("Error resizing image:", error)
        // Continue with original image if resize fails
      }
    }

    // Set a longer timeout for the fetch request (90 seconds)
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 90000)

    // Forward the request to the backend API
    const response = await fetch("https://dermascan-56zs.onrender.com/api/predict", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    })

    // Clear the timeout
    clearTimeout(timeoutId)

    // If the response is not OK, throw an error
    if (!response.ok) {
      const errorText = await response.text()
      console.error("Backend API error:", errorText)
      return NextResponse.json({ error: "Failed to get prediction from backend" }, { status: response.status })
    }

    // Get the response data
    const data = await response.json()

    // Return the response data
    return NextResponse.json(data)
  } catch (error) {
    console.error("Proxy error:", error)
    
    // Check if it's an AbortError (timeout)
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json({ 
        error: "The request timed out. The server might be under heavy load or the image might be too large." 
      }, { status: 504 })
    }
    
    return NextResponse.json({ error: "Internal server error in proxy" }, { status: 500 })
  }
}
