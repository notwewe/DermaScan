import { type NextRequest, NextResponse } from "next/server"

// This is a server-side proxy to bypass CORS issues
export async function POST(request: NextRequest) {
  try {
    // Get the request body
    const formData = await request.formData()

    // Forward the request to the backend API
    const response = await fetch("https://dermascan-56zs.onrender.com/api/predict", {
      method: "POST",
      body: formData,
      // No need to set CORS headers here since this is a server-side request
    })

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
    return NextResponse.json({ error: "Internal server error in proxy" }, { status: 500 })
  }
}
