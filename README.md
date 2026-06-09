# Holographic Window Demo

This project demonstrates a "True Parallax" effect using your webcam and Three.js.

## How to Run

Because this project uses ES6 modules and Webcam access, it works best when served via a local web server (browsers often block these features when opening `file://` directly).

### Option 1: Using Python (Recommended)
1. Open a terminal/command prompt.
2. Navigate to this folder:
   ```bash
   cd d:/Smith/HolographicWindow
   ```
3. Start a simple HTTP server:
   ```bash
   python -m http.server
   ```
4. Open your browser and go to: `http://localhost:8000`

### Option 2: VS Code Live Server
If you use VS Code, you can right-click `index.html` and select "Open with Live Server".

## How to Use
1. Allow camera access when prompted.
2. Wait a moment for the AI model to load.
3. Move your head left, right, up, and down.
4. Observe how the 3D scene perspective changes to match your head position, creating the illusion of a window into a 3D room.
