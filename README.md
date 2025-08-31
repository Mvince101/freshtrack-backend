# FreshTrack Backend

A Node.js/Express backend for the FreshTrack food freshness detection app. This backend provides image processing capabilities using YOLO object detection and manages food storage data.

## Features

- **Image Processing**: YOLO-based object detection for food items
- **Storage Management**: Comprehensive food storage and shelf life data
- **RESTful API**: Clean API endpoints for mobile app integration
- **File Upload**: Secure image upload handling
- **Error Handling**: Robust error handling and logging

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Python (for model conversion if needed)

## Installation

1. **Clone the repository and navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   ```bash
   cp env.example .env
   ```
   Edit `.env` file with your configuration.

4. **Add YOLO model (optional):**
   
   **Option A: Convert your .pt model (Recommended)**
   ```bash
   # Place your best.pt file in the backend directory
   # Then run the conversion script:
   python convert_model.py
   # Or on Windows:
   convert_model.bat
   ```
   
   **Option B: Manual conversion**
   ```python
   from ultralytics import YOLO
   model = YOLO('best.pt')
   model.export(format='onnx', opset=11)
   # Then move best.onnx to models/ directory
   ```
   
   **Option C: Use mock detection (Development)**
   - If no model is provided, the system will use mock detections for development

## Usage

### Development Mode
```bash
npm run dev
```

### Production Mode
```bash
npm start
```

The server will start on `http://localhost:3000` (or the port specified in your `.env` file).

## API Endpoints

### Health Check
- **GET** `/health` - Check server status

### Image Detection
- **POST** `/api/detect` - Detect food items in uploaded image
  - Body: `multipart/form-data` with `image` field
  - Returns: Array of detected items with storage information

### Storage Data
- **GET** `/api/storage` - Get all storage data
- **GET** `/api/storage/:item` - Get storage info for specific item

## Example Usage

### Detect Food Items
```bash
curl -X POST \
  http://localhost:3000/api/detect \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@/path/to/your/image.jpg'
```

### Get Storage Info
```bash
curl http://localhost:3000/api/storage/apple
```

## Response Format

### Detection Response
```json
{
  "success": true,
  "detections": [
    {
      "label": "apple",
      "confidence": 0.95,
      "bbox": {
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4
      },
      "storage": {
        "storage": "Refrigerate in crisper drawer",
        "shelf_life": 14,
        "tips": "Store away from other fruits to prevent ripening",
        "signs_of_spoilage": "Soft spots, mold, wrinkled skin"
      }
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Project Structure

```
backend/
├── server.js              # Main server file
├── package.json           # Dependencies and scripts
├── services/
│   ├── imageProcessor.js  # YOLO detection service
│   └── storageService.js  # Food storage data management
├── data/
│   └── storage_data.json  # Food storage database
├── models/
│   └── best.onnx         # YOLO model (optional)
├── uploads/              # Temporary file uploads
└── README.md            # This file
```

## Configuration

### Environment Variables

- `PORT`: Server port (default: 3000)
- `NODE_ENV`: Environment mode (development/production)
- `ROBOFLOW_API_KEY`: API key for Roboflow (if using)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.5)
- `NMS_THRESHOLD`: Non-maximum suppression threshold (default: 0.4)

### Model Configuration

The backend supports ONNX format YOLO models. To use your own model:

1. Convert your YOLO model to ONNX format
2. Place it in the `models/` directory as `best.onnx`
3. Update the class names in `services/imageProcessor.js` if needed

## Development

### Adding New Food Items

To add new food items to the storage database:

1. Edit `services/storageService.js`
2. Add new entries to `DEFAULT_STORAGE_DATA`
3. Restart the server

### Customizing Detection

- Modify confidence thresholds in `services/imageProcessor.js`
- Update class names array to match your model
- Adjust preprocessing parameters as needed

## Troubleshooting

### Common Issues

1. **Canvas/ONNX installation errors**: The system will use mock detections if native dependencies fail to install
2. **Model not found**: The system will use mock detections if no ONNX model is provided
3. **Memory issues**: Reduce image size or adjust batch processing
4. **CORS errors**: Update CORS configuration in `server.js`

### Windows Installation Issues

If you encounter native dependency compilation errors on Windows:

1. **Use the simplified installation:**
   ```bash
   # Copy the simplified package.json
   cp package-simple.json package.json
   npm install
   ```

2. **Or clean and reinstall:**
   ```bash
   node clean-install.js
   ```

3. **Alternative: Use yarn instead of npm:**
   ```bash
   npm install -g yarn
   yarn install
   ```

4. **Install Visual Studio Build Tools:**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install "Build Tools for Visual Studio"
   - Include "C++ build tools" and "Windows 10 SDK"

5. **Use Node.js LTS version:**
   - Download Node.js LTS from: https://nodejs.org/
   - Avoid using Node.js 22+ for better compatibility

### Logs

Check console output for detailed error messages and processing logs.

## License

MIT License - see LICENSE file for details.
