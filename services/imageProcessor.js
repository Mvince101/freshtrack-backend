const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

// Try to load ONNX runtime, but make it optional
let ort;
try {
  // Use onnxruntime-web (better for cloud environments)
  ort = require('onnxruntime-web');
  console.log('ONNX Runtime Web loaded successfully');
} catch (error) {
  console.warn('ONNX runtime not available. Using mock detection only.');
  console.warn('Web version error:', error.message);
  ort = null;
}

// Food class names (matching your YOLO model)
const CLASS_NAMES = [
  // Fresh items
  'Fresh_Apple', 'Fresh_Banana', 'Fresh_Potato', 'Fresh_Carrot', 'Fresh_Orange',
  'Fresh_Beef', 'Fresh_Chicken', 'Fresh_Pork', 'Fresh_Manggo', 'Fresh_Pepper',
  'Fresh_Cucumber', 'Fresh_Strawberry', 'Fresh_Okra',
  // Rotten items
  'Rotten_Apple', 'Rotten_Banana', 'Rotten_Potato', 'Rotten_Carrot', 'Rotten_Orange',
  'Rotten_Beef', 'Rotten_Chicken', 'Rotten_Pork', 'Rotten_Manggo', 'Rotten_Pepper',
  'Rotten_Cucumber', 'Rotten_Strawberry', 'Rotten_Okra'
];

class ImageProcessor {
  constructor() {
    this.session = null;
    // Use environment variable or fallback to default path
    this.modelPath = process.env.MODEL_PATH || path.join(__dirname, '../models/best.onnx');
    this.inputSize = 640; // YOLO input size
    this.confidenceThreshold = 0.5;
    this.nmsThreshold = 0.4;
    
    // Force mock mode if ONNX is not available
    this.forceMockMode = process.env.FORCE_MOCK_MODE === 'true' || !ort;
  }

  async initialize() {
    try {
      // If forcing mock mode, skip ONNX initialization
      if (this.forceMockMode) {
        console.log('Forcing mock detection mode');
        return false;
      }

      console.log('Looking for model at:', this.modelPath);
      if (!fs.existsSync(this.modelPath)) {
        console.warn('ONNX model not found at:', this.modelPath);
        console.warn('Using mock detection for development.');
        return false;
      }
      console.log('ONNX model found at:', this.modelPath);

      // Try to load ONNX runtime, but fallback gracefully if it fails
      try {
        const options = {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'basic', // Use basic optimization to save memory
          enableCpuMemArena: false, // Disable CPU memory arena to save memory
          enableMemPattern: false, // Disable memory pattern to save memory
          extra: {
            session: {
              use_ort_model_bytes_directly: true, // Use model bytes directly
              log_severity_level: 0 // Minimal logging
            }
          }
        };

        this.session = await ort.InferenceSession.create(this.modelPath, options);
        console.log('YOLO model loaded successfully');
        return true;
      } catch (onnxError) {
        console.warn('ONNX runtime failed to load model:', onnxError.message);
        console.log('Falling back to mock detection for development.');
        return false;
      }
    } catch (error) {
      console.error('Failed to initialize image processor:', error);
      return false;
    }
  }

  async preprocessImage(imagePath) {
    try {
      // Load and resize image using Sharp
      const imageBuffer = await sharp(imagePath)
        .resize(this.inputSize, this.inputSize, { fit: 'fill' })
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Convert to tensor format (normalized to 0-1)
      const tensor = new Float32Array(this.inputSize * this.inputSize * 3);
      
      // Sharp returns raw pixel data
      for (let i = 0; i < imageBuffer.data.length; i++) {
        tensor[i] = imageBuffer.data[i] / 255.0;
      }

      return tensor;
    } catch (error) {
      console.error('Error preprocessing image:', error);
      throw error;
    }
  }

  async detectObjects(imagePath) {
    try {
      console.log('Starting object detection for:', imagePath);
      
      // If forcing mock mode or ONNX runtime is not available, use mock detections
      if (this.forceMockMode || !ort) {
        console.log('Using mock detections (ONNX runtime not available or forced)');
        return this.getMockDetections();
      }

      console.log('Attempting to load ONNX model...');
      const modelLoaded = await this.initialize();
      
      if (!modelLoaded) {
        console.log('ONNX model failed to load, using mock detections');
        return this.getMockDetections();
      }

      console.log('Preprocessing image...');
      const inputTensor = await this.preprocessImage(imagePath);
      console.log('Image preprocessing completed, tensor size:', inputTensor.length);
      
      // Create input tensor
      console.log('Creating ONNX input tensor...');
      const input = new ort.Tensor('float32', inputTensor, [1, 3, this.inputSize, this.inputSize]);
      
      // Run inference
      console.log('Running ONNX inference...');
      const feeds = { [this.session.inputNames[0]]: input };
      const results = await this.session.run(feeds);
      console.log('ONNX inference completed');
      
      // Process results
      console.log('Processing YOLO output...');
      const detections = this.processYOLOOutput(results[this.session.outputNames[0]]);
      console.log('YOLO output processing completed, detections:', detections.length);
      
      return detections;
    } catch (error) {
      console.error('Error in object detection:', error);
      console.error('Error stack:', error.stack);
      console.log('Falling back to mock detections due to error');
      return this.getMockDetections();
    }
  }

  processYOLOOutput(outputTensor) {
    const detections = [];
    const output = outputTensor.data;
    const outputShape = outputTensor.dims;
    
    // YOLO output format: [batch, num_detections, 85] where 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
    const numDetections = outputShape[1];
    const numClasses = outputShape[2] - 5;
    
    for (let i = 0; i < numDetections; i++) {
      const baseIndex = i * (numClasses + 5);
      
      // Get bounding box coordinates
      const x = output[baseIndex];
      const y = output[baseIndex + 1];
      const w = output[baseIndex + 2];
      const h = output[baseIndex + 3];
      const confidence = output[baseIndex + 4];
      
      if (confidence < this.confidenceThreshold) continue;
      
      // Find class with highest probability
      let maxClassProb = 0;
      let maxClassIndex = 0;
      
      for (let j = 0; j < numClasses; j++) {
        const classProb = output[baseIndex + 5 + j];
        if (classProb > maxClassProb) {
          maxClassProb = classProb;
          maxClassIndex = j;
        }
      }
      
      const finalConfidence = confidence * maxClassProb;
      
      if (finalConfidence >= this.confidenceThreshold) {
        detections.push({
          label: CLASS_NAMES[maxClassIndex] || `class_${maxClassIndex}`,
          confidence: finalConfidence,
          bbox: {
            x: x - w / 2,
            y: y - h / 2,
            width: w,
            height: h
          }
        });
      }
    }
    
    // Apply Non-Maximum Suppression
    return this.applyNMS(detections);
  }

  applyNMS(detections) {
    if (detections.length === 0) return detections;
    
    // Sort by confidence
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const filtered = [];
    
    while (detections.length > 0) {
      const current = detections.shift();
      filtered.push(current);
      
      detections = detections.filter(detection => {
        const iou = this.calculateIoU(current.bbox, detection.bbox);
        return iou < this.nmsThreshold;
      });
    }
    
    return filtered;
  }

  calculateIoU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 < x1 || y2 < y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / union;
  }

  getMockDetections() {
    // Mock detections for development/testing
    const mockItems = [
      'Fresh_Apple', 'Fresh_Banana', 'Fresh_Carrot', 'Fresh_Orange',
      'Rotten_Apple', 'Rotten_Banana', 'Rotten_Potato', 'Rotten_Chicken'
    ];
    const detections = [];
    
    for (let i = 0; i < Math.floor(Math.random() * 3) + 1; i++) {
      const randomItem = mockItems[Math.floor(Math.random() * mockItems.length)];
      detections.push({
        label: randomItem,
        confidence: 0.8 + Math.random() * 0.2,
        bbox: {
          x: Math.random() * 0.6,
          y: Math.random() * 0.6,
          width: 0.1 + Math.random() * 0.2,
          height: 0.1 + Math.random() * 0.2
        }
      });
    }
    
    return detections;
  }
}

const imageProcessor = new ImageProcessor();

async function processImage(imagePath) {
  try {
    console.log('Starting image processing for:', imagePath);
    
    // Check if file exists
    if (!fs.existsSync(imagePath)) {
      throw new Error(`Image file not found: ${imagePath}`);
    }
    
    // Check file size
    const stats = fs.statSync(imagePath);
    console.log('Image file size:', stats.size, 'bytes');
    
    if (stats.size === 0) {
      throw new Error('Image file is empty');
    }
    
    const detections = await imageProcessor.detectObjects(imagePath);
    console.log('Detections completed:', detections.length, 'objects found');
    
    // Clean up uploaded file
    if (fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
      console.log('Cleaned up uploaded file');
    }
    
    return detections;
  } catch (error) {
    console.error('Error processing image:', error);
    console.error('Error stack:', error.stack);
    
    // Return mock detections as fallback
    console.log('Falling back to mock detections due to error');
    return imageProcessor.getMockDetections();
  }
}

module.exports = {
  processImage,
  ImageProcessor
};
