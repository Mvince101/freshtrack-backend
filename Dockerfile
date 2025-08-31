# Use Node.js 18 LTS
FROM node:18-alpine

# Install system dependencies for Sharp and ONNX
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    libc6-compat \
    && rm -rf /var/cache/apk/*

# Set Node.js memory limits for Railway
ENV NODE_OPTIONS="--max-old-space-size=512"

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with specific flags for ONNX
RUN npm ci --only=production --ignore-scripts=false

# Copy source code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 3000

# Health check - more lenient for Railway
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=5 \
  CMD node -e "require('http').get('http://localhost:3000/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

# Start the application with a small delay for Railway
CMD ["sh", "-c", "sleep 5 && npm start"]
