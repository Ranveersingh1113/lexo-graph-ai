# Phase 4: Final Integration & API - Detailed Plan

## Overview

Phase 4 integrates all previous stages (Stage 1: Layout Detection, Stage 2: OCR, Stage 3: Tables & Figures) into a unified API application using FastAPI. This provides a complete, production-ready system for document understanding.

## Objectives

1. **Create FastAPI Application**: Unified REST API for document processing
2. **Integrate All Stages**: Seamless pipeline from image input to complete output
3. **Final JSON Aggregator**: Combine all outputs into required format
4. **Error Handling**: Robust error handling and validation
5. **API Documentation**: Auto-generated API docs
6. **Testing & Validation**: API testing and validation endpoints

---

## Components to Build

### 1. FastAPI Application Structure

```
api/
├── main.py                 # FastAPI app entry point
├── models.py               # Pydantic models for request/response
├── dependencies.py         # Dependency injection (model loading, etc.)
├── routers/
│   ├── __init__.py
│   ├── process.py          # Document processing endpoint
│   ├── health.py           # Health check endpoint
│   └── batch.py            # Batch processing endpoint
├── services/
│   ├── __init__.py
│   ├── pipeline_service.py # Main pipeline orchestration
│   └── aggregator_service.py # Final JSON aggregation
└── utils/
    ├── __init__.py
    ├── validators.py       # Input validation
    └── formatters.py       # Output formatting
```

### 2. Request/Response Models

**Request Models:**
- `DocumentProcessRequest`: Image upload, processing options
- `BatchProcessRequest`: Multiple images, batch options
- `ConfigRequest`: Model selection, thresholds

**Response Models:**
- `DocumentProcessResponse`: Complete processed document
- `ElementResponse`: Individual element (text, table, figure)
- `ErrorResponse`: Error details
- `HealthResponse`: API health status

### 3. Main Pipeline Service

**Features:**
- Orchestrates Stage 1 → Stage 2 → Stage 3
- Handles image preprocessing
- Manages model loading and caching
- Error recovery and logging
- Progress tracking (for long operations)

### 4. Final JSON Aggregator

**Purpose:**
- Combines outputs from all stages
- Formats into challenge-required JSON structure
- Validates completeness
- Handles missing data gracefully

**Output Format:**
```json
{
  "document_id": "unique_id",
  "image": "filename.png",
  "processing_metadata": {
    "timestamp": "2025-11-05T...",
    "stages_completed": ["stage1", "stage2", "stage3"],
    "processing_time": 5.2
  },
  "elements": [
    {
      "type": "text",
      "class": 1,
      "bbox": [x, y, h, w],
      "content": {
        "text": "extracted text",
        "language": "en",
        "confidence": 0.95
      }
    },
    {
      "type": "table",
      "class": 4,
      "bbox": [x, y, h, w],
      "content": {
        "structured_data": {...},
        "summary": "table summary"
      }
    },
    {
      "type": "figure",
      "class": 5,
      "bbox": [x, y, h, w],
      "content": {
        "caption": "figure description",
        "confidence": 0.92
      }
    }
  ],
  "summary": {
    "total_elements": 10,
    "text_elements": 5,
    "tables": 2,
    "figures": 3
  }
}
```

### 5. API Endpoints

#### Core Endpoints

1. **POST `/api/v1/process`**
   - Process single document
   - Upload image file
   - Return complete JSON output

2. **POST `/api/v1/process/batch`**
   - Process multiple documents
   - Accept multiple image files
   - Return batch results

3. **GET `/api/v1/health`**
   - Health check
   - Model loading status
   - System status

4. **GET `/api/v1/status/{job_id}`**
   - Check processing status (for async jobs)
   - Get progress updates

#### Utility Endpoints

5. **GET `/api/v1/models`**
   - List available models
   - Show loaded models
   - Model information

6. **POST `/api/v1/config`**
   - Update processing configuration
   - Model selection
   - Thresholds

7. **GET `/api/v1/docs`**
   - Auto-generated API documentation (Swagger/OpenAPI)

### 6. Configuration Management

**Configuration File: `config/api_config.yaml`**
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  max_upload_size: 50MB
  allowed_extensions: [".png", ".jpg", ".jpeg", ".pdf"]
  
models:
  stage1:
    model_dir: "models/inference/ppyoloe_ps05"
    use_gpu: true
    score_threshold: 0.5
  
  stage2:
    config_path: "config/ocr_config.yaml"
  
  stage3:
    config_path: "config/stage3_config.yaml"

processing:
  enable_deskew: true
  max_skew_angle: 10.0
  save_intermediate: false
  async_processing: false  # For long-running jobs
```

### 7. Error Handling

**Error Types:**
- `ValidationError`: Invalid input
- `ProcessingError`: Pipeline errors
- `ModelError`: Model loading/execution errors
- `FileError`: File handling errors

**Error Response Format:**
```json
{
  "error": {
    "code": "PROCESSING_ERROR",
    "message": "Error description",
    "details": {...},
    "timestamp": "2025-11-05T..."
  }
}
```

### 8. Logging & Monitoring

**Features:**
- Structured logging
- Request/response logging
- Performance metrics
- Error tracking
- Processing time tracking

### 9. Testing

**Test Files:**
- `tests/test_api.py`: API endpoint tests
- `tests/test_pipeline.py`: Pipeline integration tests
- `tests/test_aggregator.py`: Aggregator tests

**Test Coverage:**
- Unit tests for services
- Integration tests for API
- End-to-end tests for complete pipeline

---

## Detailed Component Breakdown

### Component 1: FastAPI Application (`api/main.py`)

**Features:**
- FastAPI app initialization
- CORS configuration
- Middleware setup (logging, error handling)
- Route registration
- Startup/shutdown events

**Startup Events:**
- Load models (Stage 1, 2, 3)
- Initialize services
- Check dependencies

**Shutdown Events:**
- Cleanup resources
- Save state

### Component 2: Request/Response Models (`api/models.py`)

**Pydantic Models:**
- Request validation
- Response serialization
- Type checking
- Documentation generation

**Example:**
```python
class DocumentProcessRequest(BaseModel):
    image: UploadFile
    options: Optional[ProcessingOptions] = None

class DocumentProcessResponse(BaseModel):
    document_id: str
    elements: List[ElementResponse]
    summary: SummaryResponse
    metadata: ProcessingMetadata
```

### Component 3: Pipeline Service (`api/services/pipeline_service.py`)

**Methods:**
- `process_document(image, options)`: Main processing method
- `process_batch(images, options)`: Batch processing
- `validate_input(image)`: Input validation
- `preprocess_image(image)`: Image preprocessing

**Features:**
- Model caching (load once, reuse)
- Async processing support
- Progress tracking
- Error recovery

### Component 4: Aggregator Service (`api/services/aggregator_service.py`)

**Methods:**
- `aggregate_results(stage1, stage2, stage3)`: Combine all outputs
- `format_output(aggregated_data)`: Format to required structure
- `validate_completeness(output)`: Check if all required fields present
- `generate_summary(elements)`: Generate summary statistics

**Features:**
- Handles missing data gracefully
- Validates output structure
- Formats according to challenge requirements

### Component 5: API Routers

**Process Router (`api/routers/process.py`):**
- `/process` endpoint
- `/process/batch` endpoint
- Request validation
- Response formatting

**Health Router (`api/routers/health.py`):**
- `/health` endpoint
- Model status
- System health

**Batch Router (`api/routers/batch.py`):**
- `/batch` endpoint
- Job status tracking
- Progress updates

---

## Implementation Steps (in order)

### Step 1: API Structure & Models
- Create FastAPI app structure
- Define Pydantic models
- Setup basic routing

### Step 2: Pipeline Service
- Integrate Stage 1, 2, 3
- Model loading and caching
- Error handling

### Step 3: Aggregator Service
- Combine all outputs
- Format to required structure
- Validation

### Step 4: API Endpoints
- Implement core endpoints
- Add utility endpoints
- Error handling

### Step 5: Configuration & Logging
- Configuration management
- Structured logging
- Monitoring setup

### Step 6: Testing & Documentation
- API tests
- Integration tests
- API documentation

### Step 7: Deployment Preparation
- Docker configuration
- Environment variables
- Production settings

---

## Technical Specifications

### FastAPI Features

- **Async Support**: For concurrent processing
- **Automatic Docs**: Swagger UI at `/docs`
- **Type Validation**: Pydantic models
- **Dependency Injection**: Model loading, config
- **Background Tasks**: For long-running operations

### File Upload Handling

- **Max Size**: 50MB (configurable)
- **Formats**: PNG, JPG, JPEG, PDF (optional)
- **Validation**: File type, size, format
- **Storage**: Temporary files, cleanup

### Response Formatting

- **JSON**: Structured output
- **Validation**: Ensure all required fields
- **Serialization**: Pydantic models
- **Error Format**: Consistent error structure

---

## Dependencies

### Required Packages

```bash
pip install fastapi uvicorn python-multipart
pip install pydantic pydantic-settings
pip install aiofiles  # For async file handling
pip install python-jose[cryptography]  # For authentication (optional)
pip install passlib[bcrypt]  # For authentication (optional)
```

### Optional Packages

```bash
pip install redis  # For async job queue (optional)
pip install celery  # For background tasks (optional)
pip install prometheus-client  # For metrics (optional)
```

---

## API Endpoints Specification

### 1. POST `/api/v1/process`

**Request:**
- `image`: File upload (multipart/form-data)
- `options`: Optional JSON (processing options)

**Response:**
```json
{
  "document_id": "uuid",
  "elements": [...],
  "summary": {...},
  "metadata": {...}
}
```

### 2. POST `/api/v1/process/batch`

**Request:**
- `images`: Multiple file uploads
- `options`: Optional JSON

**Response:**
```json
{
  "batch_id": "uuid",
  "total": 5,
  "results": [...],
  "summary": {...}
}
```

### 3. GET `/api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "stage1": "loaded",
    "stage2": "loaded",
    "stage3": "loaded"
  },
  "timestamp": "..."
}
```

### 4. GET `/api/v1/models`

**Response:**
```json
{
  "stage1": {
    "model": "ppyoloe_ps05",
    "status": "loaded"
  },
  "stage2": {
    "provider": "google-cloud-vision",
    "status": "configured"
  },
  "stage3": {
    "tsr_model": "table-transformer",
    "caption_model": "blip2-opt-2.7b",
    "status": "loaded"
  }
}
```

---

## Error Handling Strategy

### Error Categories

1. **Client Errors (4xx)**
   - Validation errors
   - File format errors
   - Missing required fields

2. **Server Errors (5xx)**
   - Processing errors
   - Model errors
   - Internal errors

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "field": "specific error details"
    },
    "timestamp": "2025-11-05T..."
  }
}
```

---

## Performance Considerations

### Optimization Strategies

1. **Model Caching**: Load models once at startup
2. **Async Processing**: Use async/await for I/O operations
3. **Batch Processing**: Process multiple images efficiently
4. **Resource Management**: Limit concurrent requests
5. **Caching**: Cache results for duplicate images

### Scalability

- **Horizontal Scaling**: Multiple workers
- **Async Jobs**: For long-running operations
- **Queue System**: Redis/Celery for background tasks
- **Load Balancing**: Multiple API instances

---

## Security Considerations

### Authentication (Optional)

- API key authentication
- JWT tokens
- Rate limiting

### File Upload Security

- File type validation
- Size limits
- Virus scanning (optional)
- Sanitization

### Input Validation

- Pydantic validation
- File format checks
- Size limits
- Content validation

---

## Testing Strategy

### Unit Tests

- Service methods
- Aggregator logic
- Validators
- Formatters

### Integration Tests

- API endpoints
- Pipeline integration
- Error handling
- Response formatting

### End-to-End Tests

- Complete pipeline
- Real image processing
- Output validation

---

## Deployment Options

### Option 1: Local Development

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Docker

```dockerfile
FROM python:3.10
# ... Dockerfile
```

### Option 3: Cloud Deployment

- AWS Lambda (serverless)
- Google Cloud Run
- Azure Functions
- Heroku

---

## API Documentation

### Auto-generated Docs

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI JSON**: `/openapi.json`

### Manual Documentation

- Endpoint descriptions
- Request/response examples
- Error codes
- Usage examples

---

## File Structure

```
lexo-graph-ai/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── models.py               # Pydantic models
│   ├── dependencies.py         # Dependency injection
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── process.py
│   │   ├── health.py
│   │   └── batch.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pipeline_service.py
│   │   └── aggregator_service.py
│   └── utils/
│       ├── __init__.py
│       ├── validators.py
│       └── formatters.py
├── config/
│   └── api_config.yaml         # API configuration
├── tests/
│   ├── test_api.py
│   ├── test_pipeline.py
│   └── test_aggregator.py
└── requirements_api.txt        # API dependencies
```

---

## Implementation Timeline

1. **API Structure & Models** (1-2 hours)
2. **Pipeline Service** (1-2 hours)
3. **Aggregator Service** (1 hour)
4. **API Endpoints** (2-3 hours)
5. **Configuration & Logging** (1 hour)
6. **Testing** (1-2 hours)
7. **Documentation** (1 hour)

**Total Estimated Time**: 8-12 hours

---

## Success Criteria

✅ All stages integrated and working
✅ API endpoints functional
✅ Complete JSON output format
✅ Error handling robust
✅ API documentation complete
✅ Tests passing
✅ Ready for deployment

---

## Next Steps After Phase 4

1. **Performance Optimization**: Optimize for production
2. **Monitoring**: Add metrics and monitoring
3. **Deployment**: Deploy to production environment
4. **Documentation**: User documentation
5. **Maintenance**: Ongoing updates and improvements

---

Ready to proceed? This plan provides a complete, production-ready API that integrates all stages of your document understanding pipeline!

