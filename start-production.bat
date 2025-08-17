@echo off
echo 🚀 Starting AI Identities in production mode...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install it first.
    pause
    exit /b 1
)

REM Build and start services
echo 🔨 Building services...
docker-compose build
if errorlevel 1 (
    echo ❌ Build failed. Check the error messages above.
    pause
    exit /b 1
)

echo 🚀 Starting services...
docker-compose up -d
if errorlevel 1 (
    echo ❌ Failed to start services. Check the error messages above.
    pause
    exit /b 1
)

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service status
echo 📊 Service status:
docker-compose ps

REM Check if services are running
docker-compose ps | findstr "Up" >nul
if errorlevel 1 (
    echo ❌ Some services failed to start. Check logs with: docker-compose logs
    pause
    exit /b 1
)

echo.
echo ✅ Services started successfully!
echo 🌐 Frontend: http://localhost:3000
echo 🔬 Classifier API: http://localhost:8000
echo 📚 API Documentation: http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop services: docker-compose down
echo.
pause
