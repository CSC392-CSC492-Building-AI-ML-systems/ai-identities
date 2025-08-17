# Production Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- Pre-trained model files in `classifier_service/resources/`
- Ports 3000 and 8000 available

## Quick Deployment

### 1. Start All Services
```bash
# Linux/Mac
./start-production.sh

# Windows
start-production.bat

# Or manually
docker-compose up -d
```

### 2. Verify Services
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Test endpoints
curl http://localhost:8000/docs
curl http://localhost:3000
```

## Production Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
# Classifier Service
CLASSIFIER_HOST=0.0.0.0
CLASSIFIER_PORT=8000
CLASSIFIER_RELOAD=false

# Frontend
NODE_ENV=production
CLASSIFIER_SERVICE_URL=http://classifier-service:8000
```

### Port Configuration
Update `docker-compose.yml` if you need different ports:
```yaml
ports:
  - "YOUR_PORT:3000"  # Frontend
  - "YOUR_API_PORT:8000"  # Classifier API
```

## Monitoring

### Health Checks
- Frontend: http://localhost:3000
- Classifier API: http://localhost:8000/docs
- Health endpoint: Built into Docker Compose

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f classifier-service
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100
```

## Maintenance

### Updates
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose up --build -d

# Or just restart
docker-compose restart
```

### Backup
```bash
# Backup model files
cp -r classifier_service/resources/ backup/

# Backup Docker volumes (if any)
docker run --rm -v ai-identities_data:/data -v $(pwd):/backup alpine tar czf /backup/backup.tar.gz /data
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using the port
   netstat -tulpn | grep :8000
   
   # Kill the process or change ports in docker-compose.yml
   ```

2. **Model files missing**
   ```bash
   # Ensure resources directory exists
   ls -la classifier_service/resources/
   
   # Should contain: vectorizer.pkl, library_averages.pkl
   ```

3. **Service won't start**
   ```bash
   # Check logs
   docker-compose logs classifier-service
   
   # Check health status
   docker-compose ps
   ```

### Performance Tuning

- **Memory**: Adjust Docker memory limits in docker-compose.yml
- **CPU**: Use CPU limits for resource management
- **Networks**: Use custom networks for better isolation

## Security

- **Firewall**: Only expose necessary ports
- **HTTPS**: Use reverse proxy (nginx) for SSL termination
- **Authentication**: Add API key validation if needed
- **Updates**: Keep Docker images updated

## Scaling

For high traffic, consider:
- Load balancer in front of multiple classifier instances
- Redis for caching
- Database for storing results
- Monitoring with Prometheus/Grafana

