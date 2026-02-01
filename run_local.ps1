$IMAGE_NAME = "aml-api:local"
$PORT = 8080

Write-Host "▶ Building Docker image..."
docker build -t $IMAGE_NAME .

Write-Host "▶ Starting AML API on http://localhost:$PORT"
docker run -p ${PORT}:8080 $IMAGE_NAME