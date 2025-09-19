# Direct Azure CLI Installation Script
Write-Host "🔧 Installing Azure CLI directly..." -ForegroundColor Yellow

# Download Azure CLI MSI
$url = "https://azcliprod.azureedge.net/msi/azure-cli-2.77.0-x64.msi"
$output = "$env:TEMP\azure-cli.msi"

Write-Host "📥 Downloading Azure CLI..." -ForegroundColor Blue
try {
    Invoke-WebRequest -Uri $url -OutFile $output
    Write-Host "✅ Download completed" -ForegroundColor Green
    
    Write-Host "🚀 Installing Azure CLI..." -ForegroundColor Blue
    Start-Process msiexec.exe -Wait -ArgumentList "/i $output /quiet"
    
    Write-Host "✅ Installation completed!" -ForegroundColor Green
    Write-Host "🔄 Please restart PowerShell and try: az --version" -ForegroundColor Yellow
    
    # Clean up
    Remove-Item $output -Force
} catch {
    Write-Host "❌ Download failed. Please download manually from:" -ForegroundColor Red
    Write-Host "https://aka.ms/installazurecliwindows" -ForegroundColor Cyan
}