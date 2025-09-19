# BCI EEG - Windows Tools Installation Script
# Run this in PowerShell as Administrator

Write-Host "🛠️  Installing Required Tools for BCI EEG Deployment" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Blue

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")

if (-not $isAdmin) {
    Write-Host "❌ Please run PowerShell as Administrator" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Install winget if not available
if (-not (Test-Command "winget")) {
    Write-Host "📦 Installing winget..." -ForegroundColor Yellow
    # winget is usually pre-installed on Windows 10/11
    Write-Host "⚠️  winget not found. Please install from Microsoft Store: 'App Installer'" -ForegroundColor Yellow
}

# Install Azure CLI
Write-Host "🔧 Installing Azure CLI..." -ForegroundColor Yellow
if (Test-Command "winget") {
    winget install -e --id Microsoft.AzureCLI
} else {
    Write-Host "📥 Please download Azure CLI from: https://aka.ms/installazurecliwindows" -ForegroundColor Cyan
    Start-Process "https://aka.ms/installazurecliwindows"
}

# Install kubectl
Write-Host "🔧 Installing kubectl..." -ForegroundColor Yellow
if (Test-Command "winget") {
    winget install -e --id Kubernetes.kubectl
} else {
    Write-Host "📥 Please download kubectl from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/" -ForegroundColor Cyan
    Start-Process "https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/"
}

# Install Docker Desktop
Write-Host "🔧 Installing Docker Desktop..." -ForegroundColor Yellow
if (Test-Command "winget") {
    winget install -e --id Docker.DockerDesktop
} else {
    Write-Host "📥 Please download Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Cyan
    Start-Process "https://www.docker.com/products/docker-desktop/"
}

Write-Host "✅ Installation commands completed!" -ForegroundColor Green
Write-Host "🔄 Please restart PowerShell and run the following to verify:" -ForegroundColor Yellow
Write-Host "   az --version" -ForegroundColor Cyan
Write-Host "   kubectl version --client" -ForegroundColor Cyan
Write-Host "   docker --version" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 After verification, run: ./deploy-bci-eeg.sh" -ForegroundColor Green