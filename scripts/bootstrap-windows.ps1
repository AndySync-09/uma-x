$ErrorActionPreference = "Stop"

Write-Host "UMA-X bootstrap"
Write-Host "==============="

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.10+ is required."
}

python -m pip install -e .

Write-Host ""
Write-Host "UMA-X installed in editable mode."
Write-Host "Next:"
Write-Host "  1. Ensure llama-cli.exe is on PATH, or use --llama-bin."
Write-Host "  2. Run: umax doctor"
Write-Host "  3. Run: umax bench C:\path\to\small-model.gguf --runs 3"
