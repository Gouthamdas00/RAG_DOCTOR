param(
    [string]$ModelName = $env:RAG_DOCTOR_TEST_MODEL_NAME,
    [ValidateRange(30, 900)]
    [int]$TimeoutSeconds = 300,
    [string]$Endpoint = $(if ($env:RAG_DOCTOR_TEST_MODEL_ENDPOINT) { $env:RAG_DOCTOR_TEST_MODEL_ENDPOINT } else { 'http://127.0.0.1:11434/api/chat' }),
    [string]$TestName = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ModelName)) {
    throw 'Provide -ModelName (for example qwen3:4b) or set RAG_DOCTOR_TEST_MODEL_NAME.'
}

$ollama = Get-Command ollama -ErrorAction SilentlyContinue

$endpointUri = $null
if (-not [Uri]::TryCreate($Endpoint, [UriKind]::Absolute, [ref]$endpointUri) -or
    $endpointUri.Scheme -ne 'http' -or
    -not $endpointUri.IsLoopback) {
    throw 'The test endpoint must be an HTTP loopback URL, such as http://127.0.0.1:11434/api/chat.'
}

$installedModels = @()
$endpointError = $null
$tagsBuilder = [System.UriBuilder]::new($endpointUri)
$tagsBuilder.Path = '/api/tags'
$tagsBuilder.Query = ''
$tagsBuilder.Fragment = ''
try {
    # Test the same local service that the Python quality suite will call.
    # This also works when Ollama is installed as a desktop app and its CLI is
    # not available in this PowerShell session's PATH.
    $tagResponse = Invoke-RestMethod -Method Get -Uri $tagsBuilder.Uri.AbsoluteUri -TimeoutSec 5 -UseBasicParsing
    $installedModels = @(
        @($tagResponse.models) |
            ForEach-Object {
                if ($_.name) { [string]$_.name }
                elseif ($_.model) { [string]$_.model }
            } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
} catch {
    $endpointError = $_.Exception.Message
}

if ($installedModels.Count -eq 0 -and $ollama) {
    $modelOutput = @(& $ollama.Source list 2>&1)
    if ($LASTEXITCODE -eq 0) {
        $installedModels = @(
            $modelOutput |
                Select-Object -Skip 1 |
                Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
                ForEach-Object { (($_.ToString().Trim()) -split '\s+')[0] }
        )
    } else {
        $endpointError = "HTTP model discovery failed and Ollama CLI returned: $($modelOutput -join ' ')"
    }
}

if ($installedModels.Count -eq 0 -and $endpointError) {
    throw "Could not discover models from the local endpoint $($tagsBuilder.Uri.AbsoluteUri). $endpointError Start the local model server or install Ollama."
}
if ($installedModels -notcontains $ModelName) {
    $available = if ($installedModels.Count) { $installedModels -join ', ' } else { '(none)' }
    throw "Model '$ModelName' is not installed on the local server. Installed local models: $available. Install it with 'ollama pull $ModelName' or configure an installed model name."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw 'Python was not found. Create the project virtual environment or install Python.'
    }
    $python = $pythonCommand.Source
}

$oldModel = $env:RAG_DOCTOR_TEST_MODEL_NAME
$oldEndpoint = $env:RAG_DOCTOR_TEST_MODEL_ENDPOINT
$oldTimeout = $env:RAG_DOCTOR_LOCAL_MODEL_TIMEOUT
$testExitCode = 1
try {
    $env:RAG_DOCTOR_TEST_MODEL_NAME = $ModelName
    $env:RAG_DOCTOR_TEST_MODEL_ENDPOINT = $Endpoint
    $env:RAG_DOCTOR_LOCAL_MODEL_TIMEOUT = [string]$TimeoutSeconds
    if ([string]::IsNullOrWhiteSpace($TestName)) {
        & $python scripts\run_agentic_model_live.py
    }
    else {
        & $python scripts\run_agentic_model_live.py --test-name $TestName
    }
    $testExitCode = $LASTEXITCODE
}
finally {
    $env:RAG_DOCTOR_TEST_MODEL_NAME = $oldModel
    $env:RAG_DOCTOR_TEST_MODEL_ENDPOINT = $oldEndpoint
    $env:RAG_DOCTOR_LOCAL_MODEL_TIMEOUT = $oldTimeout
}

exit $testExitCode
