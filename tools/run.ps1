param([Parameter(Mandatory=$true)][string]$Task, [string[]]$TaskArgs)
$ErrorActionPreference = 'Stop'
Set-Location 'D:\ML'
switch ($Task) {
  'inspect' {
    Get-ChildItem -LiteralPath 'C:\Users\DELL\.codex\plugins\cache\openai-primary-runtime\pdf\26.909.12148' -Force | Select-Object Name,Mode
    Get-ChildItem -LiteralPath 'C:\Users\DELL\.codex' -Filter 'mark_artifact_operation_started.mjs' -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
  }
  'bootstrap' {
    if (!(Test-Path '.venv\Scripts\python.exe')) { py -m venv .venv }
    & '.venv\Scripts\python.exe' -m pip install --upgrade pip
    & '.venv\Scripts\python.exe' -m pip install numpy pandas scipy scikit-learn matplotlib seaborn nbformat nbclient ipykernel reportlab pymupdf requests huggingface-hub
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
  }
  'python' { & '.venv\Scripts\python.exe' @TaskArgs; if ($LASTEXITCODE -ne 0) { throw "Python failed: $LASTEXITCODE" } }
  default { throw "Unknown task: $Task" }
}
