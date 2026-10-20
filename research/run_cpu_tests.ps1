$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
python -m pytest research/tests tests -v -p no:cacheprovider
exit $LASTEXITCODE
