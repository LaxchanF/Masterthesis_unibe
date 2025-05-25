# Define source and destination directories
$sourceDir = "C:\Masterthesis_unibe\my_work\dnns\studyII-Blender\dataset\Testing\Prototype\H"
$destDir = "C:\Masterthesis_unibe\my_work\dnns\studyII-Blender\dataset\Trainig\Prototype\H"

# Create destination directory if it doesn't exist
if (-not (Test-Path -Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir
}

# Define regex pattern to match filenames ending with z_-30 or z_30 before .png
$pattern = '^.*z_(-?30)\.png$'

# Get all .png files and filter by regex
Get-ChildItem -Path $sourceDir -Filter *.png | Where-Object {
    $_.Name -match $pattern
} | ForEach-Object {
    $sourcePath = $_.FullName
    $destPath = Join-Path $destDir $_.Name
    Copy-Item -Path $sourcePath -Destination $destPath
}
