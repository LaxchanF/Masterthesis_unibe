# This script creates Folders by "Degree XY" in the path of Study II:

# Define the path where the folders will be created
$neuerPath = "C:\Users\Laxchan\OneDrive - Universitaet Bern\Dokumente\Psy studium\Mastser\Masterarbeit\My_code\study II\Dataset\Prototype"

# Loop through degrees in steps of 5, from -30 to 30
for ($degree = 0; $degree -le 30; $degree += 5) {
    #Create the folder name
    $folderName = "Degree $degree"
    $h = "h"
    $q = "q"
    $p = "p"
    
    # Create the full path for the folder
    $folderPath = Join-Path -Path $neuerPath -ChildPath $folderName
    
    # Create the folder
    New-Item -Path $folderPath -ItemType Directory

   #Create the full path for the underfolder
    $UnderfolderPathH = Join-Path -Path $folderPath -ChildPath $h

    # Create the H folder
    New-Item -Path $UnderfolderPathH -ItemType Directory

       #Create the full path for the underfolder
    $UnderfolderPathP = Join-Path -Path $folderPath -ChildPath $p

    # Create the P folder
    New-Item -Path $UnderfolderPathP -ItemType Directory

       #Create the full path for the underfolder
    $UnderfolderPathQ = Join-Path -Path $folderPath -ChildPath $q

    # Create the H folder
    New-Item -Path $UnderfolderPathQ -ItemType Directory

   }