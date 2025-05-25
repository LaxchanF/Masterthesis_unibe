# This Script checks the Blender Output, moves 

 # CONSTANTS:
    # Define the folder path to extract from 
    $RENDERED = "C:\Users\Laxchan\OneDrive - Universitaet Bern\Dokumente\Psy studium\Mastser\Masterarbeit\My_code\blender_laxchan\rendered"
    # Define the folder path to move to 
    $ZIELORDNER = "C:\Masterthesis_unibe\my_work\dnns\studyII-Blender\dataset\Diverse"
    #Make list of letters
    $LETTERS = @('p', 'h', 'q')

    foreach ($letter in $LETTERS){
    #Define Destination to move 
    $folderPath = Join-Path -Path $ZIELORDNER -ChildPath $letter

    #send picture to right letter folder
    Get-ChildItem -Path $RENDERED | Where-Object {$_.Name -match "second_gen_${letter}_\d+_z_-?\d+" } | Move-Item -Destination $folderPath
}