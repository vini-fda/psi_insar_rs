#!/usr/bin/bash

ASF_USERNAME='GoogleColab2023'
ASF_PASSWORD='GoogleColab_2023'
# Reference Image
wget -c --http-user=$ASF_USERNAME --http-password=$ASF_PASSWORD "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151022T122539_20151022T122606_008265_00BA51_5A48/IW3/VV/2.zip"
# Secondary Image
#wget -c --http-user=$ASF_USERNAME --http-password=$ASF_PASSWORD "https://sentinel1-burst.asf.alaska.edu/S1A_IW_SLC__1SSV_20151010T122539_20151010T122603_008090_00B578_7501/IW3/VV/2.zip"
