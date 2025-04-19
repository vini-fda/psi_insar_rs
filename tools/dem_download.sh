#!/usr/bin/zsh

# Downloads a Copernicus GLO-90 DEM over the region on Mexico City
wget -O dem.tif "https://portal.opentopography.org/API/globaldem?demtype=COP90&south=19.33&north=19.63&west=-99.37&east=-98.59&outputFormat=GTiff&API_Key=876b662e4181042f1f15a5acbb9cc236"
