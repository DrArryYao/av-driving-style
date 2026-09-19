# Data acquisition

## Waymo Open Motion Dataset v1.3
Register at the official Waymo Open Dataset site and accept the license.
Download the 20-second training scenario tfrecords (uncompressed
scenario format, 930 shards). Point AVS_DATA_DIR at a directory
containing them under tfrecords/.

## NGSIM US-101
Public dataset on the US DOT Socrata portal (dataset id 8ect-6jqj).
Example download (passenger cars):

curl -o $AVS_DATA_DIR/ngsim_us101_cars.csv \
  "https://data.transportation.gov/resource/8ect-6jqj.csv?%24limit=5000000&%24where=location%3D%27us-101%27%20AND%20v_Class%3D2"

Neither dataset may be redistributed with this code.
