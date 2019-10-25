##!/bin/bash

du -sh output &&
outname=results-packaged-$(date +%F).tar &&
echo "packaging 'output' to '$outname'" &&
tar cf $outname output &&
du -sh $outname
