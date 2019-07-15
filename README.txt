You need to dowload some h5 files:

curl http://opendata.cern.ch/eos/opendata/cms/datascience/CNNPixelSeedsProducerTool/TTbar_13TeV_PU50_PixelSeeds/TTbar_PU50_pixelTracksDoublets_0_final.h5 -o TTbar_PU50_pixelTracksDoublets_0_final.h5

pandas 0.24 strictly needed because of a bug of pandas 0.23 in hd5 file reader

suggest to use anaconda python
