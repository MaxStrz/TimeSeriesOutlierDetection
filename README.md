# TimeSeriesOutlierDetection

HBOSagg: Histogram-Based Outlier Score Aggregator (HBOSAgg) takes the mean of the 
    outlier scores of a datapoint from multiple HBOS instances using different
    bin counts. 

    HBOSAgg implements an adapted version of pyod's HBOS class, enabling the
    use of dynamic bins. See HBOS_dynamic_bins in HBOSagg file for more information.
