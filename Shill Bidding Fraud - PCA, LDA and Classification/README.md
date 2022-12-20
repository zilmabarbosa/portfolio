# About

This was my final integrated assignment for Data Preparation and Machine Learning disciplines. Its objective was to understand, explore and prepare the data, and choose between classification or clustering algorithms (supervised or unsupervised learning) to predict fraudulent bids at the online retailer eBay.

Specifically for Data Preparation, it was suggested to explore the possibility of dimensionally reducing the dataset, by applying both LDA (Linear Discriminant Analysis) and PCA (Principal Component Analysis).

For Machine Learning, we should provide a logical justification for the specific choice of machine learning approach (supervised or unsupervised learning). We were also asked to compare different algorithms and their results.

My approach was to do research to get a deeper understanding of the dataset first of all. After this, I decided on the machine learning approach and applied the chosen algorithms in different stages of the data preparation process in order to compare the results.

The Shill Bidding dataset can be found here:
https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset

# Data Understanding

The dataset object of this study was scrapped and pre-processed by Alzahrani and Sadaoui (2018b) in the report named “Scraping and Preprocessing Commercial Auction Data for Fraud Classification”, where they explain that “the unavailability of SB datasets makes the development of SB detection and classification models burdensome”.

Alzahrani and Sadaoui (2018b), scraped a large number of eBay auctions of a popular product (iPhone 7), for a period of three months in 2017, and pre-processed the raw data in order to build a high-quality SB dataset based on the most reliable SB strategies. It is important to mention this to understand the final results and evaluation.

Still according to Alzahrani and Sadaoui (2018b), “the original dataset contains irrelevant and redundant attributes, missing values and inappropriate value formatting”, which were already treated by the authors in order for them to apply clustering techniques and be able to classify the SB suspicious activities. Thus, those issues will probably not be observed in the dataset used for this current study.

Additionally, important information will be highlighted here: “The metrics are calculated from the auction dataset. Each metric is scaled to the range of [0,1]. High values refer to a suspicious bidding behaviour” (Alzahrani and Sadaoui, 2018b). This means that, except for the ID values and the duration of the auctions, all the other attributes are already scaled in the range previously mentioned.

Finally, the Record ID attribute was created to serve as a primary key, because Auction ID is repetitive as each auction may have more than one bid.

# References

Alzahrani, A. and Sadaoui, S. (2018a). Clustering and Labelling Auction Fraud Data. [online] Research Gate. Available at: https://www.researchgate.net/publication/327173391_Clustering_and_Labelling_Auction_Fraud_Data [Accessed 8 May 2022].

Alzahrani, A. and Sadaoui, S. (2018b). Scraping and Preprocessing Commercial Auction Data for Fraud Classification. [online] Research Gate. Available at: https://www.researchgate.net/publication/325157684_Scraping_and_Preprocessing_Commercial_Auction_Data_for_Fraud_Classification [Accessed 8 May 2022].
