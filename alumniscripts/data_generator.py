"""
This script will have methods/Threads that help create data for different other threads
1. offline_data_gen: This method is used to create data for off-line lstm model learning. It will 
execute the following step in order
    a. Get the raw data from multiple sources
    b. Clean the data, process(aggregation, smoothing etc) it
    c. Store the data in a suitable format  # maintain time stamp information
    Whenever lstm_train_data_lock is free, do in a never-ending loop
    d. read the stored file and create lstm related data for the last 3 months + 1 week
    e. read the stored file and create environment related data for last 3 months + 1 week over same period
"""