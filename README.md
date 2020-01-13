To run the Sentiment Analysis, do the following:
1. Ensure the sentiment_analysis.py file is in the main directory, with the datasets in their respective folders also in the directory.
	1.1 Check the source code in sentiment_analysis.py if unsure on the paths for the dataset directories
2. At the bottom of the source code for sentiment_analysis.py there is a set of functions calls for the intermediate steps, comment or uncomment these as needed
3. Using the command-line at the main directory, run the command 'python sentiment_analysis.py'
4. An alternative would be to import the SentimentAnalysis class into a python program using 'from sentiment_analysis import SentimentAnalysis'
	4.1 Below is a list of method calls to be used after importing SentimentAnalysis to run it (it's also viewable in the source code of sentiment_analysis.py
	
	s = SentimentAnalysis()
    # OR LOAD A SERIALISED SentimentAnalysis OBJECT FROM FILE - 
    s = []
    with open('sentiment_object.txt', 'rb') as f:
        s = pickle.load(f)

	# Below are the method calls for training SVMs using the optimised hyperparameters, make sure they are called in order

    s.vectorize_input_data() <br/>
	s.train_svms() \n
	s.make_predictions() \n
	s.show_metrics() \n
	s.save_training_results() \n
	s.save_models() \n
	
	# Below are the method calls for hyperparameter experiments, it's possible to change in the source code values for hyperparameters
    s.optimise_pca() \n
	s.optimise_c() \n
	s.optimise_vocab_feature() \n
	s.optimise_tfidf() \n
	s.optimise_gamma() \n
	s.save_optimisation_results() \n

    # Save SentimentAnalysis object to a file
	with open('sentiment_object.txt', 'wb') as f: \n
		pickle.dump(s, f)
		aa
	