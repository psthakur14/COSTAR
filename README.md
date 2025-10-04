# COSTAR is a technique that adopts the AST (Abstract Syntax Tree) representation to learn the structure of the source code.. 
To use the COSTAR technique, preprocess the dataset.

The dataset should be in a CSV file, where a "link" column contains the path of the source code, while the "severity" column contains the level of severity of the respective code. One more column, "smell," should be included in the CSV file to indicate the type of code smell.

Replace the name of the dataset or provide the path of the dataset in the COSTAR.py file. It will generate a new CSV file containing vectors corresponding to each given source code file. Change the name of the output file or specify the path where you want to save the final output.

Run the COSTAR.py file from the terminal using the command:
python COSTAR.py

**#For the model prediction**

The model_for_prediction.py file contains all the models used in the paper. You just need to change the name of the input dataset and run the model_for_prediction.py file.

**For the Comparision with state of the art**
The dictionary ComparisionWithStateOfTheArt contains implementations of baseline models from prior research on code smell detection and refactoring.
These implementations were re-created for comparison with our proposed method COSTAR.
