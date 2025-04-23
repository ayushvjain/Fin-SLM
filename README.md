## Financial SLM - Steps to run
**Steps**/
1. Clone the repository and go to the Demo branch using the following commands.
- `git clone https://github.com/ayushvjain/Fin-SLM.git`
- `git checkout Demo`

This will give all the code and the dataset to your local system.

2. The next step would be to install libraries. Run the following pip command to install all the required libraries
   `pip install -r requirements.txt`

3. Run the following commands step by step to run the complete code and get the output.
- `python model.py` - Running this command will train the model on the dataset that is given in the git repository.
- `python test_model.py --modelpath=<model_path> --prompt=<prompt>` - Running this will use the model to generate the answer based on your prompt given.
