# Getting started
### Using a terminal
Clone this repository to your local machine using your tool of choice. Open the [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/) (requires a working [Anaconda](https://www.anaconda.com/) installation):

Then, use the prompt to **navigate to the location of the cloned repository**. Install the [environment](env_exported.yml) using the command:  
`conda env create -f env_exported.yml`

Follow the instructions to activate the new environment:  
`conda activate sea-ice-segment`

We have two environment files: 
- [env_exported](env_exported.yml): the environment exported from  Anaconda's history. This should be enough to replicate the results.
- [env_full](env_full.yml): the full environment installed. This includes more information and might be OS dependent. The experiments were executed using Windows 10 Pro for Workstations, mostly using version 21H2. 