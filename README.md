# The wise greek lama
## Project background

This was a group project between Clara (https://github.com/ClaraLuAld), Lareen (https://github.com/lareengr) and myself. The goal was to create a llama based llm that could answer our questions in the style of the prosa it was delivered as context, in our case this the ancient greek and latin prosa Iliad, Odyssey and Aeneid. 

## How to set up:

We got our text files from here: \
The Iliad: https://www.gutenberg.org/cache/epub/22382/pg22382.txt \
The Odyssey: https://www.gutenberg.org/cache/epub/1727/pg1727.txt \
The Aeneid: https://www.gutenberg.org/cache/epub/228/pg228.txt 

Please download the .txt files to your local environment and add them to the folder structure like this:\

```
greek-llama/
├── content/                   # Data directory (gitignored)
│   ├── .gitkeep               # Placeholder to track empty directory
│   ├── data/                   
│   │   ├── data_aeneid/
│   │   │   └── pg228.txt      # Data of the Aeneid
│   │   ├── data_odyssey/
│   │   │   └── pg1727.txt     # Data of the Odyssey
│   │   ├── data_iliad/
│   │   │   └── pg22382.txt    # Data of the Iliad
│   │   ├── pg228.txt          # Data of the Aeneid combined
│   │   ├── pg1727.txt         # Data of the Odyssey combined
│   │   └── pg22382.txt        # Data of the Iliad combined
│   ├── vector_index/          # vector files for combined data
│   ├── vi_iliad/              # vector files for iliad data
│   ├── vi_aeneid/             # vector files for aeneid data 
│   └── vi_odyssey             # vector files for odyssey data
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── app.py                   # Package installation script
```

The app needs this folder structure to work properly. \
Have fun checking this out :)
