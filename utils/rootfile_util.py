import uproot4

def rootTTree2numpy(rootFilePath):
    with uproot4.open(rootFilePath) as f:
        ttree = f['tauEDAnalyzer']['Events'] # TTree name: 'Events'
        return ttree.arrays(library="np")